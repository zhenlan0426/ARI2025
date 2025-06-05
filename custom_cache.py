from transformers.cache_utils import Cache
import torch
from typing import Optional, Dict, Any, Tuple, List


class LastPositionCache(Cache):
    """
    A custom cache that returns the concatenation of past KV and current KV states,
    but only stores the last position of the current KV states in its internal cache.
    """
    
    def __init__(self, last_k=1):
        super().__init__()
        self.key_cache: List[Optional[torch.Tensor]] = []
        self.value_cache: List[Optional[torch.Tensor]] = []
        self._seq_length = 0  # Track total sequence length seen so far
        self.last_k = last_k

    def update(
        self, 
        key_states: torch.Tensor, 
        value_states: torch.Tensor, 
        layer_idx: int, 
        cache_kwargs: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with new key and value states.
        
        Args:
            key_states: New key states of shape [batch_size, num_heads, seq_len, head_dim]
            value_states: New value states of shape [batch_size, num_heads, seq_len, head_dim]
            layer_idx: Layer index
            cache_kwargs: Additional cache arguments (unused)
            
        Returns:
            Tuple of (concatenated_keys, concatenated_values) for attention computation
        """
        # Update sequence length only for the first layer (to avoid counting multiple times)
        if layer_idx == 0:
            self._seq_length += key_states.shape[-2]
        
        # Ensure we have enough layers in our cache
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)
        
        # Get current cache for this layer
        cached_keys = self.key_cache[layer_idx]
        cached_values = self.value_cache[layer_idx]
        
        # Concatenate past cache with current states for return
        if cached_keys is not None:
            # Return concatenated past + current for attention computation
            full_keys = torch.cat([cached_keys, key_states], dim=-2)
            full_values = torch.cat([cached_values, value_states], dim=-2)
        else:
            # First time for this layer, no past to concatenate
            full_keys = key_states
            full_values = value_states
        
        # Store only the LAST position of current KV in cache
        # Extract the last position from current key_states and value_states
        self.key_cache[layer_idx] = key_states[..., -self.last_k:, :]  # Shape: [batch, heads, 1, head_dim]
        self.value_cache[layer_idx] = value_states[..., -self.last_k:, :]  # Shape: [batch, heads, 1, head_dim]
        
        return full_keys, full_values

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states."""
        return self._seq_length
    
    # def get_max_cache_shape(self) -> Optional[int]:
    #     """Returns the maximum sequence length of the cache object."""
    #     # Since we're only storing the last position, there's no inherent max limit
    #     return None


class ChunkedLastPositionCache(Cache):
    """
    A custom cache that keeps the last `last_k` positions from the most recent `max_chunk` chunks.
    Uses pre-allocated tensors for efficiency.
    
    Example with max_chunk=2, last_k=1:
    - After chunk (1,2,3): cache stores [3]
    - After chunk (4,5,6): cache stores [3,6] 
    - After chunk (7,8,9): cache stores [6,9] (drops first chunk, keeps last 2)
    """
    
    def __init__(self, max_chunk=2, last_k=1):
        super().__init__()
        self.key_cache: List[Optional[torch.Tensor]] = []  # Pre-allocated tensors per layer
        self.value_cache: List[Optional[torch.Tensor]] = []  # Pre-allocated tensors per layer
        self._seq_length = 0  # Track total sequence length seen so far
        self.max_chunk = max_chunk
        self.last_k = last_k
        self.cache_size = max_chunk * last_k  # Total cache size per layer
        
        # Track state per layer
        self.current_positions: List[int] = []  # Current write position in circular buffer
        self.filled_chunks: List[int] = []  # Number of chunks actually filled

    def _initialize_layer_cache(self, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor):
        """Initialize pre-allocated cache tensors for a new layer."""
        batch_size, num_heads, _, head_dim = key_states.shape
        
        # Pre-allocate cache tensors
        key_cache = torch.zeros(
            batch_size, num_heads, self.cache_size, head_dim,
            dtype=key_states.dtype, device=key_states.device
        )
        value_cache = torch.zeros(
            batch_size, num_heads, self.cache_size, head_dim,
            dtype=value_states.dtype, device=value_states.device
        )
        
        # Extend lists to accommodate this layer
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)
            self.current_positions.append(0)
            self.filled_chunks.append(0)
        
        self.key_cache[layer_idx] = key_cache
        self.value_cache[layer_idx] = value_cache

    def update(
        self, 
        key_states: torch.Tensor, 
        value_states: torch.Tensor, 
        layer_idx: int, 
        cache_kwargs: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with new key and value states.
        
        Args:
            key_states: New key states of shape [batch_size, num_heads, seq_len, head_dim]
            value_states: New value states of shape [batch_size, num_heads, seq_len, head_dim]
            layer_idx: Layer index
            cache_kwargs: Additional cache arguments (unused)
            
        Returns:
            Tuple of (concatenated_keys, concatenated_values) for attention computation
        """
        # Update sequence length only for the first layer (to avoid counting multiple times)
        if layer_idx == 0:
            self._seq_length += key_states.shape[-2]
        
        # Initialize cache for this layer if needed
        if layer_idx >= len(self.key_cache) or self.key_cache[layer_idx] is None:
            self._initialize_layer_cache(layer_idx, key_states, value_states)
        
        # Get cache tensors for this layer
        cached_keys = self.key_cache[layer_idx]
        cached_values = self.value_cache[layer_idx]
        current_pos = self.current_positions[layer_idx]
        filled_chunks = self.filled_chunks[layer_idx]
        
        # First, build the current cache state for attention computation (BEFORE updating)
        if filled_chunks > 0:
            # Get the valid portion of the cache - order doesn't matter since RoPE is baked in
            if filled_chunks < self.max_chunk:
                # Not full yet, just use the first filled_chunks * last_k positions
                valid_cache_size = filled_chunks * self.last_k
                valid_cached_keys = cached_keys[..., :valid_cache_size, :]
                valid_cached_values = cached_values[..., :valid_cache_size, :]
            else:
                # Buffer is full, use all cache_size positions (order doesn't matter)
                valid_cached_keys = cached_keys[..., :self.cache_size, :]
                valid_cached_values = cached_values[..., :self.cache_size, :]
            
            # Return concatenated past + current for attention computation
            full_keys = torch.cat([valid_cached_keys, key_states], dim=-2)
            full_values = torch.cat([valid_cached_values, value_states], dim=-2)
        else:
            # First time for this layer, no past to concatenate
            full_keys = key_states
            full_values = value_states
        
        # Now update the cache with new values
        # Extract the last `last_k` positions from the current chunk
        # Handle edge case where chunk size < last_k
        actual_k = min(self.last_k, key_states.shape[-2])
        chunk_keys = key_states[..., -actual_k:, :]
        chunk_values = value_states[..., -actual_k:, :]
        
        # Write to cache at current position
        start_idx = current_pos * self.last_k
        end_idx = start_idx + actual_k  # Only write what we actually have
        cached_keys[..., start_idx:start_idx + actual_k, :] = chunk_keys
        cached_values[..., start_idx:start_idx + actual_k, :] = chunk_values
        
        # Update position and chunk count
        self.current_positions[layer_idx] = (current_pos + 1) % self.max_chunk
        self.filled_chunks[layer_idx] = min(filled_chunks + 1, self.max_chunk)
        
        return full_keys, full_values

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states."""
        return self._seq_length
