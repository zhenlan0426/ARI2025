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
