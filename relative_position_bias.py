# Deprecated relative position bias classes from commit 67ab3ea
import torch
import torch.nn as nn

class PositionalEmbedding2D(nn.Module):
    """
    Adds 2D positional embeddings to a flattened grid input.

    Assumes the input `x` is of shape (batch_size, seq_len, hidden_dim),
    where seq_len is height * width, and the flattening order is row-first.

    The positional embedding is the sum of a learnable row embedding and a
    learnable column embedding.
    """
    def __init__(self, hidden_dim: int, max_height: int=31, max_width: int=31):
        """
        Initializes the 2D positional embedding layer.

        Args:
            hidden_dim (int): The dimensionality of the embeddings and the input tensor.
            max_height (int): The maximum expected height of the 2D grid.
                               Used to size the row embedding table.
            max_width (int): The maximum expected width of the 2D grid.
                              Used to size the column embedding table.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_height = max_height
        self.max_width = max_width
        self.row_embed = nn.Embedding(max_height, hidden_dim)
        self.col_embed = nn.Embedding(max_width, hidden_dim)


    def reset_parameters(self,r,c):
        # r,c should be of shape (1, seq) on the right device
        self.r = r
        self.c = c

    def forward(self, module, input, output):
        """
        to match signature forward hook
        """
        # Look up embeddings for row and column indices
        # self.row_embed(rows) -> Shape: [seq_len, hidden_dim]
        # self.col_embed(cols) -> Shape: [seq_len, hidden_dim]
        row_pos_embedding = self.row_embed(self.r)
        col_pos_embedding = self.col_embed(self.c)

        return output + row_pos_embedding + col_pos_embedding
        # Need __call__ to delegate to forward for hook registration if using nn.Module
    def __call__(self, module, input, output):
        return self.forward(module, input, output)
    
class _RelPos2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bias_param, rows, cols, layer_idx):
        # save just the *bare minimum* for backward
        layers, H, max_h, max_w = bias_param.shape
        ctx.save_for_backward(rows, cols)
        ctx.layer_idx = layer_idx
        ctx.max_h = max_h
        ctx.max_w = max_w
        ctx.layers = layers
        ctx.H = H
        # do exactly the same fancy indexing
        h_idxs = torch.clamp(rows.unsqueeze(0) - rows.unsqueeze(1), 0, max_h-1)
        w_idxs = torch.clamp(cols.unsqueeze(0) - cols.unsqueeze(1), 0, max_w-1)
        out = bias_param[layer_idx:layer_idx+1, :, h_idxs, w_idxs]
        return out

    @staticmethod
    def backward(ctx, grad_out):
        rows, cols = ctx.saved_tensors
        layer_idx, max_h, max_w, layers, H = ctx.layer_idx, ctx.max_h, ctx.max_w, ctx.layers, ctx.H
        # rows and cols are 1-indexed, so it does not work with h_idxs * max_w + w_idxs
        rows -= 1
        cols -= 1
        # 1) recompute the 2D offsets
        h_idxs = torch.clamp(rows.unsqueeze(0) - rows.unsqueeze(1), 0, max_h-1)
        w_idxs = torch.clamp(cols.unsqueeze(0) - cols.unsqueeze(1), 0, max_w-1)

        # 2) flatten the 2D grid of offsets into single linear indices
        #    idx_flat[i,j] = h_idxs[i,j] * max_width + w_idxs[i,j]
        idx_flat = (h_idxs * max_w + w_idxs).view(-1)                  # (L*L,)

        # 3) flatten grad_out too
        g_flat = grad_out[0].view(H, -1)                               # (H, L*L)

        # 4) make a zero grad buffer for the entire param
        d_bias = torch.zeros(layers, H, max_h, max_w, device=grad_out.device, dtype=grad_out.dtype)  # (layers, H, max_h, max_w)

        # 5) we only write into layer layer_idx, so take a view:
        db_li = d_bias[layer_idx].view(H, -1)                                 # (H, max_h*max_w)

        # 6) scatter_add along dim=1 using our flat indices
        #    each head h adds g_flat[h,k] into db_li[h, idx_flat[k]]
        db_li.scatter_add_(1, idx_flat.unsqueeze(0).expand(H, -1), g_flat)

        return d_bias, None, None, None
        
class WithinGrid2DAttnScore(nn.Module):
    def __init__(self, layers, heads, max_height_delta, max_width_delta):
        super().__init__()
        self.layers = layers
        self.heads = heads
        self.max_height_delta = max_height_delta
        self.max_width_delta = max_width_delta
        # Parameter: (layers, heads, max_height_delta, max_width_delta)
        self.relative_position_bias = nn.Parameter(
            torch.empty(self.layers,
                        self.heads,
                        self.max_height_delta,
                        self.max_width_delta)
        )
        # Initialize parameters (e.g., small random values or zeros)
        nn.init.normal_(self.relative_position_bias, std=0.01)

    def forward(self, rows: torch.LongTensor, cols: torch.LongTensor, layer_idx: int):
        # Compute distance matrices (L, L)
        height_indices = (rows.unsqueeze(0) - rows.unsqueeze(1))  # (L, L)
        width_indices = (cols.unsqueeze(0) - cols.unsqueeze(1))   # (L, L)
        height_indices = torch.clamp(height_indices, 0, self.max_height_delta - 1)
        width_indices = torch.clamp(width_indices, 0, self.max_width_delta - 1)
        # Return shape (1, heads, L, L)
        if self.layers == 1:
            scores = self.relative_position_bias[:, :, height_indices, width_indices]
        else:
            scores = self.relative_position_bias[layer_idx:layer_idx+1, :, height_indices, width_indices]
        return scores
    
class AcrossGrid2DAttnScore(WithinGrid2DAttnScore):
    def __init__(self, layers, heads, max_height_delta, max_width_delta):
        super().__init__(layers, heads, max_height_delta, max_width_delta)

    def forward(self, rows1: torch.LongTensor, cols1: torch.LongTensor, rows2: torch.LongTensor, cols2: torch.LongTensor, layer_idx: int):
        '''
        rows1, cols1 are the coordinates of the input grid of shape (L,)
        rows2, cols2 are the coordinates of the output grid of shape (l,)
        returns a tensor of shape (1, heads, l, L), relative bias is left corner aligned, i.e. input (0,0) is aligned with output (0,0)
        '''
        # Compute distance matrices (l, L)
        height_indices = (rows2.unsqueeze(1) - rows1.unsqueeze(0))  # (l, L)
        width_indices = (cols2.unsqueeze(1) - cols1.unsqueeze(0))   # (l, L)

        # to map a delta of 0 to the middle of the parameter's dimension.
        height_center = (self.max_height_delta - 1) // 2
        width_center = (self.max_width_delta - 1) // 2
        height_indices += height_center
        width_indices += width_center
        height_indices = torch.clamp(height_indices, 0, self.max_height_delta - 1)
        width_indices = torch.clamp(width_indices, 0, self.max_width_delta - 1)

        # Return shape (1, heads, l, L)
        if self.layers == 1:
            scores = self.relative_position_bias[:, :, height_indices, width_indices]
        else:
            scores = self.relative_position_bias[layer_idx:layer_idx+1, :, height_indices, width_indices]

        # Causal mask: i > j -> -inf
        # causal_mask = torch.triu(torch.ones(L, L, device=rows.device), diagonal=1).bool()
        # scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), torch.finfo(torch.bfloat16).min)

        return scores

class MultiGridAttention(nn.Module):
    """
    A module that computes attention scores for multiple grids in a sequence.
    
    This module handles both within-grid attention (tokens attending to other tokens in the same grid)
    and across-grid attention (tokens in output grid attending to tokens in input grid).
    It's designed to work with 2D grid structures where each token has a row and column position.
    
    The module creates a full attention bias matrix that can be added to the standard attention scores
    in transformer models, enabling spatial awareness in the attention mechanism.
    
    Args:
        layers (int): Number of transformer layers.
        heads (int): Number of attention heads per layer.
        max_height_delta1 (int): Maximum height difference for within-grid attention.
        max_width_delta1 (int): Maximum width difference for within-grid attention.
        max_height_delta2 (int): Maximum height difference for across-grid attention.
        max_width_delta2 (int): Maximum width difference for across-grid attention.
    """
    def __init__(self, layers: int, heads: int, max_height_delta1: int, max_width_delta1: int, max_height_delta2: int, max_width_delta2: int):
        super().__init__()
        
        # Within-grid attention (same for all grids)
        self.within_grid_attn = WithinGrid2DAttnScore(
            layers, heads, max_height_delta1, max_width_delta1
        )
        
        # Across-grid attention (for y_i attending to x_i)
        self.across_grid_attn = AcrossGrid2DAttnScore(
            layers, heads, max_height_delta2, max_width_delta2
        )
        
        # State variables
        self.rows = None
        self.cols = None
        self.lengths = None
    
    def set_indices(self, rows, cols, lengths):
        """Set the row, column indices and lengths for attention computation."""
        self.rows = rows
        self.cols = cols
        self.lengths = lengths
    
    def forward(self, layer_idx):
        """
        Constructs a full attention score matrix combining within-grid and across-grid attention.
        
        Args:
            layer_idx: Int specifying which layer's attention scores to return
            
        Returns:
            attention_biases: Tensor of shape (1, heads, L, L)
        """
        rows, cols, lengths = self.rows, self.cols, self.lengths
        device = rows.device
        total_length = rows.size(0)
        heads = self.within_grid_attn.heads
        
        # Initialize attention bias matrix with zeros
        attention_biases = torch.zeros(
            (1, heads, total_length, total_length), 
            device=device
        )
        
        # Keep track of current position in the sequence
        current_pos = 0
        
        # Process each input-output pair
        # Calculate number of complete pairs and check if there's an unpaired input (decoding)
        num_complete_pairs = len(lengths) // 2
        has_unpaired_input = len(lengths) % 2 == 1
        
        # Process each input-output pair
        for pair_idx in range(num_complete_pairs):
            # Get indices for current input (x_i) and output (y_i)
            x_start = current_pos
            x_end = x_start + lengths[pair_idx * 2]
            y_start = x_end
            y_end = y_start + lengths[pair_idx * 2 + 1]
            
            # Extract coordinates for current input and output
            x_rows = rows[x_start:x_end]
            x_cols = cols[x_start:x_end]
            y_rows = rows[y_start:y_end]
            y_cols = cols[y_start:y_end]
            
            # 1. Within-grid attention for input (x_i to x_i)
            x_to_x_bias = self.within_grid_attn(x_rows, x_cols, layer_idx)
            attention_biases[:, :, x_start:x_end, x_start:x_end] = x_to_x_bias
            
            # 2. Within-grid attention for output (y_i to y_i)
            y_to_y_bias = self.within_grid_attn(y_rows, y_cols, layer_idx)
            attention_biases[:, :, y_start:y_end, y_start:y_end] = y_to_y_bias
            
            # 3. Across-grid attention for output attending to input (y_i to x_i)
            y_to_x_bias = self.across_grid_attn(x_rows, x_cols, y_rows, y_cols, layer_idx)
            attention_biases[:, :, y_start:y_end, x_start:x_end] = y_to_x_bias
            
            # Update current position
            current_pos = y_end
        
        # Handle the unpaired input if it exists
        if has_unpaired_input:
            # Get indices for the unpaired input
            x_start = current_pos
            x_end = x_start + lengths[-1]
            
            # Extract coordinates for unpaired input
            x_rows = rows[x_start:x_end]
            x_cols = cols[x_start:x_end]
            
            # Only apply within-grid attention for the unpaired input
            x_to_x_bias = self.within_grid_attn(x_rows, x_cols, layer_idx)
            attention_biases[:, :, x_start:x_end, x_start:x_end] = x_to_x_bias
            
            # Update current position
            current_pos = x_end
        
        # 1. Special token handling: zero out rows/cols where row value is 0 (special tokens)
        special_token_mask = (rows == 0)
        # Create mask for special tokens in attention matrix (both rows and columns)
        special_row_mask = special_token_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(
            1, heads, total_length, total_length
        )
        special_col_mask = special_token_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(
            1, heads, total_length, total_length
        )
            
        # Zero out attention scores for special tokens
        attention_biases.masked_fill_(special_row_mask, 0.0)
        attention_biases.masked_fill_(special_col_mask, 0.0)
        
        # 2. Apply causal mask to ensure past tokens don't attend to future tokens
        causal_mask = torch.triu(
            torch.ones(total_length, total_length, device=device), 
            diagonal=1
        ).bool()
        attention_biases.masked_fill_(
            causal_mask.unsqueeze(0).unsqueeze(0), 
            torch.finfo(torch.bfloat16).min  # Use -inf for bfloat16
        )
        
        return attention_biases

class _MultiGridAttn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                within_bias,    # (layers, H, mh1, mw1)
                across_bias,    # (layers, H, mh2, mw2)
                lengths,        # small Python list of ints
                layer_idx,      # int
                cached_indices, # dict of precomputed indices
                sp_mask,        # (L,L)
                causal,         # (L,L)
                attn           # (1, H, L, L)
               ):
        # unpack
        layers, H, mh1, mw1 = within_bias.shape
        _,      _, mh2, mw2 = across_bias.shape
        L = attn.shape[2]

        # Zero out the attention tensor
        # attn.zero_()

        Npairs = len(lengths)//2
        has_tail = (len(lengths) % 2 == 1)

        for i in range(Npairs):
            # Use cached indices
            hdx, wdx, xs, xe = cached_indices[f'x2x_{i}']
            attn[:, :, xs:xe, xs:xe] = within_bias[layer_idx:layer_idx+1, :, hdx, wdx] # += to avoid overwriting as we zeroed out the attention
            
            hdy, wdy, ys, ye = cached_indices[f'y2y_{i}']
            attn[:, :, ys:ye, ys:ye] = within_bias[layer_idx:layer_idx+1, :, hdy, wdy]
            
            hdxy, wdxy, ys, ye, xs, xe = cached_indices[f'y2x_{i}']
            attn[:, :, ys:ye, xs:xe] = across_bias[layer_idx:layer_idx+1, :, hdxy, wdxy]

        if has_tail:
            hdx, wdx, xs, xe = cached_indices['tail']
            attn[:, :, xs:xe, xs:xe] = within_bias[layer_idx:layer_idx+1, :, hdx, wdx]

        # --- special‐token zeroing ---
        attn.masked_fill_(sp_mask.view(1,1,L,L), 0.0)

        # --- causal mask ---
        attn.masked_fill_(causal.view(1,1,L,L), torch.finfo(attn.dtype).min)

        # Save input tensors for backward
        ctx.save_for_backward(within_bias, across_bias)  # Save original tensors
        ctx.lengths = lengths
        ctx.layer_idx = layer_idx
        ctx.shapes = (layers, H, mh1, mw1, mh2, mw2)
        ctx.cached_indices = cached_indices
        ctx.sp_mask = sp_mask
        ctx.causal = causal

        return attn

    @staticmethod
    def backward(ctx, grad_attn):
        within_bias, across_bias = ctx.saved_tensors
        lengths = ctx.lengths
        layer_idx = ctx.layer_idx
        layers, H, mh1, mw1, mh2, mw2 = ctx.shapes
        cached_indices = ctx.cached_indices
        sp_mask = ctx.sp_mask
        causal = ctx.causal
        L = grad_attn.shape[2]

        # re‐zero out masked positions so no grad leaks
        full_mask = (sp_mask | causal).view(1,1,L,L)
        grad_attn.masked_fill_(full_mask, 0.0)

        # Create empty grad tensors with same shape as inputs if they don't exist
        if within_bias.grad is None:
            within_bias.grad = torch.zeros_like(within_bias, device=within_bias.device, dtype=within_bias.dtype)
        if across_bias.grad is None:
            across_bias.grad = torch.zeros_like(across_bias, device=across_bias.device, dtype=across_bias.dtype)

        Npairs = len(lengths)//2
        has_tail = (len(lengths)%2==1)
        
        for i in range(Npairs):
            # Use cached indices for gradients
            hdx, wdx, xs, xe = cached_indices[f'x2x_{i}']
            g_xx = grad_attn[0, :, xs:xe, xs:xe].reshape(H, -1)
            idx = hdx.reshape(-1) * mw1 + wdx.reshape(-1)
            # Accumulate directly into .grad attribute
            within_bias.grad[layer_idx].view(H, -1).scatter_add_(1,
                                                         idx.unsqueeze(0).expand(H, -1),
                                                         g_xx.to(within_bias.dtype))
            
            hdy, wdy, ys, ye = cached_indices[f'y2y_{i}']
            g_yy = grad_attn[0, :, ys:ye, ys:ye].reshape(H, -1)
            idx = hdy.reshape(-1) * mw1 + wdy.reshape(-1)
            within_bias.grad[layer_idx].view(H, -1).scatter_add_(1,
                                                         idx.unsqueeze(0).expand(H, -1),
                                                         g_yy.to(within_bias.dtype))
            
            hdxy, wdxy, ys, ye, xs, xe = cached_indices[f'y2x_{i}']
            g_yx = grad_attn[0, :, ys:ye, xs:xe].reshape(H, -1)
            idx = hdxy.reshape(-1) * mw2 + wdxy.reshape(-1)
            across_bias.grad[layer_idx].view(H, -1).scatter_add_(1,
                                                         idx.unsqueeze(0).expand(H, -1),
                                                         g_yx.to(across_bias.dtype))

        if has_tail:
            hdx, wdx, xs, xe = cached_indices['tail']
            g_xx = grad_attn[0, :, xs:xe, xs:xe].reshape(H, -1)
            idx = hdx.reshape(-1) * mw1 + wdx.reshape(-1)
            within_bias.grad[layer_idx].view(H, -1).scatter_add_(1,
                                                         idx.unsqueeze(0).expand(H, -1),
                                                         g_xx.to(within_bias.dtype))

        # Return None for gradients as we've directly modified the .grad attributes
        return None, None, None, None, None, None, None, None,

class MultiGridAttention2(nn.Module):
    def __init__(self, layers: int, heads: int, max_height_delta1: int, max_width_delta1: int, max_height_delta2: int, max_width_delta2: int, device: str = 'cuda'):
        super().__init__()
        
        # Within-grid attention (same for all grids)
        self.within_grid_attn_bias = nn.Parameter(
                                    torch.empty(layers,
                                                heads,
                                                max_height_delta1,
                                                max_width_delta1)
                                )
        # Initialize with 2D multivariate normal centered at (0,0) with different std per head
        pattern = self._attention_init(
            heads=heads,
            max_height=max_height_delta1,
            max_width=max_width_delta1,
            start_std=1,
            end_std=20,
            is_centered=False,
            device=device
        )
        # Apply the same pattern to all layers
        self.within_grid_attn_bias.data = pattern.view(1, heads, max_height_delta1, max_width_delta1).repeat(layers, 1, 1, 1)
        
        # Across-grid attention (for y_i attending to x_i)
        self.across_grid_attn_bias = nn.Parameter(
                                    torch.empty(layers,
                                                heads,
                                                max_height_delta2,
                                                max_width_delta2)
                                )
        
        # Initialize with 2D multivariate normal centered at (max_height_delta2//2, max_width_delta2//2)
        pattern = self._attention_init(
            heads=heads,
            max_height=max_height_delta2,
            max_width=max_width_delta2,
            start_std=1,
            end_std=35,
            is_centered=True,
            device=device
        )
        
        # Apply the same pattern to all layers
        self.across_grid_attn_bias.data = pattern.view(1, heads, max_height_delta2, max_width_delta2).repeat(layers, 1, 1, 1) 

        self.triu_mask = torch.triu(torch.ones(12288, 12288, device=device, dtype=torch.bool), diagonal=1)
        # State variables
        self.rows = None
        self.cols = None
        self.lengths = None
        
        # Cache for computed indices
        self.cached_indices = None
        
        # Cache for masks
        self.sp_mask = None
        self.causal = None
        self.heads = heads
        self.device = device

    @staticmethod
    def _attention_init(heads, max_height, max_width, start_std=0.5, end_std=15, is_centered=False, device='cuda'):
        """init with a 2D multivariate normal pdf pattern."""
        # Create log-linear spaced std values between start_std and end_std
        std = start_std * (end_std/start_std) ** (torch.arange(heads, device=device).float() / (heads - 1)).view(heads, 1, 1)
        
        # Create coordinate grid
        y_coords = torch.arange(max_height, device=device).float().view(1, max_height, 1)
        x_coords = torch.arange(max_width, device=device).float().view(1, 1, max_width)
        
        if is_centered:
            center_h = max_height // 2
            center_w = max_width // 2
            squared_dist = ((y_coords - center_h) ** 2) + ((x_coords - center_w) ** 2)
        else:
            # Calculate distance from (0,0)
            squared_dist = (y_coords ** 2) + (x_coords ** 2)
            
        # Apply 2D normal distribution formula (without normalization constant)
        pattern = torch.exp(-squared_dist / (2 * std ** 2))
        
        return pattern
    
    def set_indices(self, rows, cols, lengths):
        """Set the row, column indices and lengths for attention computation."""
        self.rows = rows
        self.cols = cols
        self.lengths = lengths
        
        # Compute and cache indices
        del self.cached_indices
        self._compute_indices()
        
        # Compute and cache masks
        self._compute_masks()
        # Create attention tensor once
        self.attn = torch.zeros((1, self.heads, rows.shape[0], rows.shape[0]), device=self.device, dtype=torch.bfloat16)
    
    def _compute_masks(self):
        """Precompute special token mask and causal mask."""
        L = self.rows.shape[0]
        device = self.rows.device
        
        # Special token mask
        special = (self.rows == 0)  # (L,)
        self.sp_mask = special.unsqueeze(0) | special.unsqueeze(1)  # (L,L)
        
        # Causal mask
        self.causal = self.triu_mask[:L, :L]
    
    def _compute_indices(self):
        """Precompute all indices needed for attention computation."""
        mh1, mw1 = self.within_grid_attn_bias.shape[2:]
        mh2, mw2 = self.across_grid_attn_bias.shape[2:]
        
        # Precompute centers for across
        hc2 = (mh2 - 1) // 2
        wc2 = (mw2 - 1) // 2
        
        indices = {}
        cur = 0
        Npairs = len(self.lengths)//2
        has_tail = (len(self.lengths) % 2 == 1)
        
        for i in range(Npairs):
            lx, ly = self.lengths[2*i], self.lengths[2*i+1]
            xs, xe = cur, cur + lx
            ys, ye = xe, xe + ly
            
            x_rows, x_cols = self.rows[xs:xe], self.cols[xs:xe]
            y_rows, y_cols = self.rows[ys:ye], self.cols[ys:ye]
            
            # Within x→x indices
            hdx = torch.clamp(x_rows.unsqueeze(0) - x_rows.unsqueeze(1), 0, mh1-1)
            wdx = torch.clamp(x_cols.unsqueeze(0) - x_cols.unsqueeze(1), 0, mw1-1)
            indices[f'x2x_{i}'] = (hdx, wdx, xs, xe)
            
            # Within y→y indices
            hdy = torch.clamp(y_rows.unsqueeze(0) - y_rows.unsqueeze(1), 0, mh1-1)
            wdy = torch.clamp(y_cols.unsqueeze(0) - y_cols.unsqueeze(1), 0, mw1-1)
            indices[f'y2y_{i}'] = (hdy, wdy, ys, ye)
            
            # Across y→x indices
            hdxy = (y_rows.unsqueeze(1) - x_rows.unsqueeze(0)) + hc2
            wdxy = (y_cols.unsqueeze(1) - x_cols.unsqueeze(0)) + wc2
            hdxy = torch.clamp(hdxy, 0, mh2-1)
            wdxy = torch.clamp(wdxy, 0, mw2-1)
            indices[f'y2x_{i}'] = (hdxy, wdxy, ys, ye, xs, xe)
            
            cur = ye
            
        if has_tail:
            lx = self.lengths[-1]
            xs, xe = cur, cur + lx
            x_rows, x_cols = self.rows[xs:xe], self.cols[xs:xe]
            hdx = torch.clamp(x_rows.unsqueeze(0) - x_rows.unsqueeze(1), 0, mh1-1)
            wdx = torch.clamp(x_cols.unsqueeze(0) - x_cols.unsqueeze(1), 0, mw1-1)
            indices['tail'] = (hdx, wdx, xs, xe)
            
        self.cached_indices = indices

    def forward(self, layer_idx):
        # rows, cols, lengths are already stored as tensors / python list
        return _MultiGridAttn.apply(
            self.within_grid_attn_bias,
            self.across_grid_attn_bias,
            self.lengths, layer_idx,
            self.cached_indices,
            self.sp_mask,
            self.causal,
            self.attn
        )

def get_gemma_model(model_name, head_dim, isTrain, NeedPosition, saved_path=None, max_seq_length = 8192):
    model, _ = FastModel.from_pretrained(model_name = model_name,
                                         max_seq_length = max_seq_length,
                                         load_in_4bit = True,
                                         resize_model_vocab=head_dim,
                                        )
    try:
        del model.vision_tower
        model = model.base_model
    except:
        print("Not a vision language model")
    model.model.embed_tokens.padding_idx = None # otherwise token zero will be ignored
    if NeedPosition:
        embedding_layer = model.get_input_embeddings()
        PosEmbedModel = PositionalEmbedding2D(embedding_layer.weight.shape[1]).to(model.device)
        if saved_path is not None:
            PosEmbedModel.load_state_dict(torch.load(saved_path + 'PosEmbedModel.pth'))
        embedding_layer.register_forward_hook(PosEmbedModel);
    if isTrain:
        model.train();
        model.lm_head.weight.requires_grad_(True);
    gc.collect()
    torch.cuda.empty_cache()
    if saved_path is not None:
        model.lm_head.load_state_dict(torch.load(saved_path + 'lm_heads_weights.pth'))
        model = PeftModel.from_pretrained(model, saved_path + 'finetuned_model', is_trainable=isTrain)
    else:
        # start from pretrained model
        model.lm_head.load_state_dict(torch.load(f"/home/zhenlan/Desktop/Projects/ARC2/Model/gemma{head_dim}.pth"))
    if not isTrain:
        FastLanguageModel.for_inference(model);
    if NeedPosition:
        return model, PosEmbedModel
    else:
        return model

'''  ----------------------------------- Dataset Transformation utilities ------------------------------------- '''
@dataclass
class TransformPara:
    """Parameters for grid transformations"""
    fliplr: int          # Flip left-right (0 or 1)
    rot90: int           # Number of 90-degree rotations (0 to 3)
    perm_color: np.ndarray  # Color permutation array
