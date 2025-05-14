from dataclasses import dataclass
from unsloth import FastModel,FastLanguageModel
import numpy as np
import random
import torch
import torch.nn as nn
from transformers import StaticCache
from typing import List, Tuple, Optional
import gc
import os
import shutil
import time
import re
import json
from dataclasses import asdict
from dataclasses import dataclass, field
from peft import PeftModel
import math

''' ---------------------------- Model utilities ----------------------------- '''
@dataclass
class GlobalConfig:
    """Configuration class for model training and data processing."""
    model_name: str
    r: int # peft
    data_path: str
    tokenization: str
    max_length: int
    autoregressive: bool
    epochs: int
    NeedPosition: bool

    @staticmethod
    def find_largest_version():
        # Get current directory
        current_dir = os.getcwd()
        largest_number = -1  # Initialize to -1 in case no valid files are found
        # Regular expression to match numbers at the end of filename before .ipynb
        pattern = r'(\d+)\.ipynb$'
        # Iterate through files in current directory
        for filename in os.listdir(current_dir):
            # Check if file is .ipynb and matches pattern
            match = re.search(pattern, filename)
            if match:
                # Extract number and convert to integer
                number = int(match.group(1))
                # Update largest number if current is larger
                largest_number = max(largest_number, number)
        return str(largest_number)
    
    def __post_init__(self):
        if self.tokenization not in ('causal', 'oneshot'):
            raise ValueError(f"Invalid tokenization: {self.tokenization}. Must be 'causal' or 'oneshot'.")
        save_path = '../../Model/model_' + self.find_largest_version()
        # if os.path.exists(save_path):
        #     shutil.rmtree(save_path)
        #     print(f"Deleted folder and contents: {save_path}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print(f"Created folder: {save_path}")
        self.folder = save_path + '/'
        if self.tokenization == 'causal':
            self.tokenizer = tokenize_causal
            self.decoder = CausalDecoder
            self.lm_head_dim = 16
        else:
            self.tokenizer = tokenize_oneshot
            self.decoder = OneshotDecoder
            self.lm_head_dim = 48
    
    def save_to_json(self) -> None:
        """Save the dataclass instance to a JSON file."""
        with open(self.folder + 'globalConfig.json', 'w') as json_file:
            json.dump(asdict(self), json_file, indent=4)

    @classmethod
    def load_from_json(cls, file_path: str):
        """Load a dataclass instance from a JSON file."""
        with open(file_path + 'globalConfig.json', 'r') as json_file:
            data = json.load(json_file)
        return cls(**data)


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
            end_std=20,
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
    perm_example: np.ndarray  # Example permutation array
    # enlarge: tuple[int, int]  # Enlarge factors (n, m). disabled due to longer input length performance issue
    apply_to_output: int     # Apply transformations to y or not (except color)

def enlarge_grid_n_times(grid, n, m):
    """Enlarges a grid by repeating each element n times horizontally and m times vertically."""
    if n == 1 and m == 1:
        return grid
    new_grid = []
    for row in grid:
        new_row = []
        for element in row:
            new_row += [element] * n
        new_grid += [new_row] * m
    return new_grid

def shrink_grid_n_times(enlarged_grid, n, m):
    """Shrinks a grid by taking every nth element horizontally and mth element vertically."""
    if n == 1 and m == 1:
        return enlarged_grid
    original_height = len(enlarged_grid) // m
    original_width = len(enlarged_grid[0]) // n if enlarged_grid else 0
    original_grid = []
    for i in range(original_height):
        row = enlarged_grid[i * m]
        original_row = [row[j * n] for j in range(original_width)]
        original_grid.append(original_row)
    return original_grid

def generateTransformPara(n):
    """Randomly generates transformation parameters"""
    # n is the number of examples
    # (fliplr, rot90, permutate color, permutate example, enlarge, apply to output)
    return TransformPara(np.random.randint(0, 2), np.random.randint(0, 4), np.random.permutation(10), \
                         np.random.permutation(n),\
                        #  (np.random.randint(1, 3), np.random.randint(1, 3)),\
                         np.random.randint(0, 2))

def forward(x, tpara:TransformPara):
    """Applies transformations to a single grid."""
    if tpara.fliplr:
        x = np.fliplr(x)
    x = np.rot90(x, k=tpara.rot90)
    x = tpara.perm_color[x]
    # n, m = tpara.enlarge
    # x = enlarge_grid_n_times(x, n, m)
    return x
    
def backward(x, tpara:TransformPara):
    """Reverses transformations for a single grid."""
    # n, m = tpara.enlarge
    # x = shrink_grid_n_times(x, n, m)
    inv_perm = np.argsort(tpara.perm_color)  # Compute inverse permutation
    x = inv_perm[x]
    x = np.rot90(x, k=4-tpara.rot90)
    if tpara.fliplr:
        x = np.fliplr(x)
    return x

def forwardTask(task, tpara: TransformPara):
    """Applies transformations to a list of [(x1, y1), ...] examples."""
    task_out = []
    for i in tpara.perm_example:
        x, y = task[i]
        # Always apply all transformations to x
        x_transformed = forward(x, tpara)
        
        # Decide how to transform y based on apply_to_output
        if tpara.apply_to_output:
            y_transformed = forward(y, tpara)
        else:
            y_transformed = tpara.perm_color[y]
        
        task_out.append((x_transformed, y_transformed))
    return task_out

def backwardTask(task, tpara):
    """Reverses transformations for a list of [(x1, y1), ...] examples."""
    # Currently disabled - uncomment to use
    # return [(backward(x, tpara), backward(y, tpara)) for x, y in task]
    pass

'''  ----------------------------------- Tokenization utilities ------------------------------------- '''
def numpy2torch(x):
    """Convert numpy array to torch tensor and move to GPU"""
    x = torch.tensor(x)[None].to('cuda')
    return x

def find_first_exceed(task, max_len, extra_tokens=4):
    # 4 for BOS_X, row, col, EOS_X in addtion to elements in input or output
    # return the first task index that exceeds max_len, task[index-1] is the last task
    total = 0
    for i, (input_i, output_i) in enumerate(task, start=1):
        size_input = len(input_i) * len(input_i[0])
        size_output = len(output_i) * len(output_i[0])
        total += size_input + size_output + extra_tokens * 2
        if total > max_len:
            return i
    return len(task)  # If total never exceeds max_len

def tokenize_causal(task, autoregressive: bool, max_length, IsDecode=False, NeedPosition: bool = False, ReturnLengths: bool = False):
    """
    Tokenizes a task for causal (autoregressive) training or inference,
    optionally providing 2D positional indices using optimized list extensions.

    Args:
        task: List of (input_grid, output_grid) tuples.
              Each grid is a 2D list of integers.
              For decoding, the last output_grid can be None.
        autoregressive: Whether to use autoregressive training mode.
        max_length: Maximum sequence length for truncation.
        IsDecode: Whether the function is being used for inference (True) or training (False).
        NeedPosition: If True, return row and column indices for 2D position embedding.
        ReturnLengths: If True, return a list of lengths for each input and output grid.

    Returns:
        If NeedPosition is False and ReturnLengths is False:
            input_tokens: Numpy array of input token IDs.
            final_target: Numpy array of shifted target token IDs (training) or raw grid (decoding).
        If NeedPosition is True or ReturnLengths is True:
            Dictionary containing requested outputs:
            - "input_tokens": Numpy array of input token IDs.
            - "target_tokens": Same as above.
            - "row_indices": Numpy array of row indices (if NeedPosition is True).
            - "col_indices": Numpy array of column indices (if NeedPosition is True).
            - "lengths": List of lengths for each input and output grid (if ReturnLengths is True).
    """
    # Special token IDs
    BOS_X = 10  # Beginning of input grid
    EOS_X = 11  # End of input grid
    LINE_BREAK = 12  # Row separator
    BOS_Y = 13  # Beginning of output grid
    EOS_Y = 14  # End of output grid
    PAD_TOKEN = -100  # Padding/ignored token

    input_tokens = []
    target_tokens = []
    if NeedPosition:
        row_indices = []
        col_indices = []
    if ReturnLengths:
        lengths = []
    
    flag = not IsDecode and not autoregressive
    n_task = find_first_exceed(task, max_length)
    if IsDecode:
        # For decoding, must include the last task
        task = task[:n_task-1] + [task[-1]] 
    else:
        task = task[:n_task]
    n = len(task)
    for i, (x, y) in enumerate(task):
        IsLast = (i == n-1) and IsDecode
        
        # Track starting position for length calculation
        if ReturnLengths:
            input_start_pos = len(input_tokens)
        
        # Process input grid (x)
        input_tokens.append(BOS_X)
        if flag:
            target_tokens.append(PAD_TOKEN)
        if NeedPosition:
            row_indices.append(0)
            col_indices.append(0)
            
        for r_idx, row in enumerate(x):
            # Add row elements
            input_tokens.extend(row)
            if flag:
                target_tokens.extend([PAD_TOKEN]*len(row))
            if NeedPosition:
                row_len = len(row)
                row_indices.extend([r_idx + 1] * row_len)
                col_indices.extend(list(range(1, row_len + 1)))

            input_tokens.append(LINE_BREAK)
            if NeedPosition:
                row_indices.append(0)
                col_indices.append(0)
            if flag:
                target_tokens.append(PAD_TOKEN)

        input_tokens.append(EOS_X)
        if flag:
            target_tokens.append(PAD_TOKEN)
        if NeedPosition:
            row_indices.append(0)
            col_indices.append(0)
        
        # Record input length if requested
        if ReturnLengths:
            output_start_pos = len(input_tokens)
            input_length = output_start_pos - input_start_pos
            lengths.append(input_length)
            

        # Process output grid (y)
        input_tokens.append(BOS_Y)
        if flag:
            target_tokens.append(PAD_TOKEN)  # Mask BOS_Y
        if NeedPosition:
            row_indices.append(0)
            col_indices.append(0)

        if not IsLast:
            for r_idx, row in enumerate(y):
                # Add row elements
                input_tokens.extend(row)
                if flag:
                    target_tokens.extend(row)  # Keep y values in target
                if NeedPosition:
                    # Extend position indices for the row
                    row_len = len(row)
                    row_indices.extend([r_idx + 1] * row_len)
                    col_indices.extend(list(range(1, row_len + 1)))

                input_tokens.append(LINE_BREAK)
                if NeedPosition:
                    row_indices.append(0)
                    col_indices.append(0)                
                if flag:
                    target_tokens.append(LINE_BREAK)

            input_tokens.append(EOS_Y)
            if NeedPosition:
                row_indices.append(0)
                col_indices.append(0)
            if flag:
                target_tokens.append(EOS_Y)  # Include EOS_Y in target
            
            # Record output length if requested
            if ReturnLengths:
                output_length = len(input_tokens) - output_start_pos
                lengths.append(output_length)
        else:
            target_tokens = y  # For the last example in decode mode, we don't add output length
        
    # Create shifted targets (for next-token prediction)
    if not IsDecode:
        if autoregressive:
            target_tokens = input_tokens[1:] + [PAD_TOKEN]
        else:
            target_tokens = target_tokens[1:] + [PAD_TOKEN]
    
    # Convert to numpy arrays
    out = dict()
    out["input_tokens"] = numpy2torch(input_tokens)
    if IsDecode:
        out["target_tokens"] = np.array(target_tokens) if target_tokens is not None else None
    else:
        out["target_tokens"] = numpy2torch(target_tokens)
    if NeedPosition:
        out["row_indices"] = numpy2torch(row_indices)[0]
        out["col_indices"] = numpy2torch(col_indices)[0]
    if ReturnLengths:
        out["lengths"] = lengths
    return out

def tokenize_oneshot(task:list[tuple[list[list[int]], list[list[int]]]], \
                     max_length:int,\
                     IsDecode:bool, autoregressive:bool, NeedPosition:bool=False):
    """
    Tokenizes one-shot prediction for ARC tasks.

    Args:
        task (list): A list of tuples, where each tuple represents an example and
                     contains (input_grid, output_grid). The last tuple might be
                     (test_input_grid, None) during decoding or
                     (test_input_grid, test_output_grid) during training/evaluation.
                     Each grid is a list of lists of integers (0-9).
        max_length (int): Maximum total sequence length (tokens) to consider from
                          the examples. Examples exceeding this length when concatenated
                          might be partially dropped.
        IsDecode (bool): If True, tokenizes only up to the point of predicting the
                         output row dimension for the *last* task item. If False,
                         tokenizes for training, including target sequences.
        autoregressive (bool): If True and IsDecode is False, generates targets for
                               predicting *all* tokens autoregressively (shifted input).
                               If False and IsDecode is False, generates targets only
                               for the final output grid prediction (one-shot), masking
                               other parts with IGNORE_INDEX. This flag is ignored if
                               IsDecode is True.
        NeedPosition (bool, optional): If True, calculates and returns positional
                                       indices (row, column) for each input token.
                                       Defaults to False.

    Returns:
        A tuple whose contents depend on IsDecode and NeedPosition:

        If IsDecode=True:
            If NeedPosition=False:
                - input_tokens (np.array): Input token sequence for the final task,
                                           ready for starting the decoding process.
                - output_grid (np.array or None): The ground truth output grid for the
                                                  final task, if provided; otherwise None.
            If NeedPosition=True:
                - input_tokens (np.array): Input token sequence.
                - output_grid (np.array or None): The ground truth output grid.
                - row_indices (np.array): Row indices corresponding to input_tokens.
                - col_indices (np.array): Column indices corresponding to input_tokens.

        If IsDecode=False:
            If NeedPosition=False:
                - input_tokens (np.array): Full input token sequence including demonstration
                                           examples and the final task's input setup
                                           (with PREDICT_CELL_Y placeholders).
                - target_tokens (np.array): Target token sequence for training. Contains
                                            IGNORE_INDEX for non-target positions.
                - oneshot_target_idx (int): The index in `target_tokens` from which the
                                            actual one-shot output grid cell targets begin.
                                            (Corresponds to the first PREDICT_CELL_Y
                                             in input_tokens).
            If NeedPosition=True:
                - input_tokens (np.array): Full input token sequence.
                - target_tokens (np.array): Target token sequence.
                - oneshot_target_idx (int): Index where one-shot targets begin.
                - row_indices (np.array): Row indices corresponding to input_tokens.
                - col_indices (np.array): Column indices corresponding to input_tokens.
    """
    # Token definitions with direct values
    # 0-9: Grid cell values (digits)
    BOS_X = 10       # Beginning of input grid
    EOS_X = 11       # End of input grid
    BOS_Y = 12       # Beginning of output grid
    EOS_Y = 13       # End of output grid
    PREDICT_ROW_Y = 14  
    PREDICT_COL_Y = 15  
    PREDICT_CELL_Y = 16  # Placeholder for predicting output cells
    
    # Dimension tokens (SIZE_1 to SIZE_30 map to 17 to 46)
    SIZE_TOKEN_OFFSET = 16

    def get_grid_dimensions(grid):
        """Extract dimensions and flatten a grid."""
        num_rows = len(grid)
        num_cols = len(grid[0])
        flat_grid = [cell for row in grid for cell in row]    
        return num_rows, num_cols, flat_grid
    
    def get_grid_dimensions_extend(grid):
        num_rows = len(grid)
        num_cols = len(grid[0])
        cell_values = []
        row_indices = []
        col_indices = []
        col_index_pattern = list(range(1, num_cols + 1))
        for r_idx, row in enumerate(grid):
            cell_values.extend(row)
            row_indices.extend([r_idx + 1] * num_cols)
            col_indices.extend(col_index_pattern)
        return num_rows, num_cols, cell_values, row_indices, col_indices
    
    def get_dimension_token(dim_size):
        """Gets the token ID for a given dimension size."""
        return SIZE_TOKEN_OFFSET + dim_size

    get_dimention = get_grid_dimensions if not NeedPosition else get_grid_dimensions_extend
    IGNORE_INDEX = -100
    input_tokens = []
    if not IsDecode:
        target_tokens = []
    if NeedPosition:
        row_indices = []
        col_indices = []
    n_task = find_first_exceed(task, max_length)
    for input_grid, output_grid in task[:n_task-1]:
        # --- Validate and Flatten Input Grid (X) ---
        if NeedPosition:
            rows_x, cols_x, flat_x, row_indices_x, col_indices_x = get_dimention(input_grid)
        else:
            rows_x, cols_x, flat_x = get_dimention(input_grid)
        row_token_x = get_dimension_token(rows_x)
        col_token_x = get_dimension_token(cols_x)

        # --- Validate and Flatten Output Grid (Y) ---
        if NeedPosition:
            rows_y, cols_y, flat_y, row_indices_y, col_indices_y = get_dimention(output_grid)
        else:
            rows_y, cols_y, flat_y = get_dimention(output_grid)
        row_token_y = get_dimension_token(rows_y) # Actual target token for rows_y
        col_token_y = get_dimension_token(cols_y) # Actual target token for cols_y
        num_output_cells = rows_y * cols_y

        # --- Construct Model input Sequence ---
        # append the input grid
        input_tokens.append(BOS_X)
        input_tokens.append(row_token_x)
        input_tokens.append(col_token_x)
        if NeedPosition:
            row_indices.extend([0] * 3)
            col_indices.extend([0] * 3)
        input_tokens.extend(flat_x) # Add flattened input grid cells (as ints 0-9)
        if NeedPosition:
            row_indices.extend(row_indices_x)
            col_indices.extend(col_indices_x)
        input_tokens.append(EOS_X)
        if NeedPosition:
            row_indices.append(0)
            col_indices.append(0)
        
        # append the output grid
        input_tokens.append(PREDICT_ROW_Y)
        input_tokens.append(row_token_y)
        input_tokens.append(PREDICT_COL_Y)
        input_tokens.append(col_token_y)
        input_tokens.append(BOS_Y)
        if NeedPosition:
            row_indices.extend([0] * 5)
            col_indices.extend([0] * 5)
        input_tokens.extend(flat_y)
        if NeedPosition:
            row_indices.extend(row_indices_y)
            col_indices.extend(col_indices_y)
        input_tokens.append(EOS_Y)
        if NeedPosition:
            row_indices.append(0)
            col_indices.append(0)
        
        # --- Construct Model Target Sequence ---
        if not IsDecode:
            if autoregressive:
                # shifted input
                target_tokens.append(row_token_x)
                target_tokens.append(col_token_x)
                target_tokens.extend(flat_x) # Add flattened input grid cells (as ints 0-9)
                target_tokens.append(EOS_X)
                
                # append the output grid, no need to train on special token as at inference time, they will be manually added
                target_tokens.append(-100)
                target_tokens.append(row_token_y)
                target_tokens.append(-100)
                target_tokens.append(col_token_y)
                target_tokens.append(-100)
                target_tokens.extend(flat_y)
                target_tokens.append(EOS_Y)
                target_tokens.append(IGNORE_INDEX)
            else:
                target_tokens.extend([IGNORE_INDEX] * (len(input_tokens) - len(target_tokens)))

    # --- Construct Model Input and Target Sequence for the last task ---
    # if IsDecode is true, use the last task as the input and output
    input_grid, output_grid = task[n_task-1] if not IsDecode else task[-1]
    if NeedPosition:
        rows_x, cols_x, flat_x, row_indices_x, col_indices_x = get_dimention(input_grid)
    else:
        rows_x, cols_x, flat_x = get_dimention(input_grid)
    row_token_x = get_dimension_token(rows_x)
    col_token_x = get_dimension_token(cols_x)
    if output_grid is not None:
        if NeedPosition:
            rows_y, cols_y, flat_y, row_indices_y, col_indices_y = get_dimention(output_grid)
        else:
            rows_y, cols_y, flat_y = get_dimention(output_grid)
        row_token_y = get_dimension_token(rows_y)
        col_token_y = get_dimension_token(cols_y)
    # append the input grid, same as before
    input_tokens.append(BOS_X)
    input_tokens.append(row_token_x)
    input_tokens.append(col_token_x)
    if NeedPosition:
        row_indices.extend([0] * 3)
        col_indices.extend([0] * 3)
    input_tokens.extend(flat_x) # Add flattened input grid cells (as ints 0-9)
    if NeedPosition:
        row_indices.extend(row_indices_x)
        col_indices.extend(col_indices_x)
    input_tokens.append(EOS_X)
    if NeedPosition:
        row_indices.append(0)
        col_indices.append(0)
    if not IsDecode:
        if autoregressive:
            target_tokens.append(row_token_x)
            target_tokens.append(col_token_x)
            target_tokens.extend(flat_x) # Add flattened input grid cells (as ints 0-9)
            target_tokens.append(EOS_X)
            target_tokens.append(IGNORE_INDEX)
        else:
            target_tokens.extend([IGNORE_INDEX] * (len(input_tokens) - len(target_tokens)))

    len_input = len(input_tokens)
    # append the output grid
    input_tokens.append(PREDICT_ROW_Y)
    if NeedPosition:
        row_indices.append(0)
        col_indices.append(0)
    if not IsDecode:
        target_tokens.append(row_token_y)
        input_tokens.append(row_token_y)
        target_tokens.append(-100)
        input_tokens.append(PREDICT_COL_Y)
        target_tokens.append(col_token_y)
        input_tokens.append(col_token_y)
        target_tokens.append(-100)
        if NeedPosition:
            row_indices.extend([0] * 3)
            col_indices.extend([0] * 3)
        input_tokens.extend([PREDICT_CELL_Y] * (rows_y * cols_y))
        if NeedPosition:
            row_indices.extend(row_indices_y)
            col_indices.extend(col_indices_y)
        target_tokens.extend(flat_y)
    else:
        target_tokens = output_grid
    if NeedPosition:
        return {"input_tokens":numpy2torch(input_tokens), "target_tokens":numpy2torch(target_tokens) if target_tokens is not None else None, \
                "len_input":len_input, "row_indices":numpy2torch(row_indices), "col_indices":numpy2torch(col_indices)}
    else:
        return {"input_tokens":numpy2torch(input_tokens), "target_tokens":numpy2torch(target_tokens) if target_tokens is not None else None, "len_input":len_input}

def data_gen(data, IsTrain, max_length, autoregressive, NeedPosition, tokenize_func=tokenize_causal, IsDecode=False, ReturnLengths=True):
    """Generate data for training or testing.
    
    Args:
        data: Dictionary containing 'train' and 'test' datasets
        IsTrain: Boolean indicating whether to use training data
        max_length: Maximum sequence length for truncation
        autoregressive: Whether to use autoregressive training mode
        tokenize_func: Function to use for tokenization (default: tokenize_causal) or tokenize_oneshot
        IsDecode: When true, return the decoded input (x1,y1,x2,y2,...xk) and 
                 target for yk (if yk present, i.e. local run instead of leaderboard run)
                 else return None for target
    
    Yields:
        For tokenize_causal: Tokenized input and target tensors
        For tokenize_oneshot: Tokenized input, target tensors, and length of input
    """
    # Select dataset split
    dataset = data['train'] if IsTrain else data['test']
    
    # Shuffle training data
    if IsTrain:
        random.shuffle(dataset)
    
    for task in dataset:
        # Apply transformations only during training
        if IsTrain:
            # TODO: tansformation for decode
            task = forwardTask(task, generateTransformPara(len(task)))
        
        # Tokenize the task
        out = tokenize_func(task, autoregressive=autoregressive, IsDecode=IsDecode, max_length=max_length, NeedPosition=NeedPosition, ReturnLengths=ReturnLengths)
        yield out

def create_attention_mask(length: int) -> torch.Tensor:
  mask = torch.zeros(length, length, dtype=torch.bfloat16, device='cuda')
  is_upper_triangle = torch.triu(torch.ones(length, length, dtype=torch.bool, device='cuda'), diagonal=1)
  mask[is_upper_triangle] = torch.finfo(torch.bfloat16).min
  return mask.unsqueeze(0).unsqueeze(0)

def tokenize_VLM(task, processor, max_pairs=4, multiplier=14, decode=False):
    """
    This function takes a list of input-output grid pairs (ARC),
    processes them into a sequence suitable for a VLM. 

    Args:
        task: A list of tuples. Each tuple contains two elements:
              - input_grid (List[List[int]]): A 2D list representing the input grid.
              - output_grid (List[List[int]]): A 2D list representing the corresponding output grid.
        processor: An object (e.g., from Hugging Face Transformers) containing an
                   `image_processor` attribute used to preprocess the scaled color images
                   (e.g., normalization, tensor conversion).
        max_pairs: The maximum number of input-output pairs to process from the `task`.
                   Defaults to 4.
        multiplier: The integer factor used to scale the height and width of the grids
                    before image processing. Defaults to 14.
        decode: If True, return processed [x1, image_x1, y1, image_y1,... xk, image_xk, OUTPUT_TOKEN_IDX]
                If False (default), return processed [x1, image_x1, y1, image_y1,... xk, image_xk, yk]

    Returns:
        A tuple containing:
        - A dictionary containing the model inputs:
            - 'input_ids': A tensor of token IDs representing the interleaved text/special
                           tokens and image placeholder tokens.
            - 'pixel_values': A list of preprocessed image tensors, one for each grid
                              (input and output, except final output if decode=True).
                              Each tensor usually has shape (1, C, H, W).
            - 'token_type_ids': A tensor indicating the type of each token (0 for text/special,
                                1 for image soft tokens).
            - 'attention_mask': A tensor indicating which tokens should be attended to (all 1s here).
        - The target data:
            - If `decode` is False: A tensor of target token IDs, used for training (loss calculation).
                                   Contains IGNORE_INDEX (-100) for non-target tokens.
            - If `decode` is True: The raw `output_grid` np.array from the final pair.

    """
    
    BOS_TOKEN_IDX = 10
    INPUT_TOKEN_IDX = 11
    OUTPUT_TOKEN_IDX = 12
    NEWLINE_TOKEN_IDX = 13
    EOLINE_TOKEN_IDX = 14
    BEG_OF_IMAGE_TOKEN_IDX = 15
    IMAGE_SOFT_TOKEN_IDX = 16
    END_OF_IMAGE_TOKEN_IDX = 17
    IGNORE_INDEX = -100

    # Color mappings
    color_array = np.array([[255,   0,   0],
                            [  0,   0, 255],
                            [  0, 255,   0],
                            [255, 255,   0],
                            [255, 165,   0],
                            [128,   0, 128],
                            [255, 255, 255],
                            [  0, 255, 255],
                            [128, 128, 128],
                            [165,  42,  42]])

    images = []
    token_type_ids = [0]
    target_ids = [IGNORE_INDEX]
    input_ids = [BOS_TOKEN_IDX]

    def scale_image(image: List[List[int]], k: int) -> List[List[int]]:
        scaled_image = [
            # 1. Create the expanded row (scale horizontally)
            [pixel for pixel in original_row for _ in range(k)]
            # 2. Iterate through original rows
            for original_row in image
            # 3. Repeat each expanded row k times (scale vertically)
            for _ in range(k)
        ]
        return scaled_image
    
    def process_grid(grid, isInput,image_token):
        grid = np.array(grid, dtype=int)
        grid = np.concatenate([grid, np.full((grid.shape[0], 1), NEWLINE_TOKEN_IDX)], axis=1)
        grid = grid.flatten()
        grid[-1] = EOLINE_TOKEN_IDX
        input_ids = [INPUT_TOKEN_IDX if isInput else OUTPUT_TOKEN_IDX]
        input_ids.extend(grid.tolist())
        targets = input_ids[1:] + [IGNORE_INDEX]
        token_type_ids = [0] * len(input_ids) # non-image tokens
        # add image token
        if image_token is not None:
            input_ids.extend(image_token)
            token_type_ids.append(0) # BEG_OF_IMAGE_TOKEN_IDX
            token_type_ids.extend([1] * (len(image_token) - 2))
            token_type_ids.append(0) # END_OF_IMAGE_TOKEN_IDX
            targets.extend([IGNORE_INDEX] * len(image_token))
        return input_ids, targets, token_type_ids

    def process_image(grid, multiplier):
        # return image of shape (1, 3, H, W)
        grid = scale_image(grid, multiplier)
        images = color_array[np.array(grid, dtype=int)]
        images = np.transpose(images, (2, 0, 1)) # switch from (H, W, 3) to (3, H, W)
        return processor.image_processor.preprocess(images, return_tensors="pt", \
                                                    data_format="channels_first",input_data_format="channels_first",\
                                                    do_resize=False)['pixel_values'].to('cuda')
    def create_image_token(grid, multiplier):
        l, w = len(grid), len(grid[0])
        r, c = l * multiplier // 14, w * multiplier // 14
        return [BEG_OF_IMAGE_TOKEN_IDX] + [IMAGE_SOFT_TOKEN_IDX] * r * c + [END_OF_IMAGE_TOKEN_IDX]

    def process_all(grid, isInput, multiplier, isLast=False):
        # dont need to return image for last output grid
        if not isLast:
            image = process_image(grid, multiplier)
            image_token = create_image_token(grid, multiplier)
        else:
            image, image_token = None, None
        ids, target, type = process_grid(grid, isInput, image_token)
        return image, ids, target, type
    
    # Process each input-output pair
    for input_grid, output_grid in task[:max_pairs-1]:
        # process input grid
        image, ids, target, type = process_all(input_grid, isInput=True, multiplier=multiplier)
        images.append(image)
        input_ids.extend(ids)
        target_ids.extend(target)
        token_type_ids.extend(type)

        # process output grid
        image, ids, target, type = process_all(output_grid, isInput=False, multiplier=multiplier)
        images.append(image)
        input_ids.extend(ids)
        target_ids.extend(target)
        token_type_ids.extend(type)
    
    # Process the last input-output pair
    idx = len(task) - 1 if decode else max_pairs - 1
    input_grid, output_grid = task[idx]
    # input
    image, ids, target, type = process_all(input_grid, isInput=True, multiplier=multiplier)
    images.append(image)
    input_ids.extend(ids)
    target_ids.extend(target)
    token_type_ids.extend(type)
    # output
    if not decode:
        image, ids, target, type = process_all(output_grid, isInput=False, multiplier=multiplier, isLast=True)
        input_ids.extend(ids)
        target_ids.extend(target)
        token_type_ids.extend(type)
    else:
        # start with output token for decoding
        input_ids.append(OUTPUT_TOKEN_IDX)
        token_type_ids.append(0)
        target_ids = np.array(output_grid) if output_grid is not None else None
        
    return {'input_ids': numpy2torch(input_ids), \
            'pixel_values': images, \
            # 'token_type_ids': numpy2torch(token_type_ids), \
            # 'attention_mask': numpy2torch([1] * len(input_ids)), \
           },\
           target_ids if decode else numpy2torch(target_ids)

def tokenize_VLM_oneshot(task, processor, max_pairs=4, multiplier=14, decode=False):
    """
    This function takes a list of input-output grid pairs (ARC),
    processes them into a sequence suitable for a VLM. 

    Args:
        task: A list of tuples. Each tuple contains two elements:
              - input_grid (List[List[int]]): A 2D list representing the input grid.
              - output_grid (List[List[int]]): A 2D list representing the corresponding output grid.
        processor: An object (e.g., from Hugging Face Transformers) containing an
                   `image_processor` attribute used to preprocess the scaled color images
                   (e.g., normalization, tensor conversion).
        max_pairs: The maximum number of input-output pairs to process from the `task`.
                   Defaults to 4.
        multiplier: The integer factor used to scale the height and width of the grids
                    before image processing. Defaults to 14.
        decode: If True, return processed [x1, image_x1, y1, image_y1,... xk, image_xk, OUTPUT_TOKEN_IDX]
                If False (default), return processed [x1, image_x1, y1, image_y1,... xk, image_xk, yk]

    Returns:
        A tuple containing:
        - A dictionary containing the model inputs:
            - 'input_ids': A tensor of token IDs representing the interleaved text/special
                           tokens and image placeholder tokens.
            - 'pixel_values': A list of preprocessed image tensors, one for each grid
                              (input and output, except final output if decode=True).
                              Each tensor usually has shape (1, C, H, W).
            - 'token_type_ids': A tensor indicating the type of each token (0 for text/special,
                                1 for image soft tokens).
            - 'attention_mask': A tensor indicating which tokens should be attended to (all 1s here).
        - The target data:
            - If `decode` is False: A tensor of target token IDs, used for training (loss calculation).
                                   Contains IGNORE_INDEX (-100) for non-target tokens.
            - If `decode` is True: The raw `output_grid` np.array from the final pair.

    """
    
    # 0 ~ 9: Grid cell values (digits)
    INPUT_TOKEN_IDX = 11
    OUTPUT_TOKEN_IDX = 12
    NEWLINE_TOKEN_IDX = 13
    EOLINE_TOKEN_IDX = 14
    BEG_OF_IMAGE_TOKEN_IDX = 15
    IMAGE_SOFT_TOKEN_IDX = 16
    END_OF_IMAGE_TOKEN_IDX = 17
    PREDICT_CELL_Y = 10  # Placeholder for predicting output cells
    # Dimension tokens (SIZE_1 to SIZE_30 map to 18 to 47)
    SIZE_TOKEN_OFFSET = 17
    IGNORE_INDEX = -100

    max_pairs = min(max_pairs, len(task))
    # Color mappings
    color_array = np.array([[255,   0,   0],
                            [  0,   0, 255],
                            [  0, 255,   0],
                            [255, 255,   0],
                            [255, 165,   0],
                            [128,   0, 128],
                            [255, 255, 255],
                            [  0, 255, 255],
                            [128, 128, 128],
                            [165,  42,  42]])

    images = []
    token_type_ids = []
    target_ids = []
    input_ids = []

    def scale_image(image: List[List[int]], k: int) -> List[List[int]]:
        scaled_image = [
            # 1. Create the expanded row (scale horizontally)
            [pixel for pixel in original_row for _ in range(k)]
            # 2. Iterate through original rows
            for original_row in image
            # 3. Repeat each expanded row k times (scale vertically)
            for _ in range(k)
        ]
        return scaled_image
    
    def process_grid(grid, isInput, image_token):
        p,q = len(grid), len(grid[0])
        p_token, q_token = SIZE_TOKEN_OFFSET + p, SIZE_TOKEN_OFFSET + q
        grid = np.array(grid, dtype=int)
        grid = np.concatenate([grid, np.full((grid.shape[0], 1), NEWLINE_TOKEN_IDX)], axis=1)
        grid = grid.flatten()
        grid[-1] = EOLINE_TOKEN_IDX
        input_ids = [INPUT_TOKEN_IDX if isInput else OUTPUT_TOKEN_IDX]
        input_ids.append(p_token)
        input_ids.append(q_token)
        input_ids.extend(grid.tolist())
        targets = input_ids[1:] + [IGNORE_INDEX]
        token_type_ids = [0] * len(input_ids) # non-image tokens
        # add image token
        if image_token is not None:
            input_ids.extend(image_token)
            token_type_ids.append(0) # BEG_OF_IMAGE_TOKEN_IDX
            token_type_ids.extend([1] * (len(image_token) - 2))
            token_type_ids.append(0) # END_OF_IMAGE_TOKEN_IDX
            targets.extend([IGNORE_INDEX] * len(image_token))
        return input_ids, targets, token_type_ids

    def process_image(grid, multiplier):
        # return image of shape (1, 3, H, W)
        grid = scale_image(grid, multiplier)
        images = color_array[np.array(grid, dtype=int)]
        images = np.transpose(images, (2, 0, 1)) # switch from (H, W, 3) to (3, H, W)
        return processor.image_processor.preprocess(images, return_tensors="pt", \
                                                    data_format="channels_first",input_data_format="channels_first",\
                                                    do_resize=False)['pixel_values'].to('cuda')
    def create_image_token(grid, multiplier):
        l, w = len(grid), len(grid[0])
        r, c = l * multiplier // 14, w * multiplier // 14
        return [BEG_OF_IMAGE_TOKEN_IDX] + [IMAGE_SOFT_TOKEN_IDX] * r * c + [END_OF_IMAGE_TOKEN_IDX]

    def process_all(grid, isInput, multiplier):
        # dont need to return image for last output grid
        image = process_image(grid, multiplier)
        image_token = create_image_token(grid, multiplier)
        ids, target, type = process_grid(grid, isInput, image_token)
        return image, ids, target, type
    
    # Process each input-output pair
    for input_grid, output_grid in task[:max_pairs-1]:
        # process input grid
        image, ids, target, type = process_all(input_grid, isInput=True, multiplier=multiplier)
        images.append(image)
        input_ids.extend(ids)
        target_ids.extend(target)
        token_type_ids.extend(type)

        # process output grid
        image, ids, target, type = process_all(output_grid, isInput=False, multiplier=multiplier)
        images.append(image)
        input_ids.extend(ids)
        target_ids.extend(target)
        token_type_ids.extend(type)
    
    # Process the last input-output pair
    idx = len(task) - 1 if decode else max_pairs - 1
    input_grid, output_grid = task[idx]
    # input
    image, ids, target, type = process_all(input_grid, isInput=True, multiplier=multiplier)
    images.append(image)
    input_ids.extend(ids)
    target_ids.extend(target)
    token_type_ids.extend(type)
    # output
    input_ids.append(OUTPUT_TOKEN_IDX)
    l = len(input_ids) # logits[0, l:] for PREDICT_CELL_Y part
    if decode:
        # ids ends with OUTPUT_TOKEN_IDX token for decoding
        token_type_ids.append(0)
        target_ids = np.array(output_grid) if output_grid is not None else None
    else:
        p, q = len(output_grid), len(output_grid[0])
        p_token, q_token = SIZE_TOKEN_OFFSET + p, SIZE_TOKEN_OFFSET + q
        # input_ids
        input_ids.append(p_token)
        input_ids.append(q_token)
        input_ids.extend([PREDICT_CELL_Y] * (p * (q + 1))) # +1 for EOLINE_TOKEN_IDX
        # target_ids
        target_ids.append(p_token) # <- OUTPUT_TOKEN_IDX
        target_ids.append(q_token) # <- p_token
        target_ids.append(IGNORE_INDEX) # <- q_token
        for sublist in output_grid:
            target_ids.extend(sublist)
            target_ids.append(NEWLINE_TOKEN_IDX)
        # token_type_ids
        token_type_ids.extend([0] * (len(target_ids) - len(token_type_ids))) # non-image tokens
        
    return {'input_ids': numpy2torch(input_ids), \
            'pixel_values': images, \
            # 'token_type_ids': numpy2torch(token_type_ids), \
            # 'attention_mask': numpy2torch([1] * len(input_ids)), \
           },\
           target_ids if decode else numpy2torch(target_ids), l

def data_gen_VLM(data, IsTrain, processor, max_pairs, decode=False, tokenize_func=tokenize_VLM, max_len=12288):
    """Generate data for VLM
    """
    # Select dataset split
    dataset = data['train'] if IsTrain else data['test']
    attention_mask = create_attention_mask(max_len)
    # Shuffle training data
    if IsTrain:
        random.shuffle(dataset)
    
    for task in dataset:
        # Apply transformations only during training
        if IsTrain:
            # TODO: tansformation for decode
            task = forwardTask(task, generateTransformPara(len(task)))
        
        # Tokenize the task
        inputs, *others = tokenize_func(task, processor, max_pairs=max_pairs, decode=decode)
        l = inputs['input_ids'].shape[1]
        inputs['attention_mask'] = attention_mask[:,:,:l,:l]
        yield inputs, *others

# def tokenize_VLM_full_image(task, processor, max_pairs=4, decode=False):
#     """
#     fixed size image by scaling.
    
#     Args:
#         task: List of tuples [(input1, output1), (input2, output2), ...], where each input and output
#               is a 2D grid (list of lists) of integers between 0 and 9.
#         processor: A processor object for encoding images and text.
    
#     Returns:
#         tuple: (images, task_str)
#             - images: List of numpy arrays [input1_image, output1_image, input2_image, ...], each of shape (H, W, 3)
#             - task_str: String representation of the task with image placeholders
#     """
    
#     BOS_TOKEN_IDX = 10
#     INPUT_TOKEN_IDX = 11
#     OUTPUT_TOKEN_IDX = 12
#     NEWLINE_TOKEN_IDX = 13
#     EOLINE_TOKEN_IDX = 14
#     BEG_OF_IMAGE_TOKEN_IDX = 15
#     IMAGE_SOFT_TOKEN_IDX = 16
#     END_OF_IMAGE_TOKEN_IDX = 17
#     IGNORE_INDEX = -100

#     # Color mappings
#     color_array = np.array([[255,   0,   0],
#                             [  0,   0, 255],
#                             [  0, 255,   0],
#                             [255, 255,   0],
#                             [255, 165,   0],
#                             [128,   0, 128],
#                             [255, 255, 255],
#                             [  0, 255, 255],
#                             [128, 128, 128],
#                             [165,  42,  42]])

#     images = []
#     token_type_ids = [0]
#     target_ids = [IGNORE_INDEX]
#     input_ids = [BOS_TOKEN_IDX]
#     # TODO: 256 needs to be customized
#     image_token = [BEG_OF_IMAGE_TOKEN_IDX] + [IMAGE_SOFT_TOKEN_IDX] * 256 + [END_OF_IMAGE_TOKEN_IDX]

#     def process_grid(grid, isInput):
#         grid = np.array(grid, dtype=int)
#         grid = np.concatenate([grid, np.full((grid.shape[0], 1), NEWLINE_TOKEN_IDX)], axis=1)
#         grid = grid.flatten()
#         grid[-1] = EOLINE_TOKEN_IDX
#         input_ids = [INPUT_TOKEN_IDX if isInput else OUTPUT_TOKEN_IDX]
#         input_ids.extend(grid.tolist())
#         targets = input_ids[1:] + [IGNORE_INDEX]
#         token_type_ids = [0] * len(input_ids) # non-image tokens
#         # add image token
#         input_ids.extend(image_token)
#         token_type_ids.append(0) # BEG_OF_IMAGE_TOKEN_IDX
#         token_type_ids.extend([1] * (len(image_token) - 2))
#         token_type_ids.append(0) # END_OF_IMAGE_TOKEN_IDX
#         targets.extend([IGNORE_INDEX] * len(image_token))
#         return input_ids, targets, token_type_ids

#     # Process each input-output pair
#     for input_grid, output_grid in task[:max_pairs]:
#         # Convert grids to images
#         input_image = color_array[np.array(input_grid, dtype=int)]
#         input_image = np.transpose(input_image, (2, 0, 1)) # switch from (H, W, 3) to (3, H, W)
#         output_image = color_array[np.array(output_grid, dtype=int)]
#         output_image = np.transpose(output_image, (2, 0, 1))
#         images.extend([input_image, output_image])

#         # Convert grids to input_ids
#         # input grid
#         input_grid, target, token_type = process_grid(input_grid, isInput=True)
#         input_ids.extend(input_grid)
#         target_ids.extend(target)
#         token_type_ids.extend(token_type)
#         # output grid
#         output_grid, target, token_type = process_grid(output_grid, isInput=False)
#         input_ids.extend(output_grid)
#         target_ids.extend(target)
#         token_type_ids.extend(token_type)
        
#     images = processor.image_processor.preprocess(images, return_tensors="pt", data_format="channels_first",input_data_format="channels_first")
#     return {'input_ids': numpy2torch(input_ids), \
#             'pixel_values': images['pixel_values'].to('cuda'), \
#             'token_type_ids': numpy2torch(token_type_ids), \
#             'attention_mask': numpy2torch([1] * len(input_ids)), \
#            },\
#            numpy2torch(target_ids)

class OneshotDecoder(object):
    def __init__(self, model, PosEmbedModel=None, max_dim=30):
        """
        The OneshotDecoder class decodes an tokenized input sequence into a 2D output grid using a pre-trained model.
        
        This class employs a two-step process:
        1. **Dimension Prediction**: predict the dimensions (number of rows and columns) of the output grid.
        2. **Grid Prediction**: Once the dimensions are determined, it generates the entire grid in a single forward pass of the model.
        """
        self.model = model
        self.max_dim = max_dim
        self.PosEmbedModel = PosEmbedModel
        self.PREDICT_CELL_Y = 16  # Token representing a cell prediction
        self.SIZE_TOKEN_OFFSET = 16  # Offset for row/col tokens
        self.PREDICT_COL_Y = 15  # Token for predicting column size
        self.reset()  # Initialize/reset state variables

    def reset(self):
        """
        Resets the decoder's state variables.
        """
        self.min_nll = float('inf')
        self.past_key_values = None
        self.rows = None
        self.cols = None

    @torch.no_grad()
    def predict_dimensions(self, current_ids, row_indices=None, col_indices=None, past_key_values=None):
        """
        # TODO: double check cache. it may not work!!!
        argmax predicts the row and column dimensions of the output grid.
        
        Args:
            current_ids (torch.Tensor): Current sequence of token IDs, shape (1, seq_len) on cuda.
            past_key_values: Cached key/value pairs from previous model calls for efficiency.
        Returns:
            bool: True if an error occurs (invalid token prediction), False otherwise.
        """
        model = self.model
        device = model.device
        ### Step 1: Predict row_token_y after PREDICT_ROW_Y
        if self.PosEmbedModel is not None:
            self.PosEmbedModel.reset_parameters(row_indices, col_indices)
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model(
                input_ids=current_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        row_token_y = torch.argmax(next_token_logits, dim=-1).item()  # Predicted row dimension token
        if not (self.SIZE_TOKEN_OFFSET + 1 <= row_token_y <= self.SIZE_TOKEN_OFFSET + self.max_dim):
            return True
        self.rows = row_token_y - self.SIZE_TOKEN_OFFSET
        
        # Step 2: Predict col_token_y after PREDICT_COL_Y
        # Append row_token_y and PREDICT_COL_Y as input_ids
        input_ids = torch.tensor([[row_token_y, self.PREDICT_COL_Y]], dtype=torch.long, device=device)
        if self.PosEmbedModel is not None:
            row_indices = torch.tensor([[0, 0]], dtype=torch.long, device=device)
            col_indices = torch.tensor([[0, 0]], dtype=torch.long, device=device)
            self.PosEmbedModel.reset_parameters(row_indices, col_indices)
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
        self.past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        col_token_y = torch.argmax(next_token_logits, dim=-1).item()  # Predicted column dimension token
        
        # Calculate cols_y from col_token_y
        if not (self.SIZE_TOKEN_OFFSET + 1 <= col_token_y <= self.SIZE_TOKEN_OFFSET + self.max_dim):
            return True
        self.cols = col_token_y - self.SIZE_TOKEN_OFFSET
        return False  # No error

    @torch.no_grad()
    def predict_output_grid(self):
        """
        Predict the entire output grid given the dimensions.
        """
        # Encode rows and cols as tokens
        rows, cols = self.rows, self.cols
        col_token = self.SIZE_TOKEN_OFFSET + cols

        # Append col token and [PREDICT_CELL_Y] * (rows * cols) to input
        input = torch.tensor([[col_token] + [self.PREDICT_CELL_Y] * (rows * cols)], dtype=torch.long, device='cuda')
        if self.PosEmbedModel is not None:
            row_indices = [0] # col_token
            col_indices = [0] # col_token
            col_index_pattern = list(range(1, cols + 1))
            for r_idx in range(rows):
                row_indices.extend([r_idx + 1] * cols)
                col_indices.extend(col_index_pattern)
            row_indices = numpy2torch(row_indices)
            col_indices = numpy2torch(col_indices)
            self.PosEmbedModel.reset_parameters(row_indices, col_indices)
        # Get model predictions for the entire sequence
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = self.model(input_ids=input, past_key_values=self.past_key_values, use_cache=True).logits # Shape: (1, seq_len, vocab_size)
        del self.past_key_values
        cell_preds = logits[0, -rows * cols:, :].argmax(dim=-1).cpu().detach().numpy() # Shape: (rows * cols, )
        cell_preds = cell_preds.reshape(rows, cols)
        return cell_preds

    def decode(self, input_tokens):
        """
        Decode the input sequence to produce the output grid.
        
        :param input_tokens: Initial input sequence ending with BOS_Y.
        :return: 2D numpy array of the predicted output grid.
        """
        self.reset() # reset for new inputs
        # Step 1: Predict dimensions
        if self.predict_dimensions(input_tokens):
            print("Invalid token prediction for dimensions.")
            return None

        # Step 2: Predict the output grid
        output_grid = self.predict_output_grid()

        return output_grid

class CausalDecoder(object):
    """
    A depth-first search (DFS) based sequence generator for transformer models.
    **Args**:
        - `model`: The pre-trained transformer model used for sequence generation.
        - `max_depth` (int, optional): The maximum recursion depth for DFS generation. Defaults to `31 * 30 + 1`.
        - `multiplier` (float, optional): A multiplier for pruning paths based on NLL. Paths with NLL >= `min_nll * multiplier` are discarded. Defaults to `1.3`.
        - `prob_threshold` (float, optional): The cumulative probability threshold for stopping exploration of next tokens. Defaults to `0.8`.
        - `max_num_path` (int, optional): The maximum number of complete paths to collect. Defaults to `10`.
        - `IsDebug` (bool, optional): If `True`, enables debug logging. Defaults to `False`.

    **Attributes**:
        - `best_paths` (list): List of generated 2D grids, stored as numpy arrays.
        - `nlls` (list): List of negative log likelihoods (NLLs) corresponding to `best_paths`.
        - `min_nll` (float): The minimum NLL among the collected paths.

    **Methods**:
        - `decode(input_tokens, return_best_path)`:  
            Generates sequences starting from the provided input tokens.  
            **Args**:  
                - `input_tokens`: A dictionary containing `'input_ids'` (torch.Tensor).  
                - `return_best_path` (bool): If `True`, returns only the best path; otherwise, returns all collected paths.  
            **Returns**:  
                - If `return_best_path` is `True`: The best 2D grid (numpy array).  
                - If `return_best_path` is `False`: A list of all collected 2D grids (list of numpy arrays).

        - `reset()`:  
            Resets the internal state, clearing `best_paths`, `nlls`, and `min_nll`.

        - `detokenize_causal(tokens)`:  
            Converts a 1D list of tokens into a 2D grid, interpreting `LINE_BREAK` and `EOS_Y` tokens.

        - `check_equal_line_lengths(tensor)`:  
            Ensures that if the sequence ends with a line break, the last line's length matches the first line's length.

        - `check_row_col_len(tensor, max_col=30, max_row=30)`:  
            Verifies that the number of rows and columns in the tensor does not exceed the specified limits.

        - `dfs_generate(current_ids, current_seq_len, current_nll=0, past_key_values=None, current_depth=0)`:  
            The recursive DFS method that generates and explores possible sequences.
    """
    LINE_BREAK = 12
    EOS_Y = 14
    special_tokens = {LINE_BREAK, EOS_Y}
    def __init__(self, model, max_depth: int = 31 * 30 + 1, multiplier = 1.3, prob_threshold = 0.8, max_num_path = 10, IsDebug=False):
        """Initialize the searcher with a pre-trained model."""
        self.max_depth = max_depth
        self.model = model
        self.multiplier = multiplier
        self.prob_threshold = prob_threshold
        self.max_num_path = max_num_path
        self.IsDebug = IsDebug
        self.reset()  # Initialize/reset state variables

    def reset(self):
        """Reset the searcher state."""
        self.best_paths = []
        self.nlls = []
        self.min_nll = float('inf')

    def decode(self, input_tokens, return_best_path):
        self.reset()
        self.dfs_generate(input_tokens)
        if return_best_path:
            idx = np.argmin(self.nlls)
            best_path = self.best_paths[idx]
            return best_path
        return self.best_paths

    @staticmethod
    def detokenize_causal(tokens):
        """
        Detokenizes a 1D sequence of tokens (after BOS_Y) back into a 2D grid.
        Args:
            tokens: A 1D list of integer tokens starting after OUTPUT_TOKEN_IDX (12).
        Returns:
            A 2D numpy array of integers
        """

        grid = []
        row = []
        for token in tokens:
            if token == CausalDecoder.EOS_Y:
                if row:
                    grid.append(row) # this should not happen as line break is always followed by EOS_Y
                break  # Stop at EOS_Y
            elif token == CausalDecoder.LINE_BREAK:
                grid.append(row)
                row = []
            else:
                row.append(token)
        return np.array(grid)
    
    @staticmethod
    def check_equal_line_lengths(tensor):
        """ Check if length of last line is the same as first line."""
        tensor = tensor[0]
        item = tensor[-1].item()
        if item not in CausalDecoder.special_tokens: # only check if the last token is a line break or EOS_Y
            return True
        idx = (tensor == CausalDecoder.LINE_BREAK).nonzero(as_tuple=True)[0]
        if len(idx) <= 1: # first line
            return True
        if item == CausalDecoder.EOS_Y: # EOS_Y must be preceded by a line break and we dont need to check again 1, 2, line, 3, 4, line, EOS_Y
            if tensor[-2].item() != CausalDecoder.LINE_BREAK:
                return False
            else:
                return True # no need to check length as logic would fail
        return (idx[-1] - idx[-2] - 1) == idx[0]
    
    @staticmethod
    def check_row_col_len(tensor, max_col=30, max_row=30):
        """ Check if the number of rows and columns in the tensor is within limits."""
        tensor = tensor[0]
        item = tensor[-1].item()
        if item == CausalDecoder.EOS_Y:
            return True
        line_breaks = (tensor == CausalDecoder.LINE_BREAK).nonzero(as_tuple=True)[0]
        num_rows = len(line_breaks)
        if item != CausalDecoder.LINE_BREAK:
            num_rows += 1 # ongoing line
        if num_rows > max_row:
            return False  # Too many rows
        # Find start of last row (after previous line break or at 0)
        if len(line_breaks) == 0:
            last_row_start = 0
        else:
            last_row_start = line_breaks[-1] + 1
        last_row_end = len(tensor)
        last_row_length = last_row_end - last_row_start
        if last_row_length > max_col:
            return False  # Last row is too long
        return True  # All checks passed

    @torch.no_grad()
    def dfs_generate(self, current_ids, current_seq_len = 0, current_nll = 0, past_key_values = None, current_depth = 0):
        """Performs Depth-First Search to find good candidates. 
           cache position is needed to "un-do" (backtrack) changes made to past_key_values in deeper dfs
        """
        # current_ids is torch.Tensor of Shape: (1, seq_len)
        if self.IsDebug and current_depth > 0:
            print(f"Current NLL: {current_nll:.4f} | Path depth: {current_depth} | Current IDs: {current_ids[0][-1].item()}")
        model = self.model
        max_depth = self.max_depth
        device = model.device
        # Safety check for recursion depth
        if current_depth > max_depth or len(self.best_paths) >= self.max_num_path:
            return

        # Prepare inputs for the model
        if current_depth == 0:
            # First call, process the whole sequence
            current_seq_len = current_ids['input_tokens'].shape[1] - 1
            past_key_values = StaticCache(model.config, 1, current_seq_len + 30 * 31 + 2, device='cuda')
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(current_ids['input_tokens'],
                                use_cache=True,
                                past_key_values=past_key_values
                               )
        else:
            # Subsequent calls, only process the last token
            input_ids_step = current_ids[:, -1:]
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(input_ids_step,
                                use_cache=True,
                                past_key_values=past_key_values,
                                cache_position=torch.tensor([current_seq_len], device='cuda')
                               )
        # Get logits and new KV cache
        new_past_key_values = outputs.past_key_values # Updated KV cache

        # Get logits for the *next* token prediction
        next_token_logits = outputs.logits[:, -1, :].float() # Shape: (1, input_seq_len, vocab_size)

        # Calculate log probabilities and negative log likelihoods
        log_probs = torch.log_softmax(next_token_logits, dim=-1)
        nlls = -log_probs # Shape: (1, vocab_size)

        # Sort potential next tokens by NLL (ascending)
        sorted_nlls, sorted_indices = torch.sort(nlls.squeeze(), descending=False)
        cum_prob = 0.0
        
        # Iterate through the most promising next tokens
        for next_token_id, next_token_nll in zip(sorted_indices, sorted_nlls):
            if len(self.best_paths) >= self.max_num_path: return # Stop if we have enough paths
            next_token_id = next_token_id.item()
            next_token_nll = next_token_nll.item()

            potential_total_nll = current_nll + next_token_nll
            # --- Pruning ---
            if potential_total_nll >= self.min_nll * self.multiplier:
                # as nll is sorted, we skip the rest
                break

            if current_depth == 0: # first call
                next_ids = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
            else: # Append the chosen token
                next_ids = torch.cat(
                    [current_ids, torch.tensor([[next_token_id]], dtype=torch.long, device=device)],
                    dim=1
                )
            
            if not self.check_equal_line_lengths(next_ids):
                # If the line lengths are not equal, prune this branch
                continue

            if not self.check_row_col_len(next_ids):
                # skip going deeper when too many rows or cols
                continue            
            
            # --- Base Case: EOS token ---
            if next_token_id == CausalDecoder.EOS_Y: 
                # print(f"Found EOS. Path NLL: {potential_total_nll:.4f} | Path Len: {current_depth}",next_ids[0].tolist())
                self.best_paths.append(self.detokenize_causal(next_ids[0].tolist()))
                self.nlls.append(potential_total_nll)
                self.min_nll = min(self.min_nll, potential_total_nll)
                # Continue searching other branches after finding an EOS for this path
                continue # Don't recurse further down this path
            
            # --- Recursive Step ---
            # Pass the `new_past_key_values` which contains the cache state *after*
            self.dfs_generate(current_ids=next_ids,
                              current_seq_len=current_seq_len + 1,
                              current_nll=potential_total_nll,
                              past_key_values=new_past_key_values,
                              current_depth=current_depth + 1,
                             )

            cum_prob += math.exp(-next_token_nll)
            if cum_prob >= self.prob_threshold:
                break

def check_grid(y, yhat):
    # 0 shape mismatch
    # 1 not equal
    # 2 equal
    a,b = y.shape
    c,d = yhat.shape
    if a != c or b != d:
        return 0
    else:
        return (y == yhat).sum()/a/b
    
def check(decoder,targets):
    idx = np.argmin(decoder.nlls)
    best_res = check_grid(targets, decoder.best_paths[idx])
    any_res = max((check_grid(targets, grid) for grid in decoder.best_paths))
    return best_res, any_res