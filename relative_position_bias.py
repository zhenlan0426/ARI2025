# Deprecated relative position bias classes from commit 3ee89bc
import torch
import torch.nn as nn


class _RelPos2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bias_param, rows, cols, layer_idx):
        layers, H, max_h, max_w = bias_param.shape
        ctx.save_for_backward(rows, cols)
        ctx.layer_idx = layer_idx
        ctx.max_h = max_h
        ctx.max_w = max_w
        ctx.layers = layers
        ctx.H = H
        h_idxs = torch.clamp(rows.unsqueeze(0) - rows.unsqueeze(1), 0, max_h - 1)
        w_idxs = torch.clamp(cols.unsqueeze(0) - cols.unsqueeze(1), 0, max_w - 1)
        return bias_param[layer_idx:layer_idx+1, :, h_idxs, w_idxs]

    @staticmethod
    def backward(ctx, grad_out):
        rows, cols = ctx.saved_tensors
        layer_idx, max_h, max_w, layers, H = ctx.layer_idx, ctx.max_h, ctx.max_w, ctx.layers, ctx.H
        rows -= 1
        cols -= 1
        h_idxs = torch.clamp(rows.unsqueeze(0) - rows.unsqueeze(1), 0, max_h - 1)
        w_idxs = torch.clamp(cols.unsqueeze(0) - cols.unsqueeze(1), 0, max_w - 1)
        idx_flat = (h_idxs * max_w + w_idxs).view(-1)
        g_flat = grad_out[0].view(H, -1)
        d_bias = torch.zeros(layers, H, max_h, max_w, device=grad_out.device, dtype=grad_out.dtype)
        db_li = d_bias[layer_idx].view(H, -1)
        db_li.scatter_add_(1, idx_flat.unsqueeze(0).expand(H, -1), g_flat)
        return d_bias, None, None, None


class WithinGrid2DAttnScore(nn.Module):
    def __init__(self, layers, heads, max_height_delta, max_width_delta):
        super().__init__()
        self.layers = layers
        self.heads = heads
        self.max_height_delta = max_height_delta
        self.max_width_delta = max_width_delta
        self.relative_position_bias = nn.Parameter(
            torch.empty(self.layers, self.heads, self.max_height_delta, self.max_width_delta)
        )
        nn.init.normal_(self.relative_position_bias, std=0.01)

    def forward(self, rows: torch.LongTensor, cols: torch.LongTensor, layer_idx: int):
        height_indices = (rows.unsqueeze(0) - rows.unsqueeze(1))
        width_indices = (cols.unsqueeze(0) - cols.unsqueeze(1))
        height_indices = torch.clamp(height_indices, 0, self.max_height_delta - 1)
        width_indices = torch.clamp(width_indices, 0, self.max_width_delta - 1)
        return self.relative_position_bias[layer_idx:layer_idx+1, :, height_indices, width_indices]


class AcrossGrid2DAttnScore(WithinGrid2DAttnScore):
    def __init__(self, layers, heads, max_height_delta, max_width_delta):
        super().__init__(layers, heads, max_height_delta, max_width_delta)

    def forward(self, rows1, cols1, rows2, cols2, layer_idx):
        height_indices = (rows2.unsqueeze(1) - rows1.unsqueeze(0))
        width_indices = (cols2.unsqueeze(1) - cols1.unsqueeze(0))
        height_center = (self.max_height_delta - 1) // 2
        width_center = (self.max_width_delta - 1) // 2
        height_indices += height_center
        width_indices += width_center
        height_indices = torch.clamp(height_indices, 0, self.max_height_delta - 1)
        width_indices = torch.clamp(width_indices, 0, self.max_width_delta - 1)
        return self.relative_position_bias[layer_idx:layer_idx+1, :, height_indices, width_indices]


class MultiGridAttention(nn.Module):
    def __init__(self, layers, heads, max_height_delta1, max_width_delta1, max_height_delta2, max_width_delta2):
        super().__init__()
        self.within_grid_attn = WithinGrid2DAttnScore(layers, heads, max_height_delta1, max_width_delta1)
        self.across_grid_attn = AcrossGrid2DAttnScore(layers, heads, max_height_delta2, max_width_delta2)
        self.rows = None
        self.cols = None
        self.lengths = None

    def set_indices(self, rows, cols, lengths):
        self.rows = rows
        self.cols = cols
        self.lengths = lengths

    def forward(self, layer_idx):
        rows, cols, lengths = self.rows, self.cols, self.lengths
        device = rows.device
        total_length = rows.size(0)
        heads = self.within_grid_attn.heads
        attention_biases = torch.zeros((1, heads, total_length, total_length), device=device)
        current_pos = 0
        num_complete_pairs = len(lengths) // 2
        has_unpaired_input = len(lengths) % 2 == 1
        for pair_idx in range(num_complete_pairs):
            x_start = current_pos
            x_end = x_start + lengths[pair_idx * 2]
            y_start = x_end
            y_end = y_start + lengths[pair_idx * 2 + 1]
            x_rows = rows[x_start:x_end]
            x_cols = cols[x_start:x_end]
            y_rows = rows[y_start:y_end]
            y_cols = cols[y_start:y_end]
            x_to_x_bias = self.within_grid_attn(x_rows, x_cols, layer_idx)
            attention_biases[:, :, x_start:x_end, x_start:x_end] = x_to_x_bias
            y_to_y_bias = self.within_grid_attn(y_rows, y_cols, layer_idx)
            attention_biases[:, :, y_start:y_end, y_start:y_end] = y_to_y_bias
            y_to_x_bias = self.across_grid_attn(x_rows, x_cols, y_rows, y_cols, layer_idx)
            attention_biases[:, :, y_start:y_end, x_start:x_end] = y_to_x_bias
            current_pos = y_end
        if has_unpaired_input:
            x_start = current_pos
            x_end = x_start + lengths[-1]
            x_rows = rows[x_start:x_end]
            x_cols = cols[x_start:x_end]
            x_to_x_bias = self.within_grid_attn(x_rows, x_cols, layer_idx)
            attention_biases[:, :, x_start:x_end, x_start:x_end] = x_to_x_bias
            current_pos = x_end
        special_token_mask = (rows == 0)
        special_row_mask = special_token_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(1, heads, total_length, total_length)
        special_col_mask = special_token_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(1, heads, total_length, total_length)
        attention_biases.masked_fill_(special_row_mask, 0.0)
        attention_biases.masked_fill_(special_col_mask, 0.0)
        causal_mask = torch.triu(torch.ones(total_length, total_length, device=device), diagonal=1).bool()
        attention_biases.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), torch.finfo(torch.bfloat16).min)
        return attention_biases


class _MultiGridAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, within_bias, across_bias, rows, cols, lengths, layer_idx):
        layers, H, mh1, mw1 = within_bias.shape
        _, _, mh2, mw2 = across_bias.shape
        L = rows.shape[0]
        attn = torch.zeros((1, H, L, L), device=rows.device, dtype=within_bias.dtype)
        cur = 0
        lens = lengths
        Npairs = len(lens) // 2
        has_tail = len(lens) % 2 == 1
        for i in range(Npairs):
            lx, ly = lens[2*i], lens[2*i+1]
            xs, xe = cur, cur+lx
            ys, ye = xe, xe+ly
            x_rows = rows[xs:xe]; x_cols = cols[xs:xe]
            y_rows = rows[ys:ye]; y_cols = cols[ys:ye]
            hdx = torch.clamp(x_rows.unsqueeze(0)-x_rows.unsqueeze(1), 0, mh1-1)
            wdx = torch.clamp(x_cols.unsqueeze(0)-x_cols.unsqueeze(1), 0, mw1-1)
            attn[:, :, xs:xe, xs:xe] = within_bias[layer_idx:layer_idx+1, :, hdx, wdx]
            hdy = torch.clamp(y_rows.unsqueeze(0)-y_rows.unsqueeze(1), 0, mh1-1)
            wdy = torch.clamp(y_cols.unsqueeze(0)-y_cols.unsqueeze(1), 0, mw1-1)
            attn[:, :, ys:ye, ys:ye] = within_bias[layer_idx:layer_idx+1, :, hdy, wdy]
            hdxy = torch.clamp(y_rows.unsqueeze(1)-x_rows.unsqueeze(0), 0, mh2-1)
            wdxy = torch.clamp(y_cols.unsqueeze(1)-x_cols.unsqueeze(0), 0, mw2-1)
            attn[:, :, ys:ye, xs:xe] = across_bias[layer_idx:layer_idx+1, :, hdxy, wdxy]
            cur = ye
        if has_tail:
            lx = lens[-1]
            xs, xe = cur, cur+lx
            x_rows = rows[xs:xe]; x_cols = cols[xs:xe]
            hdx = torch.clamp(x_rows.unsqueeze(0)-x_rows.unsqueeze(1), 0, mh1-1)
            wdx = torch.clamp(x_cols.unsqueeze(0)-x_cols.unsqueeze(1), 0, mw1-1)
            attn[:, :, xs:xe, xs:xe] = within_bias[layer_idx:layer_idx+1, :, hdx, wdx]
        ctx.save_for_backward(rows, cols)
        ctx.lengths = lens
        ctx.layer_idx = layer_idx
        ctx.shapes = (layers, H, mh1, mw1, mh2, mw2)
        return attn

    @staticmethod
    def backward(ctx, grad_attn):
        rows, cols = ctx.saved_tensors
        lens = ctx.lengths
        layer_idx = ctx.layer_idx
        layers, H, mh1, mw1, mh2, mw2 = ctx.shapes
        d_within = torch.zeros(layers, H, mh1, mw1, device=grad_attn.device, dtype=grad_attn.dtype)
        d_across = torch.zeros(layers, H, mh2, mw2, device=grad_attn.device, dtype=grad_attn.dtype)
        cur = 0
        Npairs = len(lens) // 2
        has_tail = len(lens) % 2 == 1
        for i in range(Npairs):
            lx, ly = lens[2*i], lens[2*i+1]
            xs, xe = cur, cur+lx
            ys, ye = xe, xe+ly
            x_rows = rows[xs:xe]; x_cols = cols[xs:xe]
            y_rows = rows[ys:ye]; y_cols = cols[ys:ye]
            g_xx = grad_attn[0, :, xs:xe, xs:xe].reshape(H, -1)
            hdx = torch.clamp(x_rows.unsqueeze(0)-x_rows.unsqueeze(1), 0, mh1-1).reshape(-1)
            wdx = torch.clamp(x_cols.unsqueeze(0)-x_cols.unsqueeze(1), 0, mw1-1).reshape(-1)
            idx = hdx*mw1 + wdx
            d_within[layer_idx].view(H, -1).scatter_add_(1, idx.unsqueeze(0).expand(H, -1), g_xx)
            g_yy = grad_attn[0, :, ys:ye, ys:ye].reshape(H, -1)
            hdy = torch.clamp(y_rows.unsqueeze(0)-y_rows.unsqueeze(1), 0, mh1-1).reshape(-1)
            wdy = torch.clamp(y_cols.unsqueeze(0)-y_cols.unsqueeze(1), 0, mw1-1).reshape(-1)
            idx = hdy*mw1 + wdy
            d_within[layer_idx].view(H, -1).scatter_add_(1, idx.unsqueeze(0).expand(H, -1), g_yy)
            g_yx = grad_attn[0, :, ys:ye, xs:xe].reshape(H, -1)
            hdxy = torch.clamp(y_rows.unsqueeze(1)-x_rows.unsqueeze(0), 0, mh2-1).reshape(-1)
            wdxy = torch.clamp(y_cols.unsqueeze(1)-x_cols.unsqueeze(0), 0, mw2-1).reshape(-1)
            idx = hdxy*mw2 + wdxy
            d_across[layer_idx].view(H, -1).scatter_add_(1, idx.unsqueeze(0).expand(H, -1), g_yx)
            cur = ye
        if has_tail:
            lx = lens[-1]
            xs, xe = cur, cur+lx
            x_rows = rows[xs:xe]; x_cols = cols[xs:xe]
            g_xx = grad_attn[0, :, xs:xe, xs:xe].reshape(H, -1)
            hdx = torch.clamp(x_rows.unsqueeze(0)-x_rows.unsqueeze(1), 0, mh1-1).reshape(-1)
            wdx = torch.clamp(x_cols.unsqueeze(0)-x_cols.unsqueeze(1), 0, mw1-1).reshape(-1)
            idx = hdx*mw1 + wdx
            d_within[layer_idx].view(H, -1).scatter_add_(1, idx.unsqueeze(0).expand(H, -1), g_xx)
        return d_within, d_across, None, None, None, None
