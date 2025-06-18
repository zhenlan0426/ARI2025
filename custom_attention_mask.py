# Deprecated custom attention mask functions from commit 0b0334fe
import numpy as np


def oneshot_mask(*lengths, X_attend2_history=False):
    """Creates a custom attention mask for ARC examples with parallel decoding."""
    total_len = sum(lengths)
    mask = np.full((total_len, total_len), False, dtype=bool)

    segments = []
    current_idx = 0
    is_x_segment = True
    for length in lengths:
        start = current_idx
        end = current_idx + length
        segment_type = 'X' if is_x_segment else 'Y'
        segments.append({'start': start, 'end': end, 'type': segment_type, 'len': length})
        current_idx += length
        is_x_segment = not is_x_segment

    for segment in segments:
        start, end = segment['start'], segment['end']
        segment_type = segment['type']

        if segment_type == 'X':
            if X_attend2_history:
                mask[start:end, 0:start] = True
            else:
                mask[start:end, start:end] = True
        elif segment_type == 'Y':
            mask[start:end, 0:start] = True
            indices = np.arange(start, end)
            mask[indices, indices] = True
            mask[start:end, start] = True
            mask[start+1:end, start+1] = True

    out = np.where(mask, 0, -np.inf)
    return out[None][None]


def causal_mask(*lengths, X_attend2_history=False):
    """Creates a custom attention mask with causal attention for Y segments."""
    total_len = sum(lengths)
    mask = np.full((total_len, total_len), False, dtype=bool)

    segments = []
    current_idx = 0
    is_x_segment = True
    for length in lengths:
        start = current_idx
        end = current_idx + length
        segment_type = 'X' if is_x_segment else 'Y'
        segments.append({'start': start, 'end': end, 'type': segment_type, 'len': length})
        current_idx += length
        is_x_segment = not is_x_segment

    for segment in segments:
        start, end = segment['start'], segment['end']
        segment_type = segment['type']

        if segment_type == 'X':
            if X_attend2_history:
                mask[start:end, 0:start] = True
            else:
                mask[start:end, start:end] = True
        elif segment_type == 'Y':
            for i in range(start, end):
                mask[i, 0:i+1] = True

    out = np.where(mask, 0, -np.inf)
    return out[None][None]
