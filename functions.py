from dataclasses import dataclass
import numpy as np
import random
import torch
from typing import List, Tuple, Optional

''' Dataset processing utilities for ARC tasks '''
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

def numpy2torch(x, max_length):
    """Convert numpy array to torch tensor and move to GPU, with length truncation."""
    x = torch.tensor(x[:max_length][None]).to('cuda')
    return x

def oneshot_mask(*lengths, X_attend2_history=False):
    """
    Creates a custom attention mask for ARC examples with parallel decoding.

    Args:
        *lengths: Sequence of integers for segment lengths
                  (x1_len, y1_len, ..., xk_len, yk_len).
        X_attend2_history (bool): If True, allows X_i segments to attend to all previous segments.

    Returns:
        numpy.ndarray: A 2D numpy array of shape (1, 1, total_len, total_len) containing attention mask values.
                     Values are 0 for positions that can attend and -inf for masked positions.

    Rules applied:
    1. Xi attends fully to Xi (within-grid attention).
    2. yi attends fully to all preceding history (X1, y1, ..., Xi-1, yi-1)
       AND its corresponding input Xi.
    3. row attends to all preceding history, col attends to row in addtion to all preceding history
       cell attends to row and col and all preceding history...  
    """

    total_len = sum(lengths)
    # Initialize with False. We will selectively enable attention.
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

    # Apply attention rules segment by segment
    for segment in segments:
        start, end = segment['start'], segment['end']
        segment_type = segment['type']

        if segment_type == 'X':
            # Rule 1: Xi attends fully to itself (intra-X attention)
            if X_attend2_history:
                mask[start:end, 0:start] = True
            else:
                mask[start:end, start:end] = True
            # Optional: Allow X to attend to prior history if needed
            # mask[start:end, 0:start] = True

        elif segment_type == 'Y':
            # Rule 2: yi attends to all history up to and including Xi
            mask[start:end, 0:start] = True

            # diagonal attention for yi
            indices = np.arange(start, end)
            mask[indices, indices] = True # Allow self-attention

            # all y_i attend to both row and column tokens, column token can attend to row token
            mask[start:end, start] = True
            mask[start+1:end, start+1] = True
    
    # Convert boolean mask to attention values (0 for attend, -inf for mask)
    out = np.where(mask, 0, -np.inf)
    return out[None][None]

def causal_mask(*lengths, X_attend2_history=False):
    """
    Creates a custom attention mask for ARC examples with causal attention for Y segments.

    Args:
        *lengths: Sequence of integers for segment lengths
                  (x1_len, y1_len, ..., xk_len, yk_len).
        X_attend2_history (bool): If True, allows X_i segments to attend to all previous segments.

    Returns:
        numpy.ndarray: A 2D numpy array of shape (total_len, total_len) containing attention mask values.
                     Values are 0 for positions that can attend and -inf for masked positions.

    Rules applied:
    1. Xi attends fully to Xi (within-grid attention).
    2. yi attends fully to all preceding history (X1, y1, ..., Xi-1, yi-1, Xi)
       AND within its segment previous yi-1, yi-2, ... 
    """
    total_len = sum(lengths)
    # Initialize with False. We will selectively enable attention.
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

    # Apply attention rules segment by segment
    for segment in segments:
        start, end = segment['start'], segment['end']
        segment_type = segment['type']

        if segment_type == 'X':
            if X_attend2_history: # Xi attends to all previous segments
                mask[start:end, 0:end] = True
            else: # Xi attends only to its segment (intra-X attention)
                mask[start:end, start:end] = True
            
        elif segment_type == 'Y':
            # Each token can attend to itself and all previous tokens
            for i in range(start, end):
                mask[i, 0:i+1] = True  # Causal attention pattern
    
    # Convert boolean mask to attention values (0 for attend, -inf for mask)
    out = np.where(mask, 0, -np.inf)
    return out[None][None]

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

def tokenize_causal(task, return_lengths=False, IsDecode=False):
    """
    Tokenizes autoregressively.

    Args:
        task: List of (input_grid, output_grid) tuples.
              For decoding, the last output_grid can be None.
        return_lengths: Whether to return sequence lengths for each x and y segment.
        IsDecode: if using the model for decoding or training.

    Returns:
        input_tokens: Numpy array of input token IDs.
                      For training: (x1, y1, x2, y2, ... xN, yN)
                      For decoding: (x1, y1, x2, y2, ..., xN, BOS_Y)
        final_target: Depends on IsDecode.
                      For training: Numpy array of shifted target token IDs.
                      For decoding: The raw 2d grid (yN) or None.
        lengths (optional): List of sequence lengths corresponding to segments
                           added to input_tokens.
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
    lengths = []  # To store lengths of each x and y sequence
    
    n = len(task)
    for i, (x, y) in enumerate(task):
        IsLast = (i == n-1) and IsDecode
        # Process input grid (x)
        x_start_idx = len(input_tokens)
        
        input_tokens.append(BOS_X)
        if not IsDecode:
            target_tokens.append(PAD_TOKEN)
        
        for row in x:
            # Add row elements
            input_tokens.extend(row)
            if not IsDecode:
                target_tokens.extend([PAD_TOKEN]*len(row))

            input_tokens.append(LINE_BREAK)
            if not IsDecode:
                target_tokens.append(PAD_TOKEN)

        input_tokens.append(EOS_X)
        if not IsDecode:
            target_tokens.append(PAD_TOKEN)
        
        x_len = len(input_tokens) - x_start_idx  # Length including special tokens

        # Process output grid (y)
        y_start_idx = len(input_tokens)
        
        input_tokens.append(BOS_Y)
        if not IsDecode:
            target_tokens.append(PAD_TOKEN)  # Mask BOS_Y
        
        lengths.append(x_len)
        if not IsLast:
            for row in y:
                # Add row elements
                input_tokens.extend(row)
                if not IsDecode:
                    target_tokens.extend(row)  # Keep y values in target
        
                input_tokens.append(LINE_BREAK)
                if not IsDecode:
                    target_tokens.append(LINE_BREAK)

            input_tokens.append(EOS_Y)
            if not IsDecode:
                target_tokens.append(EOS_Y)  # Include EOS_Y in target
        
            y_len = len(input_tokens) - y_start_idx  # Length including special tokens
            lengths.append(y_len)
        else:
            target_tokens = y        
        
    # Create shifted targets (for next-token prediction)
    if not IsDecode:
        target_tokens = target_tokens[1:] + [PAD_TOKEN]
    
    # Convert to numpy arrays
    if return_lengths:
        return np.array(input_tokens), np.array(target_tokens) if target_tokens else None, lengths
    else:
        return np.array(input_tokens), np.array(target_tokens) if target_tokens else None

def parse_causal_y(input_tokens):
    # TODO: Implement this
    pass

def tokenize_arc_oneshot(task:list[tuple[list[list[int]], list[list[int]]]], autoregressive:bool, max_length:int=None):
    """
    Tokenizes one-shot prediction.
    Uses distinct placeholder tokens for predicting output cells.
    
    Args:
        task (list): A list of tuples, each containing (input_grid, output_grid), where each grid
                     is a list of lists of integers (0-9)
        max_length (int): Maximum sequence length to consider for tokenization
        autoregressive (bool): Whether to train model on x,y or just on last y (oneshot)
    Returns:
        Tuple of:
            - numpy array of input tokens
            - numpy array of target tokens
            - integer idx such that target[idx:] is the oneshot target
    """
    # Token definitions with direct values
    # 0-9: Grid cell values (digits)
    BOS_X = 10       # Beginning of input grid
    EOS_X = 11       # End of input grid
    BOS_Y = 12       # Beginning of output grid
    EOS_Y = 13       # End of output grid
    # PREDICT_ROW_Y = 14  # no longer used
    # PREDICT_COL_Y = 15  # no longer used
    PREDICT_CELL_Y = 16  # Placeholder for predicting output cells
    
    # Dimension tokens (SIZE_1 to SIZE_30 map to 17 to 46)
    SIZE_TOKEN_OFFSET = 16

    def get_grid_dimensions(grid):
        """Extract dimensions and flatten a grid."""
        num_rows = len(grid)
        num_cols = len(grid[0])
        flat_grid = [cell for row in grid for cell in row]    
        return num_rows, num_cols, flat_grid

    def get_dimension_token(dim_size):
        """Gets the token ID for a given dimension size."""
        return SIZE_TOKEN_OFFSET + dim_size

    IGNORE_INDEX = -100
    input_tokens = []
    target_tokens = []

    n_task = find_first_exceed(task, max_length)
    for input_grid, output_grid in task[:n_task-1]:
        # --- Validate and Flatten Input Grid (X) ---
        rows_x, cols_x, flat_x = get_grid_dimensions(input_grid)
        row_token_x = get_dimension_token(rows_x)
        col_token_x = get_dimension_token(cols_x)

        # --- Validate and Flatten Output Grid (Y) ---
        rows_y, cols_y, flat_y = get_grid_dimensions(output_grid)
        row_token_y = get_dimension_token(rows_y) # Actual target token for rows_y
        col_token_y = get_dimension_token(cols_y) # Actual target token for cols_y
        num_output_cells = rows_y * cols_y

        # --- Construct Model input Sequence ---
        # append the input grid
        len_input = len(input_tokens)
        input_tokens.append(BOS_X)
        input_tokens.append(row_token_x)
        input_tokens.append(col_token_x)
        input_tokens.extend(flat_x) # Add flattened input grid cells (as ints 0-9)
        input_tokens.append(EOS_X)
        
        # append the output grid
        input_tokens.append(BOS_Y)
        input_tokens.append(row_token_y)
        input_tokens.append(col_token_y)
        input_tokens.extend(flat_y)
        input_tokens.append(EOS_Y)
        
        # --- Construct Model Target Sequence ---
        if autoregressive:
            # shifted input
            target_tokens.append(row_token_x)
            target_tokens.append(col_token_x)
            target_tokens.extend(flat_x) # Add flattened input grid cells (as ints 0-9)
            target_tokens.append(EOS_X)
            
            # append the output grid
            target_tokens.append(BOS_Y)
            target_tokens.append(row_token_y)
            target_tokens.append(col_token_y)
            target_tokens.extend(flat_y)
            target_tokens.append(EOS_Y)
            target_tokens.append(IGNORE_INDEX)
        else:
            target_tokens.extend([IGNORE_INDEX] * (len(input_tokens) - len_input))

    # --- Construct Model Input and Target Sequence for the last task ---
    input_grid, output_grid = task[n_task-1]
    rows_x, cols_x, flat_x = get_grid_dimensions(input_grid)
    row_token_x = get_dimension_token(rows_x)
    col_token_x = get_dimension_token(cols_x)
    rows_y, cols_y, flat_y = get_grid_dimensions(output_grid)
    row_token_y = get_dimension_token(rows_y)
    col_token_y = get_dimension_token(cols_y)
    # append the input grid, same as before
    len_input = len(input_tokens)
    input_tokens.append(BOS_X)
    input_tokens.append(row_token_x)
    input_tokens.append(col_token_x)
    input_tokens.extend(flat_x) # Add flattened input grid cells (as ints 0-9)
    input_tokens.append(EOS_X)
    if autoregressive:
        target_tokens.append(row_token_x)
        target_tokens.append(col_token_x)
        target_tokens.extend(flat_x) # Add flattened input grid cells (as ints 0-9)
        target_tokens.append(EOS_X)
        target_tokens.append(IGNORE_INDEX)
    else:
        target_tokens.extend([IGNORE_INDEX] * (len(input_tokens) - len_input))
    len_input = len(input_tokens)

    # append the output grid
    input_tokens.append(BOS_Y)
    target_tokens.append(row_token_y)

    input_tokens.append(row_token_y)
    target_tokens.append(col_token_y)

    input_tokens.append(col_token_y)
    target_tokens.append(IGNORE_INDEX)

    input_tokens.extend([PREDICT_CELL_Y] * (rows_y * cols_y))
    target_tokens.extend(flat_y)
    
    return np.array(input_tokens), np.array(target_tokens), len_input

def data_gen(data, IsTrain, max_length, return_lengths=False, tokenize_func=tokenize_causal,\
             mask_func=causal_mask, X_attend2_history=False, IsDecode=False):
    """Generate data for training or testing.
    
    Args:
        data: Dictionary containing 'train' and 'test' datasets
        IsTrain: Boolean indicating whether to use training data
        max_length: Maximum sequence length for truncation
        return_lengths: Whether to return sequence lengths
        tokenize_func: Function to use for tokenization
        mask_func: Function to use for creating attention masks
        X_attend2_history: Whether to allow X segments to attend to previous segments
        IsDecode: when true, return the decoded input (x1,y1,x2,y2,...xk), 
        target for yk (if yk present, i.e. local run instead of leaderboard run)
            else return None for target
    Yields:
        Tokenized input, target, and optionally attention mask
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
        if return_lengths:
            x, y, lengths = tokenize_func(task, return_lengths=True, IsDecode=IsDecode)
            # Create appropriate attention mask based on tokenization function
            mask = mask_func(*lengths, X_attend2_history=X_attend2_history)
            yield numpy2torch(x, max_length), numpy2torch(y, max_length), torch.tensor(mask[:,:,:max_length,:max_length], dtype=torch.bfloat16).to('cuda')
        else:
            x, y = tokenize_func(task, return_lengths=False, IsDecode=IsDecode)
            yield numpy2torch(x, max_length), numpy2torch(y, max_length)
