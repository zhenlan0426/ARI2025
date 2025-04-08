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

def tokenize_task(task, return_lengths=False):
    """Tokenizes an ARC task into input/target sequences with special tokens.
    
    Args:
        task: List of (input_grid, output_grid) tuples
        return_lengths: Whether to return sequence lengths for each x and y
        
    Returns:
        input_tokens: Numpy array of input token IDs
        target_tokens: Numpy array of target token IDs
        lengths (optional): List of sequence lengths [x1_len, y1_len, x2_len, y2_len, ...]
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
    
    for x, y in task:
        # Process input grid (x)
        x_start_idx = len(input_tokens)
        
        input_tokens.append(BOS_X)
        target_tokens.append(PAD_TOKEN)
        
        for row in x:
            # Add row elements
            input_tokens.extend(row)
            target_tokens.extend([PAD_TOKEN]*len(row))

            input_tokens.append(LINE_BREAK)
            target_tokens.append(PAD_TOKEN)

        input_tokens.append(EOS_X)
        target_tokens.append(PAD_TOKEN)
        
        x_len = len(input_tokens) - x_start_idx  # Length including special tokens

        # Process output grid (y)
        y_start_idx = len(input_tokens)
        
        input_tokens.append(BOS_Y)
        target_tokens.append(PAD_TOKEN)  # Mask BOS_Y
        
        for row in y:
            # Add row elements
            input_tokens.extend(row)
            target_tokens.extend(row)  # Keep y values in target
        
            input_tokens.append(LINE_BREAK)
            target_tokens.append(LINE_BREAK)

        input_tokens.append(EOS_Y)
        target_tokens.append(EOS_Y)  # Include EOS_Y in target
        
        y_len = len(input_tokens) - y_start_idx  # Length including special tokens
        
        lengths.append(x_len)
        lengths.append(y_len)

    # Create shifted targets (for next-token prediction)
    target_tokens = target_tokens[1:] + [PAD_TOKEN]
    
    # Convert to numpy arrays
    if return_lengths:
        return np.array(input_tokens), np.array(target_tokens), lengths
    else:
        return np.array(input_tokens), np.array(target_tokens)

def parse_generated_y(input_tokens):
    """Parses the last Y (generated) tokenized sequence and converts it to a 2D grid.
    
    Args:
        input_tokens: List/array of tokens containing full sequence
    
    Returns:
        2D numpy array of the last generated Y grid
    """
    # Special token IDs
    BOS_Y = 13
    EOS_Y = 14
    LINE_BREAK = 12

    # Find last BOS_Y
    last_bos = max(i for i, t in enumerate(input_tokens) if t == BOS_Y)

    # Extract Y sequence between BOS_Y and end (or EOS_Y if present)
    y_tokens = input_tokens[last_bos+1:-1]  # Exclude BOS_Y and last token

    # Split into rows using line breaks
    rows = []
    current_row = []
    for t in y_tokens:
        if t == LINE_BREAK:
            if current_row:  # Skip empty rows from trailing line breaks
                rows.append(current_row)
                current_row = []
        elif t != EOS_Y:  # Skip EOS_Y if present
            current_row.append(t)
    
    # Add the last row if it's not empty and wasn't terminated by a line break
    if current_row:
        rows.append(current_row)
        
    # Validate the grid
    if not rows:
        return []
    
    # Check that all rows have the same length
    row_lengths = set(len(row) for row in rows)
    assert len(row_lengths) == 1, f"Invalid 2D grid: rows have different lengths {row_lengths}"
    
    return rows

def numpy2torch(x, max_length):
    """Convert numpy array to torch tensor and move to GPU, with length truncation."""
    x = torch.tensor(x[:max_length][None]).to('cuda')
    return x

def data_gen(data, IsTrain, max_length, return_lengths=False, tokenize_func=tokenize_task):
    """Generate data for training or testing.
    
    Args:
        data: Dictionary containing 'train' and 'test' datasets
        IsTrain: Boolean indicating whether to use training data
        max_length: Maximum sequence length for truncation
        return_lengths: Whether to return sequence lengths
        tokenize_func: Function to use for tokenization
        
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
            task = forwardTask(task, generateTransformPara(len(task)))
        
        # Tokenize the task
        if return_lengths:
            x, y, lengths = tokenize_func(task, return_lengths=True)
            # Create appropriate attention mask based on tokenization function
            if tokenize_func is tokenize_task:
                mask = create_arc_causal_attention_mask(*lengths)
            else:
                mask = create_arc_attention_mask_oneshot(*lengths)
            yield numpy2torch(x, max_length), numpy2torch(y, max_length), torch.tensor(mask[:,:,:max_length,:max_length], dtype=torch.bfloat16).to('cuda')
        else:
            x, y = tokenize_func(task, return_lengths=False)
            yield numpy2torch(x, max_length), numpy2torch(y, max_length)


def create_arc_attention_mask_oneshot(*lengths, X_attend2_history=False):
    """
    Creates a custom attention mask for ARC examples with parallel decoding.

    Args:
        *lengths: Sequence of integers for segment lengths
                  (x1_len, y1_len, ..., xk_len, yk_len).

    Returns:
        numpy.ndarray: A 2D numpy array of shape (1, 1, total_len, total_len) containing attention mask values.
                     Values are 0 for positions that can attend and -inf for masked positions.

    Rules applied:
    1. Xi attends fully to Xi (within-grid attention).
    2. yi attends fully to all preceding history (X1, y1, ..., Xi-1, yi-1)
       AND its corresponding input Xi.
    3. yi attends ONLY to itself within the yi block (diagonal is True),
       NOT to other tokens within the same yi block (off-diagonal is False).
       This means independent predictions, allowing for parallel decoding.
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

            # Rule 3: yi attends ONLY to itself within its own block
            # We ensure the yi-yi block (mask[start:end, start:end])
            # only has True values on the diagonal.
            indices = np.arange(start, end)
            mask[indices, indices] = True # Allow self-attention
    
    # Convert boolean mask to attention values (0 for attend, -inf for mask)
    out = np.where(mask, 0, -np.inf)
    return out[None][None]

def create_arc_causal_attention_mask(*lengths, X_attend2_history=False):
    """
    Creates a custom attention mask for ARC examples with causal attention for Y segments.

    Args:
        *lengths: Sequence of integers for segment lengths
                  (x1_len, y1_len, ..., xk_len, yk_len).

    Returns:
        numpy.ndarray: A 2D numpy array of shape (total_len, total_len) containing attention mask values.
                     Values are 0 for positions that can attend and -inf for masked positions.

    Rules applied:
    1. Xi attends fully to Xi (within-grid attention).
    2. yi attends fully to all preceding history (X1, y1, ..., Xi-1, yi-1)
       AND its corresponding input Xi.
    3. yi uses causal attention within its own block (like standard LLM),
       meaning each token can only attend to itself and previous tokens in the sequence.
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
            
        elif segment_type == 'Y':
            # Rule 2: yi attends to all history up to and including Xi
            mask[start:end, 0:start] = True

            # Rule 3: yi uses causal attention within its own block
            # Each token can attend to itself and previous tokens
            for i in range(start, end):
                mask[i, start:i+1] = True  # Causal attention pattern
    
    # Convert boolean mask to attention values (0 for attend, -inf for mask)
    out = np.where(mask, 0, -np.inf)
    return out[None][None]

def tokenize_arc_oneshot(task:list[tuple[list[list[int]], list[list[int]]]], return_lengths:bool=True):
    """
    Tokenizes a single ARC task into model input and target sequences.
    Uses distinct placeholder tokens for predicting output rows, columns, and cells.
    
    Args:
        task (list): A list of tuples, each containing (input_grid, output_grid), where each grid
                     is a list of lists of integers (0-9).
        return_lengths (bool): Whether to return sequence lengths
        
    Returns:
        Tuple of:
            - numpy array of input tokens
            - numpy array of target tokens
            - list of lengths [x1_len, y1_len, ...] (if return_lengths=True)
    """
    # Token definitions with direct values
    # 0-9: Grid cell values (digits)
    BOS_X = 10       # Beginning of input grid
    EOS_X = 11       # End of input grid
    BOS_Y = 12       # Beginning of output grid
    EOS_Y = 13       # End of output grid
    PREDICT_ROW_Y = 14  # Placeholder for predicting output rows
    PREDICT_COL_Y = 15  # Placeholder for predicting output columns
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
    lengths = []  # To store lengths of each x and y sequence
    
    for input_grid, output_grid in task:
        # --- Validate and Flatten Input Grid (X) ---
        rows_x, cols_x, flat_x = get_grid_dimensions(input_grid)
        row_token_x = get_dimension_token(rows_x)
        col_token_x = get_dimension_token(cols_x)

        # --- Validate and Flatten Output Grid (Y) ---
        rows_y, cols_y, flat_y = get_grid_dimensions(output_grid)
        row_token_y = get_dimension_token(rows_y) # Actual target token for rows_y
        col_token_y = get_dimension_token(cols_y) # Actual target token for cols_y
        num_output_cells = rows_y * cols_y

        # --- Construct Model Input Sequence ---
        model_input_tokens = []
        model_input_tokens.append(BOS_X)
        model_input_tokens.append(row_token_x)
        model_input_tokens.append(col_token_x)
        model_input_tokens.extend(flat_x) # Add flattened input grid cells (as ints 0-9)
        model_input_tokens.append(EOS_X)
        model_input_tokens.append(BOS_Y)
        
        if return_lengths: # include BOS_Y as each prediction y needs to attend to BOS_Y
            len_x = len(model_input_tokens)
            
        # Append the specific prediction placeholder tokens
        model_input_tokens.append(PREDICT_ROW_Y) # Placeholder for row_y prediction
        model_input_tokens.append(PREDICT_COL_Y) # Placeholder for col_y prediction
        model_input_tokens.extend([PREDICT_CELL_Y] * num_output_cells) # Placeholders for grid_y prediction
        model_input_tokens.append(EOS_Y)
        
        if return_lengths:
            len_y = len(model_input_tokens) - len_x
            
        input_tokens.extend(model_input_tokens)
        
        # --- Construct Model Target Sequence ---
        # Target sequence structure remains the same, aligning with the input placeholders
        model_target_tokens = []
        # Ignore tokens corresponding to the input part (BOS_X to BOS_Y inclusive)
        len_prefix_ignore = 1 + 1 + 1 + len(flat_x) + 1 + 1 # BOS_X, rX, cX, flat_X, EOS_X, BOS_Y
        model_target_tokens.extend([IGNORE_INDEX] * len_prefix_ignore)
        
        # Target for PREDICT_ROW_Y placeholder
        model_target_tokens.append(row_token_y)
        # Target for PREDICT_COL_Y placeholder
        model_target_tokens.append(col_token_y)
        # Targets for PREDICT_CELL_Y placeholders
        model_target_tokens.extend(flat_y) # Add actual flattened output grid cells
        # Ignore the final EOS_Y token in the input sequence
        model_target_tokens.append(IGNORE_INDEX)
        
        target_tokens.extend(model_target_tokens)
        
        if return_lengths:
            lengths.append(len_x)
            lengths.append(len_y)

    if return_lengths:
        return np.array(input_tokens), np.array(target_tokens), lengths
    else:
        return np.array(input_tokens), np.array(target_tokens)
