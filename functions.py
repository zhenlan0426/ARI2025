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

def tokenize_causal(task, autoregressive:bool, IsDecode=False):
    """
    Tokenizes a task for causal (autoregressive) training or inference.
    
    For training, converts a sequence of (input, output) grid pairs into a flat token sequence
    where each token can attend to all previous tokens. The sequence follows the pattern:
    (x1, y1, x2, y2, ..., xN, yN) with appropriate special tokens.
    
    For decoding, formats the input as (x1, y1, x2, y2, ..., xN, BOS_Y) to predict the final output.
    
    Args:
        task: List of (input_grid, output_grid) tuples.
              Each grid is a 2D list of integers.
              For decoding, the last output_grid can be None.
        autoregressive: Whether to use autoregressive training mode.
                        If True, both inputs and outputs predict the next token.
                        If False, only output tokens predict the next output token.
        IsDecode: Whether the function is being used for inference (True) or training (False).

    Returns:
        input_tokens: Numpy array of input token IDs.
                      For training: (x1, y1, x2, y2, ... xN, yN)
                      For decoding: (x1, y1, x2, y2, ..., xN, BOS_Y)
        final_target: Depends on IsDecode.
                      For training: Numpy array of shifted target token IDs for next-token prediction.
                      For decoding: The raw 2D grid (yN) or None.
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
    flag = not IsDecode and not autoregressive
    n = len(task)
    for i, (x, y) in enumerate(task):
        IsLast = (i == n-1) and IsDecode
        # Process input grid (x)
        
        input_tokens.append(BOS_X)
        if flag:
            target_tokens.append(PAD_TOKEN)
        
        for row in x:
            # Add row elements
            input_tokens.extend(row)
            if flag:
                target_tokens.extend([PAD_TOKEN]*len(row))

            input_tokens.append(LINE_BREAK)
            if flag:
                target_tokens.append(PAD_TOKEN)

        input_tokens.append(EOS_X)
        if flag:
            target_tokens.append(PAD_TOKEN)
        
        # Process output grid (y)
        
        input_tokens.append(BOS_Y)
        if flag:
            target_tokens.append(PAD_TOKEN)  # Mask BOS_Y
        
        if not IsLast:
            for row in y:
                # Add row elements
                input_tokens.extend(row)
                if flag:
                    target_tokens.extend(row)  # Keep y values in target
        
                input_tokens.append(LINE_BREAK)
                if flag:
                    target_tokens.append(LINE_BREAK)

            input_tokens.append(EOS_Y)
            if flag:
                target_tokens.append(EOS_Y)  # Include EOS_Y in target
        else:
            target_tokens = y        
        
    # Create shifted targets (for next-token prediction)
    if IsDecode:
        target_tokens = y
    else:
        if autoregressive:
            target_tokens = input_tokens[1:] + [PAD_TOKEN]
        else:
            target_tokens = target_tokens[1:] + [PAD_TOKEN]
    
    # Convert to numpy arrays
    return np.array(input_tokens), np.array(target_tokens) if target_tokens else None

def parse_causal_y(input_tokens):
    # TODO: Implement this
    pass

def tokenize_oneshot(task:list[tuple[list[list[int]], list[list[int]]]], \
                         is_decode:bool, autoregressive:bool, max_length:int=None):
    """
    Tokenizes one-shot prediction for ARC tasks.
    Uses distinct placeholder tokens for predicting output cells.
    when is_decode is true, return starting point for decoding, [x1,y1,x2,y2,...xk,BOS_Y]
    needs to run autoregressive to generate row and col prediction,
    then needs to append [PREDICT_CELL_Y] * (rows_y * cols_y) to input_tokens
    for the final oneshot prediction
    Args:
        task (list): A list of tuples, each containing (input_grid, output_grid), where each grid
                     is a list of lists of integers (0-9)
        is_decode (bool): Whether this tokenization is for decoding (inference) or training
        autoregressive (bool): Whether to train model on x,y or just on last y (oneshot)
        max_length (int, optional): Maximum sequence length to consider for tokenization
        
    Returns:
        If is_decode=True:
            - numpy array of input tokens
            - 2d output_grid for the task
        If is_decode=False:
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
    if not is_decode:
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
        if not is_decode:
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
    if output_grid is not None:
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
    if not is_decode:
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
    if is_decode:
        return np.array(input_tokens), output_grid
    else:
        target_tokens.append(row_token_y)
        input_tokens.append(row_token_y)
        target_tokens.append(col_token_y)

        input_tokens.append(col_token_y)
        target_tokens.append(IGNORE_INDEX)

        input_tokens.extend([PREDICT_CELL_Y] * (rows_y * cols_y))
        target_tokens.extend(flat_y)
        return np.array(input_tokens), np.array(target_tokens), len_input

def data_gen(data, IsTrain, max_length, autoregressive, tokenize_func=tokenize_causal, IsDecode=False):
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
        out = tokenize_func(task, autoregressive=autoregressive, IsDecode=IsDecode)
        if len(out) == 2: 
            yield numpy2torch(out[0], max_length), numpy2torch(out[1], max_length)
        else: # tokenize_oneshot return input, output, and length
            yield numpy2torch(out[0], max_length), numpy2torch(out[1], max_length), out[2]