from dataclasses import dataclass
import numpy as np
import random
import torch
from typing import List, Tuple, Optional
from unsloth import FastModel,FastLanguageModel
import gc
from peft import PeftModel

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

''' Model and tokenization utilities '''

def get_gemma_model(model_name, head_dim, lm_head_path, peft_path, isTrain, max_seq_length = 8192):
    model, _ = FastModel.from_pretrained(model_name = model_name,
                                         max_seq_length = max_seq_length,
                                         load_in_4bit = True,
                                         resize_model_vocab=head_dim,
                                        )
    del model.vision_tower
    model = model.base_model
    if isTrain:
        model.lm_head.weight.requires_grad_(True);
    gc.collect()
    torch.cuda.empty_cache()
    model.lm_head.load_state_dict(torch.load(lm_head_path))
    model = PeftModel.from_pretrained(model, peft_path, is_trainable=isTrain)
    if not isTrain:
        FastLanguageModel.for_inference(model);
    return model

def numpy2torch(x):
    """Convert numpy array to torch tensor and move to GPU, with length truncation."""
    x = torch.tensor(x[None]).to('cuda')
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

def tokenize_causal(task, autoregressive:bool, max_length, IsDecode=False):
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
        max_length: Maximum sequence length for truncation.
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
    n_task = find_first_exceed(task, max_length)
    if IsDecode:
        # For decoding, must include the last task
        task = task[:n_task-1] + [task[-1]] 
    else:
        task = task[:n_task]
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
    if not IsDecode:
        if autoregressive:
            target_tokens = input_tokens[1:] + [PAD_TOKEN]
        else:
            target_tokens = target_tokens[1:] + [PAD_TOKEN]
    
    # Convert to numpy arrays
    return np.array(input_tokens), np.array(target_tokens) if target_tokens else None

def tokenize_oneshot(task:list[tuple[list[list[int]], list[list[int]]]], \
                     max_length:int,\
                     IsDecode:bool, autoregressive:bool):
    """
    Tokenizes one-shot prediction for ARC tasks.
    Uses distinct placeholder tokens for predicting output cells.
    when IsDecode is true, return starting point for decoding, [x1,y1,x2,y2,...xk,BOS_Y]
    needs to run autoregressive to generate row and col prediction,
    then needs to append [PREDICT_CELL_Y] * (rows_y * cols_y) to input_tokens
    for the final oneshot prediction
    Args:
        task (list): A list of tuples, each containing (input_grid, output_grid), where each grid
                     is a list of lists of integers (0-9)
        IsDecode (bool): Whether this tokenization is for decoding (inference) or training
        autoregressive (bool): Whether to train model on x,y or just on last y (oneshot)
        max_length (int, optional): Maximum sequence length to consider for tokenization
        
    Returns:
        If IsDecode=True:
            - numpy array of input tokens
            - 2d output_grid for the task
        If IsDecode=False:
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

    def get_dimension_token(dim_size):
        """Gets the token ID for a given dimension size."""
        return SIZE_TOKEN_OFFSET + dim_size

    IGNORE_INDEX = -100
    input_tokens = []
    if not IsDecode:
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
        input_tokens.append(BOS_X)
        input_tokens.append(row_token_x)
        input_tokens.append(col_token_x)
        input_tokens.extend(flat_x) # Add flattened input grid cells (as ints 0-9)
        input_tokens.append(EOS_X)
        
        # append the output grid
        input_tokens.append(PREDICT_ROW_Y)
        input_tokens.append(row_token_y)
        input_tokens.append(PREDICT_COL_Y)
        input_tokens.append(col_token_y)
        input_tokens.append(BOS_Y)
        input_tokens.extend(flat_y)
        input_tokens.append(EOS_Y)
        
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
    rows_x, cols_x, flat_x = get_grid_dimensions(input_grid)
    row_token_x = get_dimension_token(rows_x)
    col_token_x = get_dimension_token(cols_x)
    if output_grid is not None:
        rows_y, cols_y, flat_y = get_grid_dimensions(output_grid)
        row_token_y = get_dimension_token(rows_y)
        col_token_y = get_dimension_token(cols_y)
    # append the input grid, same as before
    input_tokens.append(BOS_X)
    input_tokens.append(row_token_x)
    input_tokens.append(col_token_x)
    input_tokens.extend(flat_x) # Add flattened input grid cells (as ints 0-9)
    input_tokens.append(EOS_X)
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
    if IsDecode:
        return np.array(input_tokens), np.array(output_grid) if output_grid is not None else None
    else:
        target_tokens.append(row_token_y)
        input_tokens.append(row_token_y)
        target_tokens.append(-100)
        input_tokens.append(PREDICT_COL_Y)
        target_tokens.append(col_token_y)
        input_tokens.append(col_token_y)
        target_tokens.append(-100)

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
        out = tokenize_func(task, autoregressive=autoregressive, IsDecode=IsDecode, max_length=max_length)
        if len(out) == 2: 
            yield numpy2torch(out[0]), out[1] if IsDecode else numpy2torch(out[1])
        else: # tokenize_oneshot return input, output, and length
            yield numpy2torch(out[0]), out[1] if IsDecode else numpy2torch(out[1]), out[2]

class OneshotDecoder(object):
    def __init__(self, model, max_dim=30):
        """
        The OneshotDecoder class decodes an tokenized input sequence into a 2D output grid using a pre-trained model.
        
        This class employs a two-step process:
        1. **Dimension Prediction**: predict the dimensions (number of rows and columns) of the output grid.
        2. **Grid Prediction**: Once the dimensions are determined, it generates the entire grid in a single forward pass of the model.
        """
        self.model = model
        self.max_dim = max_dim
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
    def predict_dimensions(self, current_ids, current_nll=0.0, past_key_values=None, current_depth=0):
        """
        Performs Depth-First Search to predict row and column dimensions up to depth 2.
        
        Args:
            current_ids (torch.Tensor): Current sequence of token IDs, shape (1, seq_len) on cuda.
            current_nll (float): Accumulated negative log likelihood of the sequence.
            past_key_values: Cached key/value pairs from previous model calls for efficiency.
            current_depth (int): Current depth in the DFS recursion.
        """
        model = self.model
        device = model.device
        ### Step 1: Predict row_token_y after PREDICT_ROW_Y
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model(
                input_ids=current_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        row_token_y = torch.argmax(next_token_logits, dim=-1).item()  # Predicted row dimension token
        assert self.SIZE_TOKEN_OFFSET + 1 <= row_token_y <= self.SIZE_TOKEN_OFFSET + self.max_dim, \
            f"Row token {row_token_y} out of range ({self.SIZE_TOKEN_OFFSET + 1}, {self.SIZE_TOKEN_OFFSET + self.max_dim})"
        self.rows = row_token_y - self.SIZE_TOKEN_OFFSET
        
        # Step 2: Predict col_token_y after PREDICT_COL_Y
        # Append row_token_y and PREDICT_COL_Y as input_ids
        input_ids = torch.tensor([[row_token_y, self.PREDICT_COL_Y]], dtype=torch.long, device=device)
        
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
        assert self.SIZE_TOKEN_OFFSET + 1 <= col_token_y <= self.SIZE_TOKEN_OFFSET + self.max_dim, \
            f"Column token {col_token_y} out of range ({self.SIZE_TOKEN_OFFSET + 1}, {self.SIZE_TOKEN_OFFSET + self.max_dim})"
        self.cols = col_token_y - self.SIZE_TOKEN_OFFSET
        
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
        self.predict_dimensions(input_tokens)

        # Step 2: Predict the output grid
        output_grid = self.predict_output_grid()

        return output_grid

class CausalDecoder(object):
    def __init__(self, model, max_depth: int = 31 * 30 + 1, brunch_factor: int = 3):
        """Initialize the searcher with a pre-trained model."""

        self.max_depth = max_depth
        self.model = model
        self.brunch_factor = brunch_factor
        self.reset()  # Initialize/reset state variables

    def reset(self):
        """Reset the searcher state."""
        self.best_paths = []
        self.nlls = []
        self.min_nll = float('inf')

    def decode(self, input_tokens):
        self.reset
        self.dfs_generate(input_tokens)
        idx = np.argmin(self.nlls)
        best_path = self.best_paths[idx]
        return self.detokenize_causal(best_path)

    @staticmethod
    def detokenize_causal(tokens):
        """
        Detokenizes a 1D sequence of tokens (after BOS_Y) back into a 2D grid.

        This function is the inverse of the `tokenize_causal` function, specifically
        for the decoding phase (IsDecode=True).  It assumes the input `tokens`
        starts *after* the BOS_Y token.

        Args:
            tokens: A 1D list of integer tokens.

        Returns:
            A 2D numpy array of integers
        """
        LINE_BREAK = 12
        EOS_Y = 14

        grid = []
        row = []
        for token in tokens:
            if token == EOS_Y:
                if row:
                    grid.append(row)
                break  # Stop at EOS_Y
            elif token == LINE_BREAK:
                grid.append(row)
                row = []
            else:
                row.append(token)
        return np.array(grid)
    
    @staticmethod
    def check_equal_line_lengths(tensor):
        """ Check if all lines in a torch tensor have the same length when called at a line break position."""
        tensor = tensor[0]
        if tensor[-1].item() != 12: # only check if the last token is a line break
            return True
        idx = (tensor == 12).nonzero(as_tuple=True)[0]
        if len(idx) <= 1:
            return True
        return len(torch.unique(idx[1:] - idx[:-1])) == 1
        
    @torch.no_grad()
    def dfs_generate(self, current_ids, current_nll = 0, past_key_values = None, current_depth = 0):
        """Performs Depth-First Search to find the lowest NLL completion."""
        # current_ids is torch.Tensor of Shape: (1, seq_len)
        model = self.model
        max_depth = self.max_depth
        device = model.device
        # Safety check for recursion depth
        if current_depth > max_depth:
            return

        # Prepare inputs for the model
        if past_key_values is None:
            # First call, process the whole sequence
            input_ids_step = current_ids
        else:
            # Subsequent calls, only process the last token
            input_ids_step = current_ids[:, -1:]

        # Get logits and new KV cache
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids_step,
                past_key_values=past_key_values,
                use_cache=True, # Crucial for efficiency
            )

        logits = outputs.logits.float() # Shape: (1, input_seq_len, vocab_size)
        new_past_key_values = outputs.past_key_values # Updated KV cache

        # Get logits for the *next* token prediction
        # Logits shape is (batch_size, sequence_length, vocab_size)
        next_token_logits = logits[:, -1, :]

        # Calculate log probabilities and negative log likelihoods
        log_probs = torch.log_softmax(next_token_logits, dim=-1)
        nlls = -log_probs # Shape: (1, vocab_size)

        # Sort potential next tokens by NLL (ascending)
        # We only need to explore the top `branching_factor` candidates
        sorted_nlls, sorted_indices = torch.sort(nlls.squeeze(), descending=False)
        sorted_nlls = sorted_nlls[:self.brunch_factor]
        sorted_indices = sorted_indices[:self.brunch_factor]
        
        # Iterate through the most promising next tokens
        for next_token_id, next_token_nll in zip(sorted_indices, sorted_nlls):
            next_token_id = next_token_id.item()
            next_token_nll = next_token_nll.item()

            potential_total_nll = current_nll + next_token_nll

            # --- Pruning ---
            if potential_total_nll >= self.min_nll:
                # If the current path's NLL is already worse than the best complete
                # sequence found so far, prune this branch.
                break

            
            if past_key_values is None: # first call
                next_ids = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
            else: # Append the chosen token
                next_ids = torch.cat(
                    [current_ids, torch.tensor([[next_token_id]], dtype=torch.long, device=device)],
                    dim=1
                )
            
            if not self.check_equal_line_lengths(next_ids):
                # If the line lengths are not equal, prune this branch
                continue
            
            # --- Base Case: EOS token ---
            if next_token_id == 14 and next_ids[0][-1].item() == 12: # Line break followed by EOS
                # print(f"Found EOS. Path NLL: {potential_total_nll:.4f} | Path Len: {current_depth}")
                self.best_paths.append(next_ids[0].tolist())
                self.nlls.append(potential_total_nll)
                self.min_nll = min(self.min_nll, potential_total_nll)
                # Continue searching other branches even after finding an EOS
                continue # Don't recurse further down this path

            # --- Recursive Step ---
            # Pass the `new_past_key_values` which contains the cache state *after*
            self.dfs_generate(current_ids=next_ids,
                              current_nll=potential_total_nll,
                              past_key_values=new_past_key_values,
                              current_depth=current_depth + 1,
                             )
        # del new_past_key_values, past_key_values, logits, next_token_logits, log_probs, nlls
        # gc.collect()
        # torch.cuda.empty_cache()