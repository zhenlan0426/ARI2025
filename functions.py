from dataclasses import dataclass
import numpy as np
import random
import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from unsloth import FastModel,FastLanguageModel
import gc
import os
import shutil
import time
import re
import json
from dataclasses import asdict
from dataclasses import dataclass, field
from peft import PeftModel

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
    """Convert numpy array to torch tensor and move to GPU, with length truncation."""
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

def tokenize_causal(task, autoregressive: bool, max_length, IsDecode=False, NeedPosition: bool = False):
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

    Returns:
        If NeedPosition is False:
            input_tokens: Numpy array of input token IDs.
            final_target: Numpy array of shifted target token IDs (training) or raw grid (decoding).
        If NeedPosition is True:
            input_tokens: Numpy array of input token IDs.
            final_target: Same as above.
            row_indices: Numpy array of row indices.
            col_indices: Numpy array of column indices.
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
        else:
            target_tokens = y
        
    # Create shifted targets (for next-token prediction)
    if not IsDecode:
        if autoregressive:
            target_tokens = input_tokens[1:] + [PAD_TOKEN]
        else:
            target_tokens = target_tokens[1:] + [PAD_TOKEN]
    
    # Convert to numpy arrays
    if NeedPosition:
        return {"input_tokens":numpy2torch(input_tokens), "target_tokens": numpy2torch(target_tokens) if target_tokens is not None else None, \
                "row_indices": numpy2torch(row_indices), "col_indices": numpy2torch(col_indices)}
    else:
        return {"input_tokens":numpy2torch(input_tokens), "target_tokens":numpy2torch(target_tokens) if target_tokens is not None else None}

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

def data_gen(data, IsTrain, max_length, autoregressive, NeedPosition, tokenize_func=tokenize_causal, IsDecode=False):
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
        out = tokenize_func(task, autoregressive=autoregressive, IsDecode=IsDecode, max_length=max_length, NeedPosition=NeedPosition)
        yield out

def data_gen_VLM(data, IsTrain, processor, max_pairs, decode=False):
    """Generate data for VLM
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
        out = tokenize_VLM(task, processor, max_pairs=max_pairs, decode=decode)
        yield out

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
        decode: If True, the function prepares input for generation/decoding. The target
                returned will be the raw final `output_grid` instead of processed target IDs.
                If False (default), the function prepares input and labels for training,
                and the target returned will be a tensor of target token IDs.

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
            - If `decode` is True: The raw `output_grid` (List[List[int]]) from the final pair.

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
    if not decode:
        image, ids, target, type = process_all(output_grid, isInput=False, multiplier=multiplier)
        images.append(image)
        input_ids.extend(ids)
        target_ids.extend(target)
        token_type_ids.extend(type)
    else:
        target_ids = output_grid
        
    return {'input_ids': numpy2torch(input_ids), \
            'pixel_values': images, \
            'token_type_ids': numpy2torch(token_type_ids), \
            'attention_mask': numpy2torch([1] * len(input_ids)), \
           },\
           target_ids if decode else numpy2torch(target_ids)

# def tokenize_VLM(task, processor, max_pairs=4):
#     """
#     Encodes an ARC AGI task for a VLM.
    
#     Args:
#         task: List of tuples [(input1, output1), (input2, output2), ...], where each input and output
#               is a 2D grid (list of lists) of integers between 0 and 9.
#         processor: A processor object for encoding images and text.
    
#     Returns:
#         tuple: (images, task_str)
#             - images: List of numpy arrays [input1_image, output1_image, input2_image, ...], each of shape (H, W, 3)
#             - task_str: String representation of the task with image placeholders
#     """
    
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
    def __init__(self, model, max_depth: int = 31 * 30 + 1, brunch_factor: int = 3, IsDebug=False):
        """Initialize the searcher with a pre-trained model."""

        self.max_depth = max_depth
        self.model = model
        self.brunch_factor = brunch_factor
        self.IsDebug = IsDebug
        self.reset()  # Initialize/reset state variables

    def reset(self):
        """Reset the searcher state."""
        self.best_paths = []
        self.nlls = []
        self.min_nll = float('inf')

    def decode(self, input_tokens):
        self.reset()
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
        """ Check if lenth of last line is the same as first line."""
        tensor = tensor[0]
        if tensor[-1].item() != 12: # only check if the last token is a line break
            return True
        idx = (tensor == 12).nonzero(as_tuple=True)[0]
        if len(idx) <= 1:
            return True
        return (idx[-2] - idx[-1] - 1) == idx[0]
    
    @staticmethod
    def check_row_col_len(tensor, max_col=30, max_row=30, line_break_token=12):
        # Find all line break positions
        tensor = tensor[0]
        line_breaks = (tensor == line_break_token).nonzero(as_tuple=True)[0]
        num_rows = len(line_breaks)
        if tensor[-1] != line_break_token:
            num_rows += 1 # ongoing line
        if num_rows > max_row:
            return True  # Too many rows
        # Find start of last row (after previous line break or at 0)
        if num_rows == 0:
            last_row_start = 0
        else:
            last_row_start = line_breaks[-1].item() + 1
        last_row_end = len(tensor)
        last_row_length = last_row_end - last_row_start
        if last_row_length > max_col:
            return True  # Last row is too long
        return False  # All checks passed

    @torch.no_grad()
    def dfs_generate(self, current_ids, current_nll = 0, past_key_values = None, current_depth = 0):
        """Performs Depth-First Search to find the lowest NLL completion."""
        # current_ids is torch.Tensor of Shape: (1, seq_len)
        if self.IsDebug:
            print(f"Current NLL: {current_nll:.4f} | Path Len: {current_depth} | Current IDs: {current_ids[0].tolist()}")
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

        new_past_key_values = outputs.past_key_values # Updated KV cache

        # Get logits for the *next* token prediction
        next_token_logits = outputs.logits[:, -1, :].float() # Shape: (1, input_seq_len, vocab_size)

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
            if next_token_id == 14: 
                if next_ids[0][-1].item() == 12: # Line break followed by EOS
                    # print(f"Found EOS. Path NLL: {potential_total_nll:.4f} | Path Len: {current_depth}",next_ids[0].tolist())
                    self.best_paths.append(next_ids[0].tolist())
                    self.nlls.append(potential_total_nll)
                    self.min_nll = min(self.min_nll, potential_total_nll)
                # Continue searching other branches after finding an EOS for this path
                continue # Don't recurse further down this path
            
            if self.check_equal_line_lengths(next_ids):
                # skip going deeper when too many rows or cols
                continue
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
