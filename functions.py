from dataclasses import dataclass
from unsloth import FastModel,FastLanguageModel
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
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
    
class CosSinEmbedding(nn.Module):
    def __init__(self, dim=4096, theta = 30):
        super().__init__()
        freq = 1.0 / (theta ** (torch.arange(0, dim, 4) / dim))[None,:].float()
        pos = torch.arange(0, theta)[:,None].float()
        prod = pos * freq
        cos = torch.cos(prod)
        sin = torch.sin(prod)
        cos_sin = torch.cat([cos, sin], dim=1) # (max_len, dim/2)
        self.register_buffer('cos_sin', cos_sin)

    def forward(self, rows, cols):
        # rows and cols are the row / col index of shape (L) for the flattened grid
        rows_cos_sin = self.cos_sin[rows[0]]
        cols_cos_sin = self.cos_sin[cols[0]]
        return torch.cat([rows_cos_sin, cols_cos_sin], dim=1)[None,:] # (1, L, dim)
    
class FeatureEmbedding2(nn.Module):
    def __init__(self, embed_model, config, input_dim=164, hidden_dim1=4096, hidden_dim2=128, dropout=0.17):
        super().__init__()
        self.embed_model = embed_model
        self.config = config
        d = config.hidden_size
        self.features_MLP = torch.nn.Sequential(torch.nn.Linear(input_dim,hidden_dim1),torch.nn.SiLU(),torch.nn.Linear(hidden_dim1,d))
        self.cos_sin_MLP = torch.nn.Sequential(torch.nn.Linear(d,hidden_dim2),torch.nn.SiLU(),torch.nn.Linear(hidden_dim2,d))
        self.cos_sin_embedding = CosSinEmbedding(dim=d) # for decoding, position embedding
        self.dropout = nn.Dropout(dropout)
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, features, input_tokens, rows, cols):
        features = self.features_MLP(features) # (1, l, d)
        embed_features = self.embed_model(input_tokens) # (1, L, d)
        embed_features = embed_features.masked_scatter((input_tokens==15)[...,None], features) # (1, L, d)
        embed_features = torch.cat([embed_features, self.cos_sin_MLP(self.cos_sin_embedding(rows, cols))], dim=1) # (1, L+l, d)
        embed_features = self.norm(embed_features)
        embed_features = self.dropout(embed_features)
        return embed_features
    
class FeatureEmbedding(nn.Module):
    def __init__(self, embed_model, config, input_dim=162, output_dim=133, dropout=0.17):
        super().__init__()
        self.embed_model = embed_model
        self.config = config
        d = config.hidden_size
        self.input_features_MLP = torch.nn.Sequential(torch.nn.Linear(input_dim,d),torch.nn.SiLU(),torch.nn.Linear(d,d))
        self.output_features_MLP = torch.nn.Sequential(torch.nn.Linear(output_dim,d),torch.nn.SiLU(),torch.nn.Linear(d,d))
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_features, output_features, input_tokens):
        input_features = self.input_features_MLP(input_features)
        output_features = self.output_features_MLP(output_features)
        embed_features = self.embed_model(input_tokens)
        embed_features = embed_features.masked_scatter((input_tokens==15)[...,None], input_features)
        embed_features = embed_features.masked_scatter((input_tokens==16)[...,None], output_features)
        embed_features = self.norm(embed_features)
        embed_features = self.dropout(embed_features)
        return embed_features

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

def generateTransformPara(n, apply_to_output=False):
    """Randomly generates transformation parameters"""
    # n is the number of examples
    # (fliplr, rot90, permutate color, permutate example, enlarge, apply to output)
    return TransformPara(np.random.randint(0, 2), np.random.randint(0, 4), np.random.permutation(10), \
                         np.random.permutation(n),\
                        #  (np.random.randint(1, 3), np.random.randint(1, 3)),\
                        1 if apply_to_output else np.random.randint(0, 2))

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
def numpy2torch(x, dtype=None):
    """Convert numpy array to torch tensor and move to GPU"""
    if dtype is None:
        x = torch.tensor(x)[None].to('cuda')
    else:
        x = torch.tensor(x, dtype=dtype)[None].to('cuda')
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

from features_optimized import extract_features, extract_causal_features
def tokenize_features3(task, max_length, background_color,IsDecode=False, max_k=5, random_context=True):
    # used for separate head for size prediction
    BOS_X = 10  # Beginning of input grid
    EOS_X = 11  # End of input grid
    LINE_BREAK = 12  # Row separator
    BOS_Y = 13  # Beginning of output grid
    EOS_Y = 14  # End of output grid
    INPUT_PLACEHOLDER = 15
    input_tokens = []
    sizes = [] # (l, 2), where l is the number of examples in task
    length = [] # (len(input1), len(output1), ..., len(input_l))
    features = []
    # randomly sample how many tasks to use, various context length can help generalization
    if random_context:
        r = np.random.randint(2, len(task)+1)
    else:
        r = len(task)
    n_task = min(find_first_exceed(task, max_length), r)
    def get_token_from_grid(grid, bos_token, eos_token, placeholder_token, line_break_token):
        n,m = len(grid), len(grid[0])
        tokens = []
        tokens.append(bos_token)
        line = [placeholder_token] * len(grid[0])
        line.append(line_break_token)
        for _ in grid:
            tokens.extend(line)
        tokens.append(eos_token)
        return tokens, [n-1, m-1] # -1 for 0-indexed for embedding
    if IsDecode:
        # For decoding, must include the last task
        task = task[:n_task-1] + [task[-1]] 
    else:
        task = task[:n_task]
    current_length = -1 # zero-indexed
    for x, y in task[:n_task-1]:
        # Extract features
        input_feature = extract_features(x, max_k=max_k, background_color=background_color) # (l, d)
        output_feature = extract_features(y, max_k=max_k, background_color=background_color)
        rnd = np.random.rand()
        l1, l2 = input_feature.shape[0], output_feature.shape[0]
        features.append(np.concatenate([input_feature, np.zeros((l1,1)), np.ones((l1,1)) * rnd],1))
        features.append(np.concatenate([output_feature, np.ones((l2,1)), np.ones((l2,1)) * rnd],1))        
        # Tokenize input
        token, size = get_token_from_grid(x, BOS_X, EOS_X, INPUT_PLACEHOLDER, LINE_BREAK)
        input_tokens.extend(token)
        sizes.append(size)
        current_length += len(token)
        length.append(current_length)
        # Tokenize output
        token, size = get_token_from_grid(y, BOS_Y, EOS_Y, INPUT_PLACEHOLDER, LINE_BREAK)
        input_tokens.extend(token)
        sizes.append(size)
        current_length += len(token)
        length.append(current_length)
    # Tokenize last task
    x, y = task[-1]
    # Tokenize input
    input_feature = extract_features(x, max_k=max_k, background_color=background_color) # (l, d)
    rnd = np.random.rand()
    l1 = input_feature.shape[0]
    features.append(np.concatenate([input_feature, np.zeros((l1,1)), np.ones((l1,1)) * rnd],1))
    token, size = get_token_from_grid(x, BOS_X, EOS_X, INPUT_PLACEHOLDER, LINE_BREAK)
    input_tokens.extend(token)
    sizes.append(size)
    current_length += len(token)
    length.append(current_length)
    features = np.concatenate(features, axis=0)
    sizes = np.stack(sizes, axis=0)
    length = np.array(length)
    target_tokens = []
    if IsDecode:
        # TODO: add size prediction
        raise NotImplementedError
    else:
        row, col = len(y), len(y[0])
        rows = [r for r in range(row) for _ in range(col)]
        cols = [c for _ in range(row) for c in range(col)]
        for flat_y in y:
            target_tokens.extend(flat_y)
        return torch.tensor(input_tokens), torch.tensor(target_tokens), torch.tensor(features, dtype=torch.bfloat16), torch.tensor(rows), torch.tensor(cols),\
               torch.tensor(sizes), torch.tensor(length), torch.tensor([row-1, col-1]) # -1 for 0-indexed for embedding
    
def tokenize_features2(task, max_length, background_color,IsDecode=False, max_k=5, random_context=True):
    # only use non-causal features and predict rows, cols, and cells in outputN in one-shot.
    BOS_X = 10  # Beginning of input grid
    EOS_X = 11  # End of input grid
    LINE_BREAK = 12  # Row separator
    BOS_Y = 13  # Beginning of output grid
    EOS_Y = 14  # End of output grid
    INPUT_PLACEHOLDER = 15
    PREDICT_ROW = 16
    PREDICT_COL = 17
    SIZE_OFFSET = 17 # size goes from 1 to 30 mapping to token space as 18 to 47
    PAD_TOKEN = -100  # Padding/ignored token

    input_tokens = []
    features = []
    # randomly sample how many tasks to use, various context length can help generalization
    if random_context:
        r = np.random.randint(2, len(task)+1)
    else:
        r = len(task)
    n_task = min(find_first_exceed(task, max_length), r)
    def get_token_from_grid(grid, bos_token, eos_token, placeholder_token, line_break_token):
        n,m = len(grid), len(grid[0])
        tokens = []
        tokens.append(bos_token)
        line = [placeholder_token] * len(grid[0])
        line.append(line_break_token)
        for _ in grid:
            tokens.extend(line)
        tokens.append(eos_token)
        tokens.append(n + SIZE_OFFSET)
        tokens.append(m + SIZE_OFFSET)
        return tokens
    if IsDecode:
        # For decoding, must include the last task
        task = task[:n_task-1] + [task[-1]] 
    else:
        task = task[:n_task]
    for x, y in task[:n_task-1]:
        # Extract features
        input_feature = extract_features(x, max_k=max_k, background_color=background_color) # (l, d)
        output_feature = extract_features(y, max_k=max_k, background_color=background_color)
        rnd = np.random.rand()
        l1, l2 = input_feature.shape[0], output_feature.shape[0]
        features.append(np.concatenate([input_feature, np.zeros((l1,1)), np.ones((l1,1)) * rnd],1))
        features.append(np.concatenate([output_feature, np.ones((l2,1)), np.ones((l2,1)) * rnd],1))        
        # Tokenize input
        token = get_token_from_grid(x, BOS_X, EOS_X, INPUT_PLACEHOLDER, LINE_BREAK)
        input_tokens.extend(token)
        
        # Tokenize output
        token = get_token_from_grid(y, BOS_Y, EOS_Y, INPUT_PLACEHOLDER, LINE_BREAK)
        input_tokens.extend(token)
    # Tokenize last task
    x, y = task[-1]
    # Tokenize input
    input_feature = extract_features(x, max_k=max_k, background_color=background_color) # (l, d)
    rnd = np.random.rand()
    l1 = input_feature.shape[0]
    features.append(np.concatenate([input_feature, np.zeros((l1,1)), np.ones((l1,1)) * rnd],1))
    token = get_token_from_grid(x, BOS_X, EOS_X, INPUT_PLACEHOLDER, LINE_BREAK)
    input_tokens.extend(token)
    features = np.concatenate(features, axis=0)

    input_tokens.append(PREDICT_ROW)
    input_tokens.append(PREDICT_COL)
    if IsDecode:
        if y is None: # leaderboard
            return torch.tensor(input_tokens), torch.tensor(features, dtype=torch.bfloat16)
        else: # local test
            return torch.tensor(input_tokens), torch.tensor(features, dtype=torch.bfloat16), torch.tensor(y)
    else:
        target_tokens = []
        row, col = len(y), len(y[0])
        rows = [r for r in range(row) for _ in range(col)]
        cols = [c for _ in range(row) for c in range(col)]
        target_tokens.append(row-1) # -1 for 0-indexed
        target_tokens.append(col-1) # -1 for 0-indexed
        for flat_y in y:
            target_tokens.extend(flat_y)
        return torch.tensor(input_tokens), torch.tensor(target_tokens), torch.tensor(features, dtype=torch.bfloat16), torch.tensor(rows), torch.tensor(cols)

def tokenize_features(task, max_length, background_color,IsDecode=False, max_k=5):
    BOS_X = 10  # Beginning of input grid
    EOS_X = 11  # End of input grid
    LINE_BREAK = 12  # Row separator
    BOS_Y = 13  # Beginning of output grid
    EOS_Y = 14  # End of output grid
    INPUT_PLACEHOLDER = 15
    OUTPUT_PLACEHOLDER = 16
    PAD_TOKEN = -100  # Padding/ignored token

    input_tokens = []
    target_tokens = []
    input_features = []
    output_features = []
    n_task = find_first_exceed(task, max_length)
    def get_token_from_grid(grid, bos_token, eos_token, placeholder_token, line_break_token, IsOutput):
        tokens = [bos_token]
        targets = [PAD_TOKEN]
        line = [placeholder_token] * len(grid[0])
        line.append(line_break_token)
        for row in grid:
            tokens.extend(line)
            if IsOutput:
                targets.extend(row)
                targets.append(line_break_token)
        tokens.append(eos_token)
        if IsOutput:
            targets.append(eos_token)
            targets = targets[1:]
            targets.append(PAD_TOKEN)
        else:
            targets = [PAD_TOKEN] * len(tokens)
        assert len(tokens) == len(targets), f"token length and target length mismatch, tokens: {len(tokens)}, targets: {len(targets)}"
        return tokens, targets
    if IsDecode:
        # For decoding, must include the last task
        task = task[:n_task-1] + [task[-1]] 
    else:
        task = task[:n_task]
    for x, y in task:
        # Extract features
        input_feature = extract_features(x, background_color, max_k) # (l, d)
        output_feature = extract_causal_features(y, background_color, max_k)
        input_features.append(input_feature)
        output_features.append(output_feature)
        # Tokenize input and targets
        token, target = get_token_from_grid(x, BOS_X, EOS_X, INPUT_PLACEHOLDER, LINE_BREAK, False)
        input_tokens.extend(token)
        target_tokens.extend(target)
        token, target = get_token_from_grid(y, BOS_Y, EOS_Y, OUTPUT_PLACEHOLDER, LINE_BREAK, True)
        input_tokens.extend(token)
        target_tokens.extend(target)
    input_features = np.concatenate(input_features, axis=0)
    output_features = np.concatenate(output_features, axis=0)
    return torch.tensor(input_tokens), torch.tensor(target_tokens), torch.tensor(input_features, dtype=torch.bfloat16), torch.tensor(output_features, dtype=torch.bfloat16)
    

def tokenize_causal(task, autoregressive: bool, max_length, IsDecode=False, NeedPosition: bool = False, ReturnLengths: bool = False, offset1: int = 0, offset2: int = 0, background_color: int = 0):
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
    if offset2 == 0:
        global_r, global_c = 1, 1
    else:
        global_r, global_c = offset2, offset2 # special token has (0,0). So we start from offset2 to differentiate special token.
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
                row_indices.extend([r_idx + global_r] * row_len)
                col_indices.extend(list(range(global_c, row_len + global_c)))

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

        # separate out input and output grid    
        global_r += offset1
        global_c += offset1

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
                    row_indices.extend([r_idx + global_r] * row_len)
                    col_indices.extend(list(range(global_c, row_len + global_c)))

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
            
            # separate out different examples
            global_r += offset2
            global_c += offset2
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

def data_gen(data, IsTrain, tokenize_func, apply_to_output=False, **kwargs):
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
            para = generateTransformPara(len(task), apply_to_output=apply_to_output)
            task = forwardTask(task, para)
            kwargs['background_color'] = para.perm_color[0]
        else:
            kwargs['background_color'] = 0
        # Tokenize the task
        out = tokenize_func(task, **kwargs)
        yield out

class CustomDataset(Dataset):    
    def __init__(self, data, IsTrain, tokenize_func, apply_to_output=False, **kwargs):
        self.data_source = data['train'] if IsTrain else data['test']
        self.is_train = IsTrain
        self.tokenize_func = tokenize_func
        self.apply_to_output = apply_to_output
        self.kwargs = kwargs
        
    def __len__(self):
        return len(self.data_source)
    
    def __getitem__(self, idx):
        task = self.data_source[idx]
        if self.is_train:
            para = generateTransformPara(len(task), apply_to_output=self.apply_to_output)
            task = forwardTask(task, para)
            self.kwargs['background_color'] = para.perm_color[0]
        else:
            self.kwargs['background_color'] = 0
        out = self.tokenize_func(task, **self.kwargs)
        return out
    
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

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_attention_patterns(pattern, layers=0, num_heads=32, rows=4, cols=8, figsize=(20, 10), title='Attention Patterns for Different Heads'):
    """
    Plot attention patterns for different heads.
    
    Args:
        pattern: Tensor containing attention patterns for different heads
        num_heads: Number of attention heads to plot
        rows: Number of rows in the subplot grid
        cols: Number of columns in the subplot grid
        figsize: Figure size as (width, height)
        title: Title for the overall figure
    """
    pattern = pattern[layers]
    # Create a figure with subplots for different heads
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    # Plot each head's pattern
    for h in range(num_heads):
        ax = axes[h]
        im = ax.imshow(pattern[h].detach().cpu().numpy(), cmap='viridis')
        ax.set_title(f'Head {h}')
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        
        # Add colorbar for each subplot
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    # Adjust layout
    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.92)
    return fig