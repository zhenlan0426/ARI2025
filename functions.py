from dataclasses import dataclass
import numpy as np
import random
import torch

''' dataset processing '''
@dataclass
class TransformPara:
    fliplr: int          # Flip left-right (0 or 1)
    rot90: int           # Number of 90-degree rotations (0 to 3)
    perm_color: np.ndarray  # Color permutation array
    perm_example: np.ndarray  # Example permutation array
    # enlarge: tuple[int, int]  # Enlarge factors (n, m). disabled due to longer input length performance issue
    apply_to_output: int     # Apply transformations to y or not (except color)

def enlarge_grid_n_times(grid, n, m):
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

def backwardTask(task,tpara):
    pass
    # return [(backward(x,tpara), backward(y,tpara)) for x,y in task]

def tokenize_task(task, return_lengths=False):
    """Tokenizes an ARC task into input/target sequences with special tokens."""
    # TODO: for inference, last task only has x, it should be BOS_X | X | EOS_X | BOS_Y
    # Special token IDs
    BOS_X = 10
    EOS_X = 11
    LINE_BREAK = 12
    BOS_Y = 13
    EOS_Y = 14
    PAD_TOKEN = -100

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
        target_tokens.append(EOS_Y)  # Mask EOS_Y
        
        y_len = len(input_tokens) - y_start_idx  # Length including special tokens
        
        lengths.append(x_len)
        lengths.append(y_len)

    # Create shifted targets
    target_tokens = target_tokens[1:] + [PAD_TOKEN]
    
    # Convert to numpy arrays
    if return_lengths:
        return np.array(input_tokens), np.array(target_tokens), lengths
    else:
        return np.array(input_tokens), np.array(target_tokens)

def parse_generated_y(input_tokens):
    """Parses the last Y (generated) tokenized sequence and then coverts it to a 2D grid.    
    Args:
        input_tokens: List/array of tokens containing full sequence
    
    Returns:
        2D numpy array of the last generated Y grid
    """
    # Special token IDs
    BOS_Y = 13
    EOS_Y = 14
    LINE_BREAK = 12

    # Find last BOS_Y and its matching EOS_Y
    last_bos = max(i for i, t in enumerate(input_tokens) if t == BOS_Y)

    # Extract Y sequence between BOS_Y
    y_tokens = input_tokens[last_bos+1:-1]

    # Split into rows using line breaks
    rows = []
    current_row = []
    for t in y_tokens:
        if t == LINE_BREAK:
            if current_row:  # Skip empty rows from trailing line breaks
                rows.append(current_row)
                current_row = []
        else:
            current_row.append(t)
    assert len(set([len(row) for row in rows])) == 1, "not valid 2d grid"
    return rows

# x,y = tokenize_task([([[2,3],[4,5]],[[6,7]]),([[1,2],[3,4]],[[5,6,0],[7,8,10]])])
# parse_generated_y(x) gives the same result as last output y

def numpy2torch(x,max_length):
    x = torch.tensor(x[:max_length][None]).to('cuda')
    return x

def data_gen(data, IsTrain, max_length, return_lengths=False):
    """Generate data for training or testing.
    
    Args:
        data: Dictionary containing 'train' and 'test' datasets
        IsTrain: Boolean indicating whether to use training data
        max_length: Maximum sequence length for truncation
        return_lengths: Whether to return sequence lengths
        
    Yields:
        Tokenized and processed examples as PyTorch tensors
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
            x, y, lengths = tokenize_task(task, return_lengths=True)
            yield numpy2torch(x, max_length), numpy2torch(y, max_length), lengths
        else:
            x, y = tokenize_task(task)
            yield numpy2torch(x, max_length), numpy2torch(y, max_length)


def create_arc_attention_mask(*lengths):
    """
    Creates a custom attention mask for ARC examples.

    Args:
        *lengths: Sequence of integers for segment lengths
                  (x1_len, y1_len, ..., xk_len, yk_len).

    Returns:
        np.ndarray: Boolean attention mask (True = Attend).

    Rules applied:
    1. Xi attends fully to Xi (within-grid attention).
    2. yi attends fully to all preceding history (X1, y1, ..., Xi-1, yi-1)
       AND its corresponding input Xi.
    3. yi attends ONLY to itself within the yi block (diagonal is True),
       NOT to other tokens within the same yi block (off-diagonal is False).
       This allows positional information to be processed via self-attention
       without leaking information between output tokens during generation.
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
            mask[start:end, start:end] = True
            # Optional: Allow X to attend to prior history if needed
            # mask[start:end, 0:start] = True

        elif segment_type == 'Y':
            # Rule 2: yi attends to all history up to and including Xi
            mask[start:end, 0:start] = True

            # Rule 3: yi attends ONLY to itself within its own block
            # We ensure the yi-yi block (mask[start:end, start:end])
            # only has True values on the diagonal.
            # Since the block is currently False (from initialization or
            # if Rule 2 didn't overlap), we just need to set the diagonal.
            indices = np.arange(start, end)
            mask[indices, indices] = True # Allow self-attention
    out = np.where(mask, 0, -np.inf)
    return torch.tensor(out[None][None],dtype=torch.bfloat16).to('cuda')

def create_arc_causal_attention_mask(*lengths):
    """
    Creates a custom attention mask for ARC examples with causal attention for Y segments.

    Args:
        *lengths: Sequence of integers for segment lengths
                  (x1_len, y1_len, ..., xk_len, yk_len).

    Returns:
        torch.Tensor: Attention mask with 0 for positions to attend to and -inf for masked positions.

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
    for i, segment in enumerate(segments):
        start, end = segment['start'], segment['end']
        segment_type = segment['type']

        if segment_type == 'X':
            # Rule 1: Xi attends fully to itself (intra-X attention)
            mask[start:end, start:end] = True
            
        elif segment_type == 'Y':
            # Rule 2: yi attends to all history up to and including Xi
            mask[start:end, 0:start] = True

            # Rule 3: yi uses causal attention within its own block
            # Each token can attend to itself and previous tokens
            for i in range(start, end):
                mask[i, start:i+1] = True  # Causal attention pattern
    
    out = np.where(mask, 0, -np.inf)
    return torch.tensor(out[None][None], dtype=torch.bfloat16).to('cuda')

