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
    enlarge: tuple[int, int]  # Enlarge factors (n, m)
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
                         np.random.permutation(n),(np.random.randint(1, 3), np.random.randint(1, 3)),\
                         np.random.randint(0, 2))

def forward(x, tpara:TransformPara):
    """Applies transformations to a single grid."""
    if tpara.fliplr:
        x = np.fliplr(x)
    x = np.rot90(x, k=tpara.rot90)
    x = tpara.perm_color[x]
    n, m = tpara.enlarge
    x = enlarge_grid_n_times(x, n, m)
    return x
    
def backward(x, tpara:TransformPara):
    """Reverses transformations for a single grid."""
    n, m = tpara.enlarge
    x = shrink_grid_n_times(x, n, m)
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

def tokenize_task(task):
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
    is_y_sequence = False  # Track if we're processing y sequence

    for x, y in task:
        # Process input grid (x)
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

        # Process output grid (y)
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

    # Create shifted targets
    target_tokens = target_tokens[1:] + [PAD_TOKEN]
    
    # Convert to numpy arrays
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

def data_gen(data, IsTrain, max_length):
    if IsTrain:
        data = data['train']
        random.shuffle(data)
        for task in data:
            task = forwardTask(task,generateTransformPara(len(task)))
            x,y = tokenize_task(task)
            yield numpy2torch(x,max_length), numpy2torch(y,max_length)
    else:
        data = data['test']
        for task in data:
            x,y = tokenize_task(task)
            yield numpy2torch(x,max_length), numpy2torch(y,max_length)