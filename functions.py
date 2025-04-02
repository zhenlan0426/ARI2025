from dataclasses import dataclass
import numpy as np
import random
import torch

''' dataset processing '''
@dataclass
class TransformPara:
    fliplr: int
    rot90: int
    perm_color: np.ndarray
    perm_example: np.ndarray

def generateTransformPara(n):
    """Randomly generates transformation parameters"""
    # n is the number of examples
    # (fliplr, rot90, permutate color, permutate example)
    return TransformPara(np.random.randint(0, 2), np.random.randint(0, 4), np.random.permutation(10), np.random.permutation(n))

def forward(x, tpara:TransformPara):
    """Applies transformations to a single grid."""
    if tpara.fliplr:
        x = np.fliplr(x)
    x = np.rot90(x, k=tpara.rot90)
    x = tpara.perm_color[x]
    return x
    
def backward(x, tpara:TransformPara):
    """Reverses transformations for a single grid."""
    inv_perm = np.argsort(tpara.perm_color)  # Compute inverse permutation
    x = inv_perm[x]
    x = np.rot90(x, k=4-tpara.rot90)
    if tpara.fliplr:
        x = np.fliplr(x)
    return x

def forwardTask(task,tpara):
    """Applies transformation to list of [(x1,y1),...] examples."""
    task_out = []
    for i in tpara.perm_example:
        x,y = task[i]
        task_out.append((forward(x,tpara), forward(y,tpara)))
    return task_out

def backwardTask(task,tpara):
    return [(backward(x,tpara), backward(y,tpara)) for x,y in task]
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
        
        for row_idx, row in enumerate(x):
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
        
        for row_idx, row in enumerate(y):
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