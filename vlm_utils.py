# Deprecated VLM processing utilities from commit 81798af
import numpy as np
import torch
import random


def create_attention_mask(length: int) -> torch.Tensor:
    mask = torch.zeros(length, length, dtype=torch.bfloat16, device='cuda')
    upper = torch.triu(torch.ones(length, length, dtype=torch.bool, device='cuda'), diagonal=1)
    mask[upper] = torch.finfo(torch.bfloat16).min
    return mask.unsqueeze(0).unsqueeze(0)


def tokenize_VLM(task, processor, max_pairs=4, multiplier=14, decode=False):
    """Tokenize ARC tasks for a vision-language model."""
    BOS_TOKEN_IDX = 10
    INPUT_TOKEN_IDX = 11
    OUTPUT_TOKEN_IDX = 12
    NEWLINE_TOKEN_IDX = 13
    EOLINE_TOKEN_IDX = 14
    BEG_OF_IMAGE_TOKEN_IDX = 15
    IMAGE_SOFT_TOKEN_IDX = 16
    END_OF_IMAGE_TOKEN_IDX = 17
    IGNORE_INDEX = -100

    color_array = np.array([
        [255,0,0],[0,0,255],[0,255,0],[255,255,0],[255,165,0],
        [128,0,128],[255,255,255],[0,255,255],[128,128,128],[165,42,42]
    ])

    images = []
    token_type_ids = [0]
    target_ids = [IGNORE_INDEX]
    input_ids = [BOS_TOKEN_IDX]

    def scale_image(image, k):
        return [pixel for row in image for pixel in row for _ in range(k) for _ in range(k)]

    def process_grid(grid, isInput, image_token):
        grid = np.array(grid, dtype=int)
        grid = np.concatenate([grid, np.full((grid.shape[0],1), NEWLINE_TOKEN_IDX)], axis=1)
        grid = grid.flatten()
        grid[-1] = EOLINE_TOKEN_IDX
        ids = [INPUT_TOKEN_IDX if isInput else OUTPUT_TOKEN_IDX]
        ids.extend(grid.tolist())
        targets = ids[1:] + [IGNORE_INDEX]
        types = [0] * len(ids)
        if image_token is not None:
            ids.extend(image_token)
            types.append(0)
            types.extend([1]*(len(image_token)-2))
            types.append(0)
            targets.extend([IGNORE_INDEX]*len(image_token))
        return ids, targets, types

    def process_image(grid, multiplier):
        grid = scale_image(grid, multiplier)
        img = color_array[np.array(grid, dtype=int)]
        img = np.transpose(img, (2,0,1))
        return processor.image_processor.preprocess(
            img, return_tensors="pt",
            data_format="channels_first",
            input_data_format="channels_first",
            do_resize=False
        )["pixel_values"].to('cuda')

    def create_image_token(grid, multiplier):
        l, w = len(grid), len(grid[0])
        r, c = l*multiplier//14, w*multiplier//14
        return [BEG_OF_IMAGE_TOKEN_IDX] + [IMAGE_SOFT_TOKEN_IDX]*(r*c) + [END_OF_IMAGE_TOKEN_IDX]

    def process_all(grid, isInput, multiplier, isLast=False):
        if not isLast:
            image = process_image(grid, multiplier)
            token = create_image_token(grid, multiplier)
        else:
            image, token = None, None
        ids, targets, types = process_grid(grid, isInput, token)
        return image, ids, targets, types

    for input_grid, output_grid in task[:max_pairs-1]:
        image, ids, targets, types = process_all(input_grid, True, multiplier)
        images.append(image)
        input_ids.extend(ids)
        target_ids.extend(targets)
        token_type_ids.extend(types)

        image, ids, targets, types = process_all(output_grid, False, multiplier)
        images.append(image)
        input_ids.extend(ids)
        target_ids.extend(targets)
        token_type_ids.extend(types)

    idx = len(task)-1 if decode else max_pairs-1
    input_grid, output_grid = task[idx]
    image, ids, targets, types = process_all(input_grid, True, multiplier)
    images.append(image)
    input_ids.extend(ids)
    target_ids.extend(targets)
    token_type_ids.extend(types)

    if not decode:
        image, ids, targets, types = process_all(output_grid, False, multiplier, isLast=True)
        input_ids.extend(ids)
        target_ids.extend(targets)
        token_type_ids.extend(types)
    else:
        input_ids.append(OUTPUT_TOKEN_IDX)
        token_type_ids.append(0)
        target_ids = np.array(output_grid) if output_grid is not None else None

    return {
        'input_ids': torch.tensor(input_ids)[None].to('cuda'),
        'pixel_values': images,
    }, target_ids if decode else torch.tensor(target_ids)[None].to('cuda')


def tokenize_VLM_oneshot(task, processor, max_pairs=4, multiplier=14, decode=False):
    """Tokenize ARC tasks for VLM one-shot mode."""
    INPUT_TOKEN_IDX = 11
    OUTPUT_TOKEN_IDX = 12
    NEWLINE_TOKEN_IDX = 13
    EOLINE_TOKEN_IDX = 14
    BEG_OF_IMAGE_TOKEN_IDX = 15
    IMAGE_SOFT_TOKEN_IDX = 16
    END_OF_IMAGE_TOKEN_IDX = 17
    PREDICT_CELL_Y = 10
    SIZE_TOKEN_OFFSET = 17
    IGNORE_INDEX = -100

    max_pairs = min(max_pairs, len(task))
    color_array = np.array([
        [255,0,0],[0,0,255],[0,255,0],[255,255,0],[255,165,0],
        [128,0,128],[255,255,255],[0,255,255],[128,128,128],[165,42,42]
    ])

    images = []
    token_type_ids = []
    target_ids = []
    input_ids = []

    def scale_image(image, k):
        return [pixel for row in image for pixel in row for _ in range(k) for _ in range(k)]

    def process_grid(grid, isInput, image_token):
        p, q = len(grid), len(grid[0])
        p_token, q_token = SIZE_TOKEN_OFFSET + p, SIZE_TOKEN_OFFSET + q
        grid = np.array(grid, dtype=int)
        grid = np.concatenate([grid, np.full((grid.shape[0],1), NEWLINE_TOKEN_IDX)], axis=1)
        grid = grid.flatten()
        grid[-1] = EOLINE_TOKEN_IDX
        ids = [INPUT_TOKEN_IDX if isInput else OUTPUT_TOKEN_IDX, p_token, q_token]
        ids.extend(grid.tolist())
        targets = ids[1:] + [IGNORE_INDEX]
        types = [0] * len(ids)
        if image_token is not None:
            ids.extend(image_token)
            types.append(0)
            types.extend([1]*(len(image_token)-2))
            types.append(0)
            targets.extend([IGNORE_INDEX]*len(image_token))
        return ids, targets, types

    def process_image(grid, multiplier):
        grid = scale_image(grid, multiplier)
        img = color_array[np.array(grid, dtype=int)]
        img = np.transpose(img, (2,0,1))
        return processor.image_processor.preprocess(
            img, return_tensors="pt",
            data_format="channels_first",
            input_data_format="channels_first",
            do_resize=False
        )["pixel_values"].to('cuda')

    def create_image_token(grid, multiplier):
        l, w = len(grid), len(grid[0])
        r, c = l*multiplier//14, w*multiplier//14
        return [BEG_OF_IMAGE_TOKEN_IDX] + [IMAGE_SOFT_TOKEN_IDX]*(r*c) + [END_OF_IMAGE_TOKEN_IDX]

    def process_all(grid, isInput, multiplier):
        image = process_image(grid, multiplier)
        token = create_image_token(grid, multiplier)
        ids, targets, types = process_grid(grid, isInput, token)
        return image, ids, targets, types

    for input_grid, output_grid in task[:max_pairs-1]:
        image, ids, targets, types = process_all(input_grid, True, multiplier)
        images.append(image)
        input_ids.extend(ids)
        target_ids.extend(targets)
        token_type_ids.extend(types)

        image, ids, targets, types = process_all(output_grid, False, multiplier)
        images.append(image)
        input_ids.extend(ids)
        target_ids.extend(targets)
        token_type_ids.extend(types)

    idx = len(task)-1 if decode else max_pairs-1
    input_grid, output_grid = task[idx]
    image, ids, targets, types = process_all(input_grid, True, multiplier)
    images.append(image)
    input_ids.extend(ids)
    target_ids.extend(targets)
    token_type_ids.extend(types)

    input_ids.append(OUTPUT_TOKEN_IDX)
    l = len(input_ids)
    if decode:
        token_type_ids.append(0)
        target_ids = np.array(output_grid) if output_grid is not None else None
    else:
        p, q = len(output_grid), len(output_grid[0])
        p_token, q_token = SIZE_TOKEN_OFFSET + p, SIZE_TOKEN_OFFSET + q
        input_ids.extend([p_token, q_token, *([PREDICT_CELL_Y] * (p * (q + 1)))])
        target_ids.append(p_token)
        target_ids.append(q_token)
        target_ids.append(IGNORE_INDEX)
        for sub in output_grid:
            target_ids.extend(sub)
            target_ids.append(NEWLINE_TOKEN_IDX)
        token_type_ids.extend([0]*(len(target_ids) - len(token_type_ids)))

    return {
        'input_ids': torch.tensor(input_ids)[None].to('cuda'),
        'pixel_values': images,
    }, target_ids if decode else torch.tensor(target_ids)[None].to('cuda'), l


def data_gen_VLM(data, IsTrain, processor, max_pairs, decode=False, tokenize_func=tokenize_VLM, max_len=8192):
    """Generate data for VLM."""
    dataset = data['train'] if IsTrain else data['test']
    attention_mask = create_attention_mask(max_len)
    if IsTrain:
        random.shuffle(dataset)
    for task in dataset:
        if IsTrain:
            task = forwardTask(task, generateTransformPara(len(task)))
        inputs, *others = tokenize_func(task, processor, max_pairs=max_pairs, decode=decode)
        l = inputs['input_ids'].shape[1]
        inputs['attention_mask'] = attention_mask[:, :, :l, :l]
        yield inputs, *others
