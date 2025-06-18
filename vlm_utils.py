# Deprecated VLM processing utilities from commit 6c8b249
import numpy as np
import torch
import random

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
