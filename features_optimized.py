import numpy as np
from typing import List, Tuple, Optional
from scipy.ndimage import label

def extract_features(grid: List[List[int]], background_color, max_k: int = 5) -> np.ndarray:
    """
    Extract features from a grid of integers (0-9).
    For k-dependent features, iterates over k = 3, 5, 7, ... max_k and concatenates results.
    For k-independent features, computes them once.
    
    Args:
        grid: List of lists of integers between 0 and 9
        background_color: Integer representing the background color (0-9)
        max_k: Maximum size of the box for local features (default: 5)
    
    Returns:
        A numpy array of shape (n_pixels, n_features) containing the features for each pixel
    """
    # Convert grid to numpy array
    grid_array = np.array(grid, dtype=int)
    height, width = grid_array.shape
    
    # Precompute binary mask for each color for the entire grid
    color_masks = [grid_array == color for color in range(10)]

    # Precompute color counts for the entire grid using precomputed masks
    grid_color_counts = np.array([np.sum(mask) for mask in color_masks], dtype=float)
    most_common_color = np.argmax(grid_color_counts)
    grid_total_pixels = height * width

    # Precompute connected components for the entire grid for each color
    # Store both 4-way and 8-way connectivity
    connected_labels_4way = {}  # Dictionary to store labeled arrays for each color (4-way)
    connected_labels_8way = {}  # Dictionary to store labeled arrays for each color (8-way)
    
    structure_4way = np.array([[0,1,0],
                               [1,1,1],
                               [0,1,0]], dtype=bool)
    structure_8way = np.ones((3,3), dtype=bool)
    
    for color in range(10):
        color_mask = (grid_array == color)
        connected_labels_4way[color], _ = label(color_mask, structure=structure_4way)
        connected_labels_8way[color], _ = label(color_mask, structure=structure_8way)
    
    # NEW: Precompute non-background connected components for the entire grid
    # Create a mask for non-background pixels
    non_background_mask = (grid_array != background_color)
    # Get connected components for non-background pixels (both 4-way and 8-way connectivity)
    non_bg_labels_4way, num_non_bg_labels_4way = label(non_background_mask, structure=structure_4way)
    non_bg_labels_8way, num_non_bg_labels_8way = label(non_background_mask, structure=structure_8way)
    
    # NEW: Precompute color distribution for each non-background component (both 4-way and 8-way)
    non_bg_component_color_dist_4way = {}  # Will store color distributions for each 4-way component
    non_bg_component_color_dist_8way = {}  # Will store color distributions for each 8-way component
    for component_id in range(1, num_non_bg_labels_4way + 1):
        # Create mask for this component
        component_mask = (non_bg_labels_4way == component_id)
        # Get color counts for this component
        color_counts = np.zeros(10)
        for color in range(10):
            color_counts[color] = np.sum((grid_array == color) & component_mask)
        # Normalize color counts
        normalized_color_counts = color_counts / grid_total_pixels
        # Store in dictionary
        non_bg_component_color_dist_4way[component_id] = normalized_color_counts
    
    for component_id in range(1, num_non_bg_labels_8way + 1):
        # Create mask for this component
        component_mask = (non_bg_labels_8way == component_id)
        # Get color counts for this component
        color_counts = np.zeros(10)
        for color in range(10):
            color_counts[color] = np.sum((grid_array == color) & component_mask)
        # Normalize color counts
        normalized_color_counts = color_counts / grid_total_pixels
        # Store in dictionary
        non_bg_component_color_dist_8way[component_id] = normalized_color_counts
            
    # Create a matrix to store all features for all pixels
    features = []
    
    # Process each pixel
    for i in range(height):
        for j in range(width):
            pixel_features = []
            # Get the center pixel color
            center_color = grid_array[i, j]
            pixel_features.append(float(most_common_color==center_color))
            pixel_features.append(float(background_color==center_color))
            offset = center_color - 10
            # K-dependent features: iterate over k values
            k_values = range(3, max_k + 1, 2)  # 3, 5, 7, ..., max_k
            for k in k_values:
                # Define the k x k box around the center pixel
                box_start_i = max(0, i - k // 2)
                box_end_i = min(height, i + k // 2 + 1)
                box_start_j = max(0, j - k // 2)
                box_end_j = min(width, j + k // 2 + 1)
                
                box = grid_array[box_start_i:box_end_i, box_start_j:box_end_j]
                box_size = box.shape[0] * box.shape[1]
                
                # 1. Count of each color in the k x k box (normalized)
                for color in range(10):
                    count = np.sum(color_masks[color][box_start_i:box_end_i, box_start_j:box_end_j])
                    pixel_features.append(count / box_size)
                
                # 2. Count of color in the box that is the same as the center pixel (normalized)
                pixel_features.append(pixel_features[offset])
                
                # 5. Count of connected cells (4-ways and 8-ways) in k x k box
                # Extract the relevant portion of the precomputed connected components
                labels_4way = connected_labels_4way[center_color][box_start_i:box_end_i, box_start_j:box_end_j]
                labels_8way = connected_labels_8way[center_color][box_start_i:box_end_i, box_start_j:box_end_j]
                
                # Find the label at the center position
                center_label_4way = connected_labels_4way[center_color][i, j]
                center_label_8way = connected_labels_8way[center_color][i, j]
                
                # Count cells with the same label if the center has a non-zero label
                connected_4way = np.sum(labels_4way == center_label_4way) if center_label_4way > 0 else 0
                connected_8way = np.sum(labels_8way == center_label_8way) if center_label_8way > 0 else 0
                
                pixel_features.append(connected_4way / box_size)
                pixel_features.append(connected_8way / box_size)
                
                # NEW: Check for non-background connected components in k x k box
                # Extract the relevant portion of non-background labels for the box
                box_non_bg_labels_4way = non_bg_labels_4way[box_start_i:box_end_i, box_start_j:box_end_j]
                box_non_bg_labels_8way = non_bg_labels_8way[box_start_i:box_end_i, box_start_j:box_end_j]
                
                # Find the component ID at the center pixel
                center_non_bg_label_4way = non_bg_labels_4way[i, j]
                center_non_bg_label_8way = non_bg_labels_8way[i, j]
                
                # Initialize color distribution for center component in the box (4-way)
                if center_non_bg_label_4way > 0 and center_color != background_color:
                    # Get the box section for this component
                    box_component_mask = (box_non_bg_labels_4way == center_non_bg_label_4way)
                    # Compute color counts for this component in the box
                    box_component_color_counts = np.zeros(10)
                    for color in range(10):
                        mask = color_masks[color][box_start_i:box_end_i, box_start_j:box_end_j] & box_component_mask
                        box_component_color_counts[color] = np.sum(mask)
                    # Normalize
                    box_color_dist_4way = box_component_color_counts / grid_total_pixels
                else:
                    # No non-background component at center
                    box_color_dist_4way = np.zeros(10)
                
                # Initialize color distribution for center component in the box (8-way)
                if center_non_bg_label_8way > 0 and center_color != background_color:
                    # Get the box section for this component
                    box_component_mask = (box_non_bg_labels_8way == center_non_bg_label_8way)
                    # Compute color counts for this component in the box
                    box_component_color_counts = np.zeros(10)
                    for color in range(10):
                        mask = color_masks[color][box_start_i:box_end_i, box_start_j:box_end_j] & box_component_mask
                        box_component_color_counts[color] = np.sum(mask)
                    # Normalize
                    box_color_dist_8way = box_component_color_counts / grid_total_pixels
                else:
                    # No non-background component at center
                    box_color_dist_8way = np.zeros(10)
                
                # Add color distribution for non-background component in k x k box (both 4-way and 8-way)
                pixel_features.extend(box_color_dist_4way)
                pixel_features.extend(box_color_dist_8way)
                
                # 10. Symmetry features for k x k box
                h_sym = is_horizontally_symmetric(box)
                v_sym = is_vertically_symmetric(box)
                r_sym = is_rotationally_symmetric(box)
                pixel_features.append(float(h_sym))
                pixel_features.append(float(v_sym))
                pixel_features.append(float(r_sym))
                
                # NEW: Non-background symmetry features for k x k box
                # Create binary matrix where 1=non-background, 0=background
                binary_box = (box != background_color).astype(int)
                h_sym_non_bg = is_horizontally_symmetric(binary_box)
                v_sym_non_bg = is_vertically_symmetric(binary_box)
                r_sym_non_bg = is_rotationally_symmetric(binary_box)
                pixel_features.append(float(h_sym_non_bg))
                pixel_features.append(float(v_sym_non_bg))
                pixel_features.append(float(r_sym_non_bg))
                
                # 12. Edge detection in k x k box
                h_edge = has_horizontal_edge(box)
                v_edge = has_vertical_edge(box)
                pixel_features.extend(h_edge)
                pixel_features.extend(v_edge)
                
                # 16. Is there a bounding square box of same color at distance k
                has_bounding = has_bounding_box(grid_array, i, j, k // 2)
                pixel_features.append(float(has_bounding))
            
            # K-independent features (computed once)
            
            # 3. Row and Column Indices normalized
            pixel_features.append(i / 30)
            pixel_features.append(j / 30)
            
            # 4. One-hot encoding of the center pixel color
            for color in range(10):
                pixel_features.append(1.0 if center_color == color else 0.0)
            
            # 6. Count of connected cells (4-ways and 8-ways) in the whole grid
            # Use the precomputed connected components for the entire grid
            grid_label_4way = connected_labels_4way[center_color][i, j]
            grid_label_8way = connected_labels_8way[center_color][i, j]
            
            connected_4way_grid = np.sum(connected_labels_4way[center_color] == grid_label_4way) if grid_label_4way > 0 else 0
            connected_8way_grid = np.sum(connected_labels_8way[center_color] == grid_label_8way) if grid_label_8way > 0 else 0
            
            pixel_features.append(connected_4way_grid / grid_total_pixels)
            pixel_features.append(connected_8way_grid / grid_total_pixels)
            
            # NEW: Add color distribution for non-background component in the entire grid
            center_non_bg_component_4way = non_bg_labels_4way[i, j]
            center_non_bg_component_8way = non_bg_labels_8way[i, j]
            
            # For 4-way connectivity
            if center_non_bg_component_4way > 0 and center_color != background_color:
                # Use precomputed color distribution for this component
                grid_color_dist_4way = non_bg_component_color_dist_4way[center_non_bg_component_4way]
            else:
                # No non-background component at center or center is background color
                grid_color_dist_4way = np.zeros(10)
            
            # For 8-way connectivity
            if center_non_bg_component_8way > 0 and center_color != background_color:
                # Use precomputed color distribution for this component
                grid_color_dist_8way = non_bg_component_color_dist_8way[center_non_bg_component_8way]
            else:
                # No non-background component at center or center is background color
                grid_color_dist_8way = np.zeros(10)
            
            # Add color distribution for non-background component in the entire grid (both 4-way and 8-way)
            pixel_features.extend(grid_color_dist_4way)
            pixel_features.extend(grid_color_dist_8way)
            
            # 7. Count of color in row and column of center pixel
            row_counts = np.zeros(10)
            col_counts = np.zeros(10)
            for color in range(10):
                row_counts[color] = np.sum(grid_array[i, :] == color) / width
                col_counts[color] = np.sum(grid_array[:, j] == color) / height
            pixel_features.extend(row_counts)
            pixel_features.extend(col_counts)
            
            # 8. Count of color in row/column connected to center pixel
            row_connected = count_connected_in_line(grid_array[i, :], j, center_color) / width
            col_connected = count_connected_in_line(grid_array[:, j], i, center_color) / height
            pixel_features.append(row_connected)
            pixel_features.append(col_connected)
            
            # 9. Count of each color in the entire grid (normalized) - using precomputed counts
            for color in range(10):
                pixel_features.append(grid_color_counts[color] / grid_total_pixels)
            
            # 11. Height and width of input grid (normalized)
            pixel_features.append(height / 30)
            pixel_features.append(width / 30)
            
            # 13. Count of contiguous same color in diagonals
            diag_count = count_diagonal(grid_array, i, j, center_color) / 30
            anti_diag_count = count_anti_diagonal(grid_array, i, j, center_color) / 30
            pixel_features.append(diag_count)
            pixel_features.append(anti_diag_count)
            
            # 14. Is the cell on the grid border
            is_border = (i == 0 or i == height - 1 or j == 0 or j == width - 1)
            pixel_features.append(float(is_border))
            
            # 15. Is the cell in a corner of the grid
            is_corner = ((i == 0 and j == 0) or 
                        (i == 0 and j == width - 1) or 
                        (i == height - 1 and j == 0) or 
                        (i == height - 1 and j == width - 1))
            pixel_features.append(float(is_corner))
            
            features.append(pixel_features)
    
    return np.array(features)

def extract_causal_features(grid: List[List[int]], max_k: int = 5, background_color: int = 0) -> np.ndarray:
    """
    Extract features from a grid with causality constraint.
    Cell i,j can only depend on current row i with smaller or equal j and all previous rows.
    For k-dependent features, iterates over k = 3, 5, 7, ... max_k and concatenates results.
    For k-independent features, computes them once.
    
    Args:
        grid: List of lists of integers between 0 and 9
        max_k: Maximum size of the box for local features (default: 3)
    
    Returns:
        A numpy array of shape (n_pixels, n_features) containing the features for each pixel
    """
    # TODO: extract_causal_features needs to be updated as extract_features
    # Convert grid to numpy array
    grid_array = np.array(grid, dtype=int)
    height, width = grid_array.shape
    
    # Create a matrix to store all features for all pixels
    features = []
    
    # Process each pixel
    for i in range(height):
        for j in range(width):
            pixel_features = []
            
            # Get the center pixel color
            center_color = grid_array[i, j]
            
            # Create a causal mask for each position
            causal_mask = np.zeros_like(grid_array, dtype=bool)
            causal_mask[:i, :] = True  # All previous rows
            causal_mask[i, :j+1] = True  # Current row up to and including j
            
            # Precompute color counts for the causal part of the grid
            causal_grid_color_counts = np.zeros(10)
            for color in range(10):
                causal_grid_color_counts[color] = np.sum((grid_array == color) & causal_mask)
            causal_grid_total_pixels = np.sum(causal_mask)
            
            # K-dependent features: iterate over k values
            k_values = range(3, max_k + 1, 2)  # 3, 5, 7, ..., max_k
            for k in k_values:
                # Define the k x k box around the center pixel (causal part only)
                box_start_i = max(0, i - k // 2)
                box_end_i = min(height, i + k // 2 + 1)
                box_start_j = max(0, j - k // 2)
                box_end_j = min(width, j + k // 2 + 1)
                
                # Get box and apply causal mask
                box = grid_array[box_start_i:box_end_i, box_start_j:box_end_j]
                box_mask = causal_mask[box_start_i:box_end_i, box_start_j:box_end_j]
                box_size = np.sum(box_mask)  # Count of valid pixels in the box
                
                # 1. Count of each color in the causally visible part of k x k box
                for color in range(10):
                    if box_size > 0:
                        count = np.sum((box == color) & box_mask)
                        pixel_features.append(count / box_size)
                    else:
                        pixel_features.append(0.0)
                
                # 2. Count of color in the box that is the same as the center pixel (normalized)
                if box_size > 0:
                    same_color_count = np.sum((box == center_color) & box_mask)
                    pixel_features.append(same_color_count / box_size)
                else:
                    pixel_features.append(0.0)
                
                # 5. Count of connected cells (4-ways and 8-ways) in causally visible k x k box
                connected_4way = count_connected_cells_causal(grid_array, i, j, k, causal_mask, diagonal=False)
                connected_8way = count_connected_cells_causal(grid_array, i, j, k, causal_mask, diagonal=True)
                if box_size > 0:
                    pixel_features.append(connected_4way / box_size)
                    pixel_features.append(connected_8way / box_size)
                else:
                    pixel_features.append(0.0)
                    pixel_features.append(0.0)
                
                # NEW: Symmetry features for causally visible k x k box
                # We need to handle masked areas carefully when checking symmetry
                # Use the part that's visible in causal mask for symmetry analysis
                masked_box = np.where(box_mask, box, -1)  # Use -1 for masked areas
                
                # Standard symmetry features
                h_sym = is_horizontally_symmetric(masked_box)
                v_sym = is_vertically_symmetric(masked_box)
                r_sym = is_rotationally_symmetric(masked_box)
                pixel_features.append(float(h_sym))
                pixel_features.append(float(v_sym))
                pixel_features.append(float(r_sym))
                
                # Non-background symmetry features
                binary_box = np.where(box_mask, (box != background_color).astype(int), -1)
                h_sym_non_bg = is_horizontally_symmetric(binary_box)
                v_sym_non_bg = is_vertically_symmetric(binary_box)
                r_sym_non_bg = is_rotationally_symmetric(binary_box)
                pixel_features.append(float(h_sym_non_bg))
                pixel_features.append(float(v_sym_non_bg))
                pixel_features.append(float(r_sym_non_bg))
            
            # K-independent features (computed once)
            
            # 3. Row and Column Indices normalized
            pixel_features.append(i / 30)
            pixel_features.append(j / 30)
            
            # 4. One-hot encoding of the center pixel color
            for color in range(10):
                pixel_features.append(1.0 if center_color == color else 0.0)
            
            # 6. Count of connected cells in the causal part of the whole grid
            connected_4way_grid = count_connected_cells_causal(grid_array, i, j, max(height, width), causal_mask, diagonal=False)
            connected_8way_grid = count_connected_cells_causal(grid_array, i, j, max(height, width), causal_mask, diagonal=True)
            if causal_grid_total_pixels > 0:
                pixel_features.append(connected_4way_grid / causal_grid_total_pixels)
                pixel_features.append(connected_8way_grid / causal_grid_total_pixels)
            else:
                pixel_features.append(0.0)
                pixel_features.append(0.0)
            
            # 7. Count of color in causally visible part of row and column
            row_counts = np.zeros(10)
            col_counts = np.zeros(10)
            for color in range(10):
                row_visible = np.sum(causal_mask[i, :])
                col_visible = np.sum(causal_mask[:, j])
                
                if row_visible > 0:
                    row_counts[color] = np.sum((grid_array[i, :] == color) & causal_mask[i, :]) / row_visible
                if col_visible > 0:
                    col_counts[color] = np.sum((grid_array[:, j] == color) & causal_mask[:, j]) / col_visible
            
            pixel_features.extend(row_counts)
            pixel_features.extend(col_counts)
            
            # 8. Count of color in row/column connected to center pixel (causal)
            row_connected = count_connected_in_line_causal(grid_array[i, :], j, center_color, causal_mask[i, :])
            col_connected = count_connected_in_line_causal(grid_array[:, j], i, center_color, causal_mask[:, j])
            
            row_visible = np.sum(causal_mask[i, :])
            col_visible = np.sum(causal_mask[:, j])
            
            if row_visible > 0:
                pixel_features.append(row_connected / row_visible)
                pixel_features.append(col_connected / col_visible)
            else:
                pixel_features.append(0.0)
                pixel_features.append(0.0)
            
            # 9. Count of each color in the causally visible part of the grid
            for color in range(10):
                if causal_grid_total_pixels > 0:
                    pixel_features.append(causal_grid_color_counts[color] / causal_grid_total_pixels)
                else:
                    pixel_features.append(0.0)
            
            # 14. Is the cell on the grid border
            is_border = (i == 0 or j == 0)  # Only top and left borders are causal
            pixel_features.append(float(is_border))
            
            # 15. Is the cell in a corner of the grid
            is_corner = (i == 0 and j == 0)  # Only top-left corner is causal
            pixel_features.append(float(is_corner))
            
            features.append(pixel_features)
    
    return np.array(features)

def count_connected_cells_causal(grid, center_i, center_j, k, causal_mask, diagonal=False):
    """Count connected cells of the same color in a k x k box, within the causal mask, using scipy."""
    height, width = grid.shape
    color = grid[center_i, center_j]
    
    # Define the box
    box_start_i = max(0, center_i - k // 2)
    box_end_i   = min(height, center_i + k // 2 + 1)
    box_start_j = max(0, center_j - k // 2)
    box_end_j   = min(width, center_j + k // 2 + 1)
    
    # Box and corresponding mask
    box      = grid[box_start_i:box_end_i, box_start_j:box_end_j]
    mask_box = causal_mask[box_start_i:box_end_i, box_start_j:box_end_j]
    color_mask = (box == color)
    valid_mask = color_mask & mask_box
    
    # Label using scipy
    structure = np.ones((3,3), dtype=bool) if diagonal else np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=bool)
    labeled, _ = label(valid_mask, structure=structure)
    
    # Center position in box
    ci, cj = center_i - box_start_i, center_j - box_start_j
    lblval = labeled[ci, cj]
    if lblval == 0:
        return 0
    return np.sum(labeled == lblval)

def count_connected_in_line(line, center_idx, color):
    """Count connected cells of the same color in a line."""
    count = 0
    # Start from center and go both ways
    idx = center_idx
    while idx < len(line) and line[idx] == color:
        count += 1
        idx += 1
    
    idx = center_idx - 1
    while idx >= 0 and line[idx] == color:
        count += 1
        idx -= 1
    
    return count

def count_connected_in_line_causal(line, center_idx, color, causal_mask):
    """Count connected cells of the same color in a line respecting causality."""
    count = 0
    # Start from center and go left (causal direction)
    idx = center_idx
    while idx >= 0 and line[idx] == color and causal_mask[idx]:
        count += 1
        idx -= 1
    
    # For current position, go right only if causal
    idx = center_idx + 1
    while idx < len(line) and line[idx] == color and causal_mask[idx]:
        count += 1
        idx += 1
    
    return count

def is_horizontally_symmetric(box: np.ndarray) -> bool:
    return np.array_equal(box, np.fliplr(box))

def is_vertically_symmetric(box: np.ndarray) -> bool:
    return np.array_equal(box, np.flipud(box))

def is_rotationally_symmetric(box: np.ndarray) -> bool: # 180 degrees
    return np.array_equal(box, np.rot90(box, 2))

def _is_uniform_and_get_color(arr: np.ndarray) -> Tuple[bool, Optional[int]]:
    """
    Checks if a 1D array is uniform in color and returns the color if so.
    An empty array is not considered uniform for this purpose.
    """
    if arr.size == 0:
        return False, None 
    unique_colors = np.unique(arr)
    if len(unique_colors) == 1:
        return True, unique_colors[0]
    return False, None

def has_vertical_edge(box: np.ndarray) -> Tuple[float, float]:
    """
    Checks for a vertical edge in a box as per Feature 12.
    - Center column is uniformly one color.
    - Left column is uniformly another color (different from center),
    - Right column is uniformly another color (different from center).
    """
    _box_rows, box_cols = box.shape
    
    # Needs at least 3 columns for a center and at least one side column.
    if box_cols < 3: 
        return 0.0, 0.0

    center_c_idx = box_cols // 2 

    is_center_uniform, color_center = _is_uniform_and_get_color(box[:, center_c_idx])
    if not is_center_uniform:
        return 0.0, 0.0

    # Check with right column
    # The column index for right is center_c_idx + 1
    is_right_uniform, color_right = _is_uniform_and_get_color(box[:, center_c_idx + 1])
    if is_right_uniform and color_right != color_center:
        res0 = 1.0
    else:
        res0 = 0.0

    # Check with left column
    # The column index for left is center_c_idx - 1
    is_left_uniform, color_left = _is_uniform_and_get_color(box[:, center_c_idx - 1])
    if is_left_uniform and color_left != color_center:
        res1 = 1.0
    else:
        res1 = 0.0
            
    return res0, res1

def has_horizontal_edge(box: np.ndarray) -> Tuple[float, float]:
    """
    Checks for a horizontal edge in a box as per Feature 12.
    - Center row is uniformly one color.
    - EITHER Top row is uniformly another color (different from center)
    - OR Bottom row is uniformly another color (different from center).
    """
    box_rows, _box_cols = box.shape

    # Needs at least 3 rows for a center and at least one side row.
    if box_rows < 3:
        return 0.0, 0.0

    center_r_idx = box_rows // 2

    is_center_uniform, color_center = _is_uniform_and_get_color(box[center_r_idx, :])
    if not is_center_uniform:
        return 0.0, 0.0

    # Check with bottom row
    # The row index for bottom is center_r_idx + 1
    is_bottom_uniform, color_bottom = _is_uniform_and_get_color(box[center_r_idx + 1, :])
    if is_bottom_uniform and color_bottom != color_center:
        res0 = 1.0
    else:
        res0 = 0.0
    
    # Check with top row
    # The row index for top is center_r_idx - 1
    is_top_uniform, color_top = _is_uniform_and_get_color(box[center_r_idx - 1, :])
    if is_top_uniform and color_top != color_center:
        res1 = 1.0
    else:
        res1 = 0.0
            
    return res0, res1

def count_diagonal(grid, center_i, center_j, color):
    """Count contiguous same color in the main diagonal."""
    height, width = grid.shape
    count = 0
    
    # Start from center and go both ways
    i, j = center_i, center_j
    while 0 <= i < height and 0 <= j < width and grid[i, j] == color:
        count += 1
        i += 1
        j += 1
    
    i, j = center_i - 1, center_j - 1
    while 0 <= i < height and 0 <= j < width and grid[i, j] == color:
        count += 1
        i -= 1
        j -= 1
    
    return count

def count_anti_diagonal(grid, center_i, center_j, color):
    """Count contiguous same color in the anti-diagonal."""
    height, width = grid.shape
    count = 0
    
    # Start from center and go both ways
    i, j = center_i, center_j
    while 0 <= i < height and 0 <= j < width and grid[i, j] == color:
        count += 1
        i += 1
        j -= 1
    
    i, j = center_i - 1, center_j + 1
    while 0 <= i < height and 0 <= j < width and grid[i, j] == color:
        count += 1
        i -= 1
        j += 1
    
    return count

def has_bounding_box(grid, center_i, center_j, distance):
    nrows, ncols = grid.shape
    height, width = nrows, ncols
    top = max(0, center_i - distance)
    bottom = min(height - 1, center_i + distance)
    left = max(0, center_j - distance)
    right = min(width - 1, center_j + distance)
    
    # find the number to compare with
    number = None
    if center_i - distance >= 0:
        if center_j - distance >= 0:
            number = grid[center_i - distance, center_j - distance]
        elif center_j + distance < ncols:
            number = grid[center_i - distance, center_j + distance]
    elif center_i + distance < nrows:
        if center_j - distance >= 0:
            number = grid[center_i + distance, center_j - distance]
        elif center_j + distance < ncols:
            number = grid[center_i + distance, center_j + distance]
    if number is None: return False

    # Top Side
    row = center_i - distance
    if row >= 0:
        if grid[row, left] != number or np.unique(grid[row, left:right+1]).size > 1:
            return False

    # Bottom Side
    row = center_i + distance
    if row < nrows:
        if grid[row, left] != number or np.unique(grid[row, left:right+1]).size > 1:
            return False

    # Left Side
    col = center_j - distance
    if 0 <= col:
        if grid[top, col] != number or np.unique(grid[top+1:bottom, col]).size > 1:
            return False

    # Right Side
    col = center_j + distance
    if col < ncols:
        if grid[top, col] != number or np.unique(grid[top+1:bottom, col]).size > 1:
            return False

    return True  # At least one side within bounds and all valid were OK



def test_causality_constraint(grid, max_k: int = 5):
    """
    Tests that modifying grid[i,j] only causes differences in features
    with row-column order >= (i,j) (flatten index >= i*width + j).
    """
    grid = np.array(grid)
    h, w = grid.shape
    orig_features = extract_causal_features(grid.tolist(), max_k=max_k)
    tot = h * w
    idx = 0
    for i in range(h):
        for j in range(w):
            modified_grid = grid.copy()
            # Flip the value in a deterministic way (must be different from original)
            modified_grid[i, j] = (grid[i, j] + 1) % 10

            new_features = extract_causal_features(modified_grid.tolist(), max_k=max_k)

            # Compare
            diff_locs = np.where(np.any(orig_features != new_features, axis=1))[0]
            if np.any(diff_locs != np.arange(idx, tot)):
                raise ValueError(f"Causality constraint violated at {i}, {j}")
            idx += 1

