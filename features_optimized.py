import numpy as np
from typing import List, Tuple
import copy

def extract_features(grid: List[List[int]], k: int = 3) -> np.ndarray:
    """
    Extract features from a grid of integers (0-9).
    Optimized version that precomputes color counts.
    
    Args:
        grid: List of lists of integers between 0 and 9
        k: Size of the box for local features (default: 3)
    
    Returns:
        A numpy array of shape (n_pixels, n_features) containing the features for each pixel
    """
    # Convert grid to numpy array
    grid_array = np.array(grid, dtype=int)
    height, width = grid_array.shape
    
    # Precompute color counts for the entire grid
    grid_color_counts = np.zeros(10)
    for color in range(10):
        grid_color_counts[color] = np.sum(grid_array == color)
    grid_total_pixels = height * width
    
    # Create a matrix to store all features for all pixels
    features = []
    
    # Process each pixel
    for i in range(height):
        for j in range(width):
            pixel_features = []
            
            # Get the center pixel color
            center_color = grid_array[i, j]
            
            # Define the k x k box around the center pixel
            box_start_i = max(0, i - k // 2)
            box_end_i = min(height, i + k // 2 + 1)
            box_start_j = max(0, j - k // 2)
            box_end_j = min(width, j + k // 2 + 1)
            
            box = grid_array[box_start_i:box_end_i, box_start_j:box_end_j]
            box_size = box.shape[0] * box.shape[1]
            
            # 1. Count of each color in the k x k box (normalized)
            for color in range(10):
                count = np.sum(box == color)
                pixel_features.append(count / box_size)
            
            # 2. Count of color in the box that is the same as the center pixel (normalized)
            same_color_count = np.sum(box == center_color)
            pixel_features.append(same_color_count / box_size)
            
            # 3. Row and Column Indices normalized
            pixel_features.append(i / 30)
            pixel_features.append(j / 30)
            
            # 4. One-hot encoding of the center pixel color
            for color in range(10):
                pixel_features.append(1.0 if center_color == color else 0.0)
            
            # 5. Count of connected cells (4-ways and 8-ways) in k x k box
            connected_4way = count_connected_cells(grid_array, i, j, k, diagonal=False)
            connected_8way = count_connected_cells(grid_array, i, j, k, diagonal=True)
            pixel_features.append(connected_4way / box_size)
            pixel_features.append(connected_8way / box_size)
            
            # 6. Count of connected cells (4-ways and 8-ways) in the whole grid
            connected_4way_grid = count_connected_cells(grid_array, i, j, max(height, width), diagonal=False)
            connected_8way_grid = count_connected_cells(grid_array, i, j, max(height, width), diagonal=True)
            pixel_features.append(connected_4way_grid / grid_total_pixels)
            pixel_features.append(connected_8way_grid / grid_total_pixels)
            
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
            
            # 10. Symmetry features for k x k box
            h_sym = is_horizontally_symmetric(box)
            v_sym = is_vertically_symmetric(box)
            r_sym = is_rotationally_symmetric(box)
            pixel_features.append(float(h_sym))
            pixel_features.append(float(v_sym))
            pixel_features.append(float(r_sym))
            
            # 11. Height and width of input grid (normalized)
            pixel_features.append(height / 30)
            pixel_features.append(width / 30)
            
            # 12. Edge detection in k x k box
            h_edge = has_horizontal_edge(box)
            v_edge = has_vertical_edge(box)
            pixel_features.append(float(h_edge))
            pixel_features.append(float(v_edge))
            
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
            
            # 16. Is there a bounding square box of same color at different k distances
            for dist in range(1, min(4, min(height, width) // 2)):
                has_bounding = has_bounding_box(grid_array, i, j, dist)
                pixel_features.append(float(has_bounding))
            
            features.append(pixel_features)
    
    return np.array(features)

def extract_causal_features(grid: List[List[int]], k: int = 3) -> np.ndarray:
    """
    Extract features from a grid with causality constraint.
    Cell i,j can only depend on current row i with smaller or equal j and all previous rows.
    Optimized version that precomputes color counts.
    
    Args:
        grid: List of lists of integers between 0 and 9
        k: Size of the box for local features (default: 3)
    
    Returns:
        A numpy array of shape (n_pixels, n_features) containing the features for each pixel
    """
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
            
            # Define the k x k box around the center pixel (causal part only)
            box_start_i = max(0, i - k // 2)
            box_end_i = min(height, i + k // 2 + 1)
            box_start_j = max(0, j - k // 2)
            box_end_j = min(width, j + k // 2 + 1)
            
            # Get box and apply causal mask
            box = grid_array[box_start_i:box_end_i, box_start_j:box_end_j]
            box_mask = causal_mask[box_start_i:box_end_i, box_start_j:box_end_j]
            box_size = np.sum(box_mask)  # Count of valid pixels in the box
            
            # Skip features that cannot be implemented causally
            
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
                
            # 3. Row and Column Indices normalized
            pixel_features.append(i / 30)
            pixel_features.append(j / 30)
            
            # 4. One-hot encoding of the center pixel color
            for color in range(10):
                pixel_features.append(1.0 if center_color == color else 0.0)
            
            # 5. Count of connected cells (4-ways and 8-ways) in causally visible k x k box
            connected_4way = count_connected_cells_causal(grid_array, i, j, k, causal_mask, diagonal=False)
            connected_8way = count_connected_cells_causal(grid_array, i, j, k, causal_mask, diagonal=True)
            if box_size > 0:
                pixel_features.append(connected_4way / box_size)
                pixel_features.append(connected_8way / box_size)
            else:
                pixel_features.append(0.0)
                pixel_features.append(0.0)
            
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
            
            # Skip symmetry features (10) as they don't have causal counterparts
            
            # 11. skip height and width features as they are not causal
            # pixel_features.append(height / 30)
            # pixel_features.append(width / 30)
            
            # Skip edge detection (12) as it doesn't have a causal counterpart
            
            # Skip diagonal features (13) as they don't have clear causal counterparts
            
            # 14. Is the cell on the grid border
            is_border = (i == 0 or j == 0)  # Only top and left borders are causal
            pixel_features.append(float(is_border))
            
            # 15. Is the cell in a corner of the grid
            is_corner = (i == 0 and j == 0)  # Only top-left corner is causal
            pixel_features.append(float(is_corner))
            
            # Skip bounding box feature (16) as it doesn't have a causal counterpart
            
            features.append(pixel_features)
    
    return np.array(features)

from scipy.ndimage import label

def count_connected_cells(grid, i, j, k=3, diagonal=False):
    h, w = grid.shape
    c = grid[i, j]
    half_k = k // 2

    i0 = max(0, i - half_k)
    i1 = min(h, i + half_k + 1)
    j0 = max(0, j - half_k)
    j1 = min(w, j + half_k + 1)

    box = grid[i0:i1, j0:j1]
    color_mask = (box == c)

    structure = np.ones((3,3), dtype=bool) if diagonal else np.array([[0,1,0],
                                                                      [1,1,1],
                                                                      [0,1,0]], dtype=bool)
    labeled_box, _ = label(color_mask, structure=structure)
    ci, cj = i - i0, j - j0
    lblval = labeled_box[ci, cj]
    if lblval == 0:
        return 0
    return np.sum(labeled_box == lblval)

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

def has_horizontal_edge(box: np.ndarray) -> bool:
    return np.any(box[:-1, :] != box[1:, :])

def has_vertical_edge(box: np.ndarray) -> bool:
    return np.any(box[:, :-1] != box[:, 1:])

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
    """Check if there's a bounding box of the same color at given distance."""
    height, width = grid.shape
    color = grid[center_i, center_j]
    
    # Define the coordinates of the bounding box
    top = max(0, center_i - distance)
    bottom = min(height - 1, center_i + distance)
    left = max(0, center_j - distance)
    right = min(width - 1, center_j + distance)
    
    # Check if the bounding box exists within the grid
    if top >= bottom or left >= right:
        return False
    
    # Check top and bottom edges
    for j in range(left, right + 1):
        if grid[top, j] != color or grid[bottom, j] != color:
            return False
    
    # Check left and right edges
    for i in range(top + 1, bottom):
        if grid[i, left] != color or grid[i, right] != color:
            return False
    
    return True

def test_causality_constraint(grid, k=3):
    """
    Tests that modifying grid[i,j] only causes differences in features
    with row-column order >= (i,j) (flatten index >= i*width + j).
    """
    grid = np.array(grid)
    h, w = grid.shape
    orig_features = extract_causal_features(grid.tolist(), k=k)
    
    for i in range(h):
        for j in range(w):
            modified_grid = grid.copy()
            # Flip the value in a deterministic way (must be different from original)
            modified_grid[i, j] = (grid[i, j] + 1) % 10

            new_features = extract_causal_features(modified_grid.tolist(), k=k)

            # Compare
            diff_locs = np.where(np.any(orig_features != new_features, axis=1))[0]
            print(diff_locs)