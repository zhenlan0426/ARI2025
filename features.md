1. input: list of list of integer between 0 and 9, if this is for input or output, output: list of features for each pixel in the grid flattened to 1d. Feature given as below.
2. Features needs to be normalized when possible, e.g. count of color in the k by k box for each color normalized by total count of pixels in box.Note that as we dont know the size of grid, we need to normalize by k by k box or max possible width/height of grid (30). our features should not depend on the size of grid! Also note that k by k box at boundary of grid will have fewer than k by k pixels when normalizing.
3. input features (non-causal):
    - count of color in the k by k box for each color. result will be a vector one for each color.
    - count of color in the box that is the same as the center pixel.
    - Row and Column Indices of the center pixel divided by 30 (max).
    - one hot encoding of the color of the center pixel.
    - count of connected cells (same color, 4-ways and 8-ways as separate features) in k by k box.
    - count of connected cells (same color, 4-ways and 8-ways as separate features) in the whole grid.
    - count of color in row (or column as separate feature) of center pixel.
    - count of color in row (or column as separate feature) of center pixel that is connected to the center pixel.
    - count of each color in the entire grid divided by 30 * 30 as a vector of 10 one for each color.
    Following features do not have causal counterpart;
    - various forms of symmetry as 0/1 features, e.g. horizontal, vertical symmetry, rotational symmetry around center pixel over k by k box.
    - height and width of input grid divided by 30.
    - vertical and horizontal edge detection (0/1) in k by k box, i.e. center column (k rows) are one color and right column (k rows) are another color, similar for left column and center row.
    - count of contiguous same color in diagonal and anti-diagonal divided by 30 (two features).
    - Is the cell on the grid border (0/1)
    - Is the cell in a corner of the grid (0/1)
    - Is there a bounding square box of same color at k distance from the center pixel (0/1) for different k
4. output features:
    - similar to input features but with causality constraint. cell i,j can only depend on current row i with smaller or equal j and all previous rows. ignore features that cannot be causal.
5. write test to ensure causality constraint is satisfied.
6. use numpy for efficient implementation.