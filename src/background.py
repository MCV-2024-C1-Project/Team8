import cv2
import numpy as np
from PIL import Image
from skimage import filters


def convert_bool_to_uint8(binary_array: np.ndarray) -> np.ndarray:
    """
    Convert a boolean array to an `uint8` array, scaling values by 255.

    Parameters:
    -----------
    binary_array : np.ndarray
        Input boolean array where True represents foreground and False represents background.

    Returns:
    --------
    np.ndarray
        Converted array in `uint8` format with values 0 (background) and 255 (foreground).
    """
    # Ensure the input is a boolean array
    assert binary_array.dtype == np.bool_, "Input array must be of boolean type."

    return binary_array.astype(np.uint8) * 255


def apply_otsu_threshold(image: np.ndarray) -> np.ndarray:
    """
    Apply Otsu's threshold to an image and convert the result to a binary mask
    with values 0 and 255 (in `uint8` format).

    Parameters:
    -----------
    image : np.ndarray
        Input grayscale image for thresholding, where Otsu's method will be applied.

    Returns:
    --------
    np.ndarray
        Binary mask of the image after applying Otsu's threshold, with 255 for foreground
        and 0 for background.
    """
    # Apply Otsu's threshold to create a binary mask
    binary_mask = image > filters.threshold_otsu(image)
    # Convert the boolean mask to uint8 and scale by 255
    return convert_bool_to_uint8(binary_mask)


def get_gradient_magnitude(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Calculate the gradient magnitude of an image using a specified kernel size
    for detecting horizontal and vertical edges.

    Parameters:
    -----------
    image : np.ndarray
        Input grayscale image for which gradient magnitude is calculated.
    kernel_size : int, optional
        Size of the edge-detection kernel in one axis. Must be an odd number, default is 5
        and generates a kernel of (5, kernel_size) in y-axis and (kernel_size, 5) in x-axis.

    Returns:
    --------
    np.ndarray
        Gradient magnitude of the input image.

    Raises:
    -------
    ValueError
        If `kernel_size` is not an odd integer.
    """
    # Ensure kernel size is an odd integer
    assert kernel_size % 2 != 0, f"'kernel_size' must be an odd integer, instead got {kernel_size}."

    # Define kernels for different edge orientations
    kernely = np.array([
        [1] * kernel_size,
        [1] * kernel_size,
        [0] * kernel_size,
        [-1] * kernel_size,
        [-1] * kernel_size
    ])  # Vertical
    kernelx = np.copy(kernely).T  # Horizontal

    # Calculate gradients
    gradient_x = cv2.filter2D(image, cv2.CV_64F, kernelx)
    gradient_y = cv2.filter2D(image, cv2.CV_64F, kernely)

    # Calculate gradient magnitudes
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    return gradient_magnitude


def get_painting_side_masks(image: Image) -> (np.ndarray, np.ndarray):
    """
    Generate binary masks for the left and right halves of a painting image by calculating
    gradient magnitudes and applying Otsu's threshold.

    Parameters:
    -----------
    image : PIL.Image
        Input RGB image of the painting to be processed.

    Returns:
    --------
    (np.ndarray, np.ndarray)
        Binary masks for the left and right halves of the image, where 255 represents
        foreground and 0 represents background.

    Raises:
    -------
    AssertionError
        If the input image is not in RGB format.
    """
    # Assert the image is in RGB format
    assert image.mode == "RGB", "Input image must be in RGB format."

    # Convert to grayscale and split into left and right halves
    grayscale = np.array(image.convert("L"))
    half_image = grayscale.shape[1] // 2
    left_grayscale = grayscale[:, :half_image]
    right_grayscale = grayscale[:, half_image:]

    # Calculate gradient magnitudes for both halves
    left_grayscale_gradient_magnitude = get_gradient_magnitude(left_grayscale)
    right_grayscale_gradient_magnitude = get_gradient_magnitude(right_grayscale)

    # Apply Otsu's threshold to create binary masks
    left_mask = apply_otsu_threshold(left_grayscale_gradient_magnitude)
    right_mask = apply_otsu_threshold(right_grayscale_gradient_magnitude)

    return left_mask, right_mask


def inpaint_mask(mask: np.ndarray, crop_side: str) -> np.ndarray:
    """
    Fill gaps in a binary mask by identifying edges of the painting and filling the area between them.
    Adjusts fill range based on whether the painting touches the left or right side of the mask.

    Parameters:
    -----------
    mask : np.ndarray
        Binary mask array with gaps to be filled, where 255 represents the painting and 0 represents the background.
    crop_side : str
        Specifies the side that may have been cropped ('left' or 'right'), guiding the fill to the correct boundary.

    Returns:
    --------
    np.ndarray
        Mask array with filled gaps, confined to the detected boundaries of the painting.

    Notes:
    ------
    Uses Otsu's thresholding to identify valid rows and columns of the painting, then fills between
    detected boundaries. Special cases are handled if the painting reaches the left or right side of the mask.
    """

    # Create a copy of the input mask to avoid modifying the original
    filled_mask = np.copy(mask)

    # Compute row and column non-zero pixel densities (percentage of the mask that is filled)
    row_fill_ratios = (filled_mask > 0).mean(axis=1)
    col_fill_ratios = (filled_mask > 0).mean(axis=0)

    # Determine fill thresholds for rows and columns using Otsu's method
    row_threshold = filters.threshold_otsu(row_fill_ratios)
    col_threshold = filters.threshold_otsu(col_fill_ratios)

    # Identify rows and columns with fill density above the threshold
    filled_rows = np.nonzero(row_fill_ratios > row_threshold)[0]
    filled_cols = np.nonzero(col_fill_ratios > col_threshold)[0]

    # Get the range of valid rows and columns for the inpaint area
    row_start, row_end = filled_rows.min(), filled_rows.max()
    col_start, col_end = filled_cols.min(), filled_cols.max()

    # Handle cases where the painting reaches the side boundaries
    if crop_side == "left" and filled_mask[(row_fill_ratios > row_threshold), -1].any():
        col_end = filled_mask.shape[1]  # Fill to the right edge if painting touches the right side
    elif crop_side == "right" and filled_mask[(row_fill_ratios > row_threshold), 0].any():
        col_start = 0  # Fill to the left edge if painting touches the left side

    # Fill the inpaint area within detected boundaries
    filled_mask[row_start:row_end + 1, col_start:col_end + 1] = 255

    # Set values outside the boundaries to 0 to ensure a clean background
    filled_mask[:row_start, :] = 0
    filled_mask[row_end + 1:, :] = 0
    filled_mask[:, :col_start] = 0
    filled_mask[:, col_end + 1:] = 0

    return filled_mask


def fill_painting_mask(mask: np.ndarray, side: str, n_margin_cols: int = 10, n_margin_rows: int = 10) -> np.ndarray:
    """
    Fills the mask of a painting by keeping only the largest connected components
    and clearing margins based on specified side.

    Parameters:
    -----------
    mask : np.ndarray
        Input binary mask where connected components are detected.
    side : str
        Specifies the side ("left" or "right") where extra filling should be applied.
    n_margin_cols : int, optional
        Number of columns at the left or right edge to clear for component identification. Default is 10.
    n_margin_rows : int, optional
        Number of rows at the top and bottom edges to clear for component identification. Default is 10.

    Returns:
    --------
    np.ndarray
        Mask where the largest components are filled and specified edges cleared.
    """
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    # Areas of each component, ignoring the background
    areas = stats[1:, cv2.CC_STAT_AREA]
    indices = np.arange(len(areas)) + 1  # Ignore background, so index starts from 1

    # Clear the specified margins based on the side
    if side == "left":
        labels[:, :n_margin_cols] = 0
    elif side == "right":
        labels[:, -n_margin_cols:] = 0
    labels[:n_margin_rows, :] = 0
    labels[-n_margin_rows:, :] = 0

    # Mask for the largest connected components
    largest_components_mask = np.isin(labels, indices).astype(np.uint8) * 255
    filled_largest_components_mask = inpaint_mask(largest_components_mask, side)

    return filled_largest_components_mask


def correct_vertical_transition_in_concatenated_mask(mask: np.ndarray) -> np.ndarray:
    """
    Corrects vertical transitions in a binary mask by filling a region defined by
    the boundary points in the leftmost and rightmost columns.

    Parameters:
    -----------
    mask : np.ndarray
        Input binary mask where transitions between regions need to be smoothed
        vertically across the concatenated areas.

    Returns:
    --------
    np.ndarray
        Mask with corrected vertical transitions, where the filled region is represented
        by 255 (foreground) and the rest remains 0 (background).
    """
    # Initialize a new mask for the corrected output
    corrected_mask = np.zeros(mask.shape, dtype=np.float32)

    # Identify non-zero pixel coordinates
    nonzero_rows, nonzero_cols = np.nonzero(mask)

    # Determine boundary coordinates in the leftmost and rightmost columns
    top_left_y = nonzero_rows[nonzero_cols == nonzero_cols.min()].min()
    top_right_y = nonzero_rows[nonzero_cols == nonzero_cols.max()].min()
    bottom_left_y = nonzero_rows[nonzero_cols == nonzero_cols.min()].max()
    bottom_right_y = nonzero_rows[nonzero_cols == nonzero_cols.max()].max()

    # Define the horizontal range and linearly interpolate top and bottom boundaries
    col_span = nonzero_cols.max() - nonzero_cols.min()
    top_boundary_y = np.linspace(top_left_y, top_right_y, num=col_span, endpoint=True, dtype=int)
    bottom_boundary_y = np.linspace(bottom_left_y, bottom_right_y, num=col_span, endpoint=True, dtype=int)
    x_positions = np.linspace(nonzero_cols.min(), nonzero_cols.max(), num=col_span, endpoint=True, dtype=int)

    # Fill the corrected mask based on interpolated boundary lines
    for y in range(corrected_mask.shape[0]):
        for idx, x in enumerate(x_positions):
            if top_boundary_y[idx] <= y <= bottom_boundary_y[idx]:
                corrected_mask[y, x] = 255

    return corrected_mask


def get_painting_masks(image: Image, n_margin_cols: int = 10, n_margin_rows: int = 10) -> np.ndarray:
    """
    Generates a concatenated mask of a painting image by processing the left and right sides separately,
    filling the masks, and applying vertical transition correction if necessary.

    Parameters:
    -----------
    image : PIL.Image
        Input RGB image of the painting.
    n_margin_cols : int, optional
        Number of columns to clear at the left or right edges when filling the mask. Default is 10.
    n_margin_rows : int, optional
        Number of rows to clear at the top and bottom edges when filling the mask. Default is 10.

    Returns:
    --------
    np.ndarray
        A concatenated binary mask for the painting, with vertical transition correction if required.
    """
    # Obtain initial side masks for the painting
    left_mask, right_mask = get_painting_side_masks(image)

    # Fill the painting masks for each side
    filled_left_mask = fill_painting_mask(left_mask, side="left", n_margin_cols=n_margin_cols, n_margin_rows=n_margin_rows)
    filled_right_mask = fill_painting_mask(right_mask, side="right", n_margin_cols=n_margin_cols, n_margin_rows=n_margin_rows)

    # Concatenate the filled masks along the horizontal axis
    concatenated_mask = np.concatenate((filled_left_mask, filled_right_mask), axis=1)

    # Apply vertical transition correction if there is overlap between left and right masks at the boundary
    if filled_left_mask[:, -1].any() or filled_right_mask[:, 0].any():
        concatenated_mask = correct_vertical_transition_in_concatenated_mask(concatenated_mask)

    return concatenated_mask
