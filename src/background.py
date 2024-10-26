import cv2
import numpy as np
from PIL import Image
from skimage import filters


def convert_bool_to_uint8(binary_array: np.ndarray) -> np.ndarray:
    """Convert a boolean array to `uint8`, scaling values by 255."""
    assert binary_array.dtype == np.bool_, "Input must be boolean."
    return binary_array.astype(np.uint8) * 255


def apply_otsu_threshold(image: np.ndarray) -> np.ndarray:
    """Apply Otsu's threshold, returning a binary mask in `uint8` format."""
    binary_mask = image > filters.threshold_otsu(image)
    return convert_bool_to_uint8(binary_mask)


def get_gradient_magnitude(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Calculate gradient magnitude using specified kernel size."""
    assert kernel_size % 2 != 0, f"Kernel size must be odd, got {kernel_size}."
    
    kernely = np.array([[1] * kernel_size, [1] * kernel_size, [0] * kernel_size, [-1] * kernel_size, [-1] * kernel_size])
    kernelx = kernely.T
    gradient_x = cv2.filter2D(image, cv2.CV_64F, kernelx)
    gradient_y = cv2.filter2D(image, cv2.CV_64F, kernely)
    return np.sqrt(gradient_x ** 2 + gradient_y ** 2)


def get_painting_side_masks(image: Image) -> (np.ndarray, np.ndarray):
    """Generate binary masks for the left and right halves of an image."""
    assert image.mode == "RGB", "Image must be RGB."
    
    grayscale = np.array(image.convert("L"))
    half_image = grayscale.shape[1] // 2
    left_mask = apply_otsu_threshold(get_gradient_magnitude(grayscale[:, :half_image]))
    right_mask = apply_otsu_threshold(get_gradient_magnitude(grayscale[:, half_image:]))
    
    return left_mask, right_mask


def inpaint_mask(mask: np.ndarray, crop_side: str, n_margin_cols: int = 10, min_pctg_col: float = 0.2) -> np.ndarray:
    """Fill gaps in a binary mask, adjusting fill based on crop side."""
    filled_mask = mask.copy()
    row_fill_ratios, col_fill_ratios = (filled_mask > 0).mean(axis=1), (filled_mask > 0).mean(axis=0)
    row_start, row_end = np.nonzero(row_fill_ratios > filters.threshold_otsu(row_fill_ratios))[0][[0, -1]]
    col_start, col_end = np.nonzero(col_fill_ratios > filters.threshold_otsu(col_fill_ratios))[0][[0, -1]]
    
    if crop_side == "left" and mask[:, -1].any():
        col_end = filled_mask.shape[1]
    elif crop_side == "right" and mask[:, 0].any():
        col_start = 0

    filled_mask[row_start:row_end + 1, col_start:col_end + 1] = 255
    filled_mask[:row_start, :], filled_mask[row_end + 1:, :] = 0, 0
    filled_mask[:, :col_start], filled_mask[:, col_end + 1:] = 0, 0
    
    return filled_mask


def fill_painting_mask(mask: np.ndarray, side: str, n_margin_cols: int = 10, n_margin_rows: int = 10) -> np.ndarray:
    """Fill mask with largest connected components and clear specified margins."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if side == "left":
        labels[:, :n_margin_cols] = 0
    elif side == "right":
        labels[:, -n_margin_cols:] = 0
    labels[:n_margin_rows, :], labels[-n_margin_rows:, :] = 0, 0

    largest_components_mask = np.isin(labels, np.arange(1, num_labels)).astype(np.uint8) * 255
    return inpaint_mask(largest_components_mask, side, n_margin_cols)


def correct_vertical_transition_in_concatenated_mask(mask: np.ndarray) -> np.ndarray:
    """Correct vertical transitions by filling regions between boundary points."""
    corrected_mask = np.zeros(mask.shape, dtype=np.float32)
    nonzero_rows, nonzero_cols = np.nonzero(mask)
    
    col_span = nonzero_cols.max() - nonzero_cols.min()
    top_boundary_y = np.linspace(nonzero_rows[nonzero_cols == nonzero_cols.min()].min(),
                                 nonzero_rows[nonzero_cols == nonzero_cols.max()].min(), col_span, dtype=int)
    bottom_boundary_y = np.linspace(nonzero_rows[nonzero_cols == nonzero_cols.min()].max(),
                                    nonzero_rows[nonzero_cols == nonzero_cols.max()].max(), col_span, dtype=int)
    
    for y in range(corrected_mask.shape[0]):
        for idx, x in enumerate(np.linspace(nonzero_cols.min(), nonzero_cols.max(), col_span, dtype=int)):
            if top_boundary_y[idx] <= y <= bottom_boundary_y[idx]:
                corrected_mask[y, x] = 255

    return corrected_mask


def get_painting_masks(image: Image, n_margin_cols: int = 10, n_margin_rows: int = 10) -> np.ndarray:
    """Generate a mask for a painting by processing each side and applying vertical correction."""
    left_mask, right_mask = get_painting_side_masks(image)
    filled_left_mask = fill_painting_mask(left_mask, "left", n_margin_cols, n_margin_rows)
    filled_right_mask = fill_painting_mask(right_mask, "right", n_margin_cols, n_margin_rows)
    
    concatenated_mask = np.concatenate((filled_left_mask, filled_right_mask), axis=1)
    if filled_left_mask[:, -1].any() and filled_right_mask[:, 0].any():
        concatenated_mask = correct_vertical_transition_in_concatenated_mask(concatenated_mask)
    return concatenated_mask


import numpy as np
import cv2

def crop_image_by_mask(image, mask):
    image = np.array(image)
    mask = (mask > 0).astype(np.uint8)
    num_labels, labels_im = cv2.connectedComponents(mask)
    cropped_images = []

    for label in range(1, num_labels):
        component_mask = np.zeros_like(mask)
        component_mask[labels_im == label] = 1
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check if contours are found
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            
            # Check if the width and height are greater than 10 pixels
            if w > 10 and h > 10:
                cropped_image = image[y:y + h, x:x + w]
                cropped_images.append(Image.fromarray(cropped_image))

    return cropped_images
