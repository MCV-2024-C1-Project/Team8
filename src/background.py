import copy

import cv2
import numpy as np
from PIL import Image
from skimage import feature


def get_edges_canny(img_array: np.ndarray, sigma=1.0, low_threshold=None, high_threshold=None,
                    use_quantiles=False) -> np.ndarray:
    """
    Applies Canny edge detection to each color channel of the input image and combines the edges.

    Parameters:
    - img_array: np.ndarray - Input image array with 3 channels (RGB).
    - sigma: float - Standard deviation of the Gaussian filter (default is 1.0).
    - low_threshold: float or None - Low threshold for edge detection (optional).
    - high_threshold: float or None - High threshold for edge detection (optional).
    - use_quantiles: bool - Whether to interpret thresholds as quantiles (default is False).

    Returns:
    - np.ndarray - Binary edge image with detected edges in all channels combined.
    """
    # Apply Canny edge detection to each channel
    edges_r = feature.canny(img_array[:, :, 0], sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold,
                            use_quantiles=use_quantiles)
    edges_g = feature.canny(img_array[:, :, 1], sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold,
                            use_quantiles=use_quantiles)
    edges_b = feature.canny(img_array[:, :, 2], sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold,
                            use_quantiles=use_quantiles)

    # Combine edges (logical OR)
    edges = np.logical_or(np.logical_or(edges_r, edges_g), edges_b)

    # Convert boolean to int (optional)
    edges = edges.astype(np.uint8) * 255

    return edges


def merge_contours(contour1, contour2) -> np.ndarray:
    """
    Merges two contours by concatenating them and obtaining the convex hull.

    Parameters:
    - contour1: np.ndarray - First contour.
    - contour2: np.ndarray - Second contour.

    Returns:
    - np.ndarray - Merged contour representing the convex hull of the two input contours.
    """
    # Concatenate the two contours
    combined = np.concatenate((contour1, contour2))
    # Get the convex hull of the combined contours
    merged_contour = cv2.convexHull(combined)
    return merged_contour


def calculate_contour_distance(contour1, contour2, horizontal_threshold=0, vertical_threshold=0):
    """
    Calculates the distance between two contours if they meet specified horizontal and vertical overlap thresholds.

    Parameters:
    - contour1: np.ndarray - First contour.
    - contour2: np.ndarray - Second contour.
    - horizontal_threshold: int - Minimum horizontal overlap required (default is 0).
    - vertical_threshold: int - Minimum vertical overlap required (default is 0).

    Returns:
    - float - Distance between contour centers if overlap conditions are met; otherwise, infinity.
    """
    # Get bounding rectangles for both contours
    x1, y1, w1, h1 = cv2.boundingRect(contour1)
    c_x1 = x1 + w1 / 2
    c_y1 = y1 + h1 / 2

    x2, y2, w2, h2 = cv2.boundingRect(contour2)
    c_x2 = x2 + w2 / 2
    c_y2 = y2 + h2 / 2

    # Calculate horizontal and vertical overlaps
    horizontal_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    vertical_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))

    # Check if the horizontal and vertical overlap is above the given thresholds
    horizontal_condition = horizontal_overlap > horizontal_threshold
    vertical_condition = vertical_overlap > vertical_threshold

    # Return distance only if both conditions are met
    if horizontal_condition and vertical_condition:
        return max(abs(c_x1 - c_x2) - (w1 + w2) / 2, abs(c_y1 - c_y2) - (h1 + h2) / 2)
    else:
        return float('inf')  # Return a large distance if the conditions aren't met


def agglomerative_cluster(contours, threshold_distance=40.0, horizontal_threshold=0, vertical_threshold=0):
    """
    Performs agglomerative clustering on contours based on a distance threshold.

    Parameters:
    - contours: list of np.ndarray - List of contours to be clustered.
    - threshold_distance: float - Maximum distance to consider contours for merging (default is 40.0).
    - horizontal_threshold: int - Minimum horizontal overlap required for clustering (default is 0).
    - vertical_threshold: int - Minimum vertical overlap required for clustering (default is 0).

    Returns:
    - list of np.ndarray - List of clustered contours.
    """
    current_contours = copy.copy(contours)
    while len(current_contours) > 1:
        min_distance = None
        min_coordinate = None

        for x in range(len(current_contours) - 1):
            for y in range(x + 1, len(current_contours)):
                distance = calculate_contour_distance(
                    current_contours[x], current_contours[y],
                    horizontal_threshold=horizontal_threshold, vertical_threshold=vertical_threshold
                )
                if min_distance is None:
                    min_distance = distance
                    min_coordinate = (x, y)
                elif distance < min_distance:
                    min_distance = distance
                    min_coordinate = (x, y)

        if min_distance < threshold_distance:
            index1, index2 = min_coordinate
            current_contours[index1] = merge_contours(current_contours[index1], current_contours[index2])
            del current_contours[index2]
        else:
            break

    return current_contours


def enhance_image_clahe(image_array: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Enhances the input image using CLAHE on the L channel in LAB color space.

    Parameters:
    - image_array: np.ndarray - Input image in BGR color space.
    - clip_limit: float - Threshold for contrast limiting (default is 2.0).
    - tile_grid_size: tuple - Size of the grid for histogram equalization (default is (8, 8)).

    Returns:
    - np.ndarray - Enhanced image in BGR color space.
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image_array, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)

    # Merge channels and convert back to BGR color space
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return enhanced_img


def crop_image(image: np.ndarray, contour) -> np.ndarray:
    """
    Crops the region of the input image that is enclosed by a given contour.

    Parameters:
    - image: np.ndarray - Input image.
    - contour: np.ndarray - Contour defining the region to crop.

    Returns:
    - np.ndarray - Cropped region of the input image.
    """
    x, y, w, h = cv2.boundingRect(contour)
    return image[y:y + h, x:x + w]


def sort_contours(contours):
    """
    Sorts contours based on their spatial arrangement:
    - Horizontally if side by side
    - Vertically if on top of each other

    Parameters:
    - contours: list of contours (each contour is a numpy array of shape (N, 1, 2))

    Returns:
    - list of sorted contours
    """
    # Calculate bounding boxes for each contour
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

    # Determine orientation based on bounding box overlaps
    total_y_overlap = 0
    total_x_overlap = 0
    for i in range(len(bounding_boxes) - 1):
        x1, y1, w1, h1 = bounding_boxes[i]
        x2, y2, w2, h2 = bounding_boxes[i + 1]

        # Calculate overlap in y-axis and x-axis
        y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))

        total_y_overlap += y_overlap
        total_x_overlap += x_overlap

    # Determine the predominant alignment based on total overlap
    is_horizontal = total_y_overlap > total_x_overlap

    # Sort contours based on x-coordinates if horizontal, or y-coordinates if vertical
    if is_horizontal:
        sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])  # Sort by x-coordinate
    else:
        sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])  # Sort by y-coordinate

    return sorted_contours

def resize_image(image_array: np.ndarray, image_size: int) -> np.ndarray:
    """
    Resizes the input image to the specified size using cubic interpolation.

    Parameters:
    - image_array: np.ndarray - Input image to be resized.
    - image_size: int - The new size for both dimensions (height and width) of the image.

    Returns:
    - np.ndarray - Resized image.
    """
    return cv2.resize(image_array, dsize=(image_size, image_size), interpolation=cv2.INTER_CUBIC)

def get_paintings_cropped_images(
        image: Image,
        image_size: int | None = 2 ** 9,
        min_aspect_ratio: float = 0.125,
        max_aspect_ratio: float = 8.0,
        max_distance_to_merge: int = 10,
        sigma: float = 2.0,
        output_size: int | None = 2 ** 9
) -> list[Image]:
    """
    Detects and crops rectangular regions of interest (such as paintings) from an input image. The function applies
    edge detection, contour extraction, filtering based on aspect ratio, and agglomerative clustering for grouping
    nearby contours.

    Parameters:
    - image: PIL Image - Input image containing regions of interest to be detected and cropped.
    - image_size: int or None - Size to which the input image should be resized before processing. Default is 512.
    - min_aspect_ratio: float - Minimum aspect ratio of contours to be considered. Default is 0.125.
    - max_aspect_ratio: float - Maximum aspect ratio of contours to be considered. Default is 8.0.
    - max_distance_to_merge: int - Maximum distance for merging nearby contours during agglomerative clustering. Default is 10.
    - sigma: float - Standard deviation for Gaussian filter in Canny edge detection. Controls the degree of edge smoothing. Default is 2.0.
    - output_size: int or None - Size to which each cropped region should be resized. Default is 512.

    Returns:
    - list[Image] - List of cropped PIL Images of detected regions of interest.

    Process:
    1. Resizes the input image if `image_size` is provided.
    2. Enhances the image contrast using CLAHE.
    3. Detects edges in the enhanced image using Canny edge detection with the specified `sigma`.
    4. Extracts contours from the edge-detected image.
    5. Filters contours based on their aspect ratio and size.
    6. Merges nearby contours using agglomerative clustering with a distance threshold.
    7. Sorts the merged contours for a consistent output order.
    8. Crops the regions corresponding to each merged contour.
    9. Resizes each cropped region if `output_size` is specified and returns them as a list of PIL Images.
    """
    # Convert the image to a NumPy array for processing
    image_array = np.array(image)

    # Resize image if `image_size` is specified
    if image_size:
        image_array = resize_image(image_array, image_size)

    # Enhance image contrast using CLAHE
    enhanced_image = enhance_image_clahe(image_array)

    # Apply Canny edge detection
    edges = get_edges_canny(enhanced_image, sigma=sigma)

    # Find contours from the edge image
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Filter contours based on aspect ratio and size criteria
    filtered_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if (h >= image_array.shape[0] / 4 or w >= image_array.shape[1] / 4) and (
                min_aspect_ratio < aspect_ratio < max_aspect_ratio):
            filtered_contours.append(contour)

    # Merge nearby contours using agglomerative clustering
    merged_contours = agglomerative_cluster(filtered_contours, max_distance_to_merge)

    # Sort contours if more than one exists
    if len(merged_contours) > 1:
        merged_contours = sort_contours(merged_contours)

    # Crop each region corresponding to a merged contour
    cropped_images = [crop_image(image_array, contour) for contour in merged_contours]

    # Resize each cropped image if `output_size` is specified
    if output_size:
        cropped_images = [resize_image(crop, output_size) for crop in cropped_images]

    # Convert cropped images to PIL format and return
    return [Image.fromarray(crop) for crop in cropped_images]
