import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from skimage import filters


def show_images(images: list[np.ndarray], n_cols: int = 3, output_file: str | Path | None = None):
    """
    Displays a list of images in a grid format or saves them to a file.

    Parameters:
    -----------
    images : List[np.ndarray]
        List of images to display.
    n_cols : int, optional
        Number of columns in the grid layout. Default is 3.
    output_file : str | Path | None, optional
        Path to save the image grid. If None, displays the images on screen. Default is None.
    """
    # Calculate the required number of rows
    n_rows = (len(images) + n_cols - 1) // n_cols

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 3 * n_rows))

    # Flatten axes array for easier iteration in case of multiple rows
    axes = axes.ravel()

    # Display each image in its respective subplot
    for idx, image in enumerate(images):
        axes[idx].imshow(image)
        axes[idx].axis('off')

    # Turn off any unused axes
    for idx in range(len(images), n_rows * n_cols):
        axes[idx].axis('off')

    plt.tight_layout()

    # Save to file if output path is provided, otherwise show plot
    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_file), bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def show_image_with_mean_values(image_array: np.ndarray, output_file: str | Path | None = None) -> None:
    """
    Displays an image along with the mean values of its rows and columns.

    The function normalizes the input image, computes the mean values for both
    rows and columns, applies Otsu's method to find optimal thresholds, and
    visualizes the results in a grid layout.

    Parameters:
    -----------
    image_array : np.ndarray
        The input image array to be displayed. It should be a 2D array
        representing a grayscale image.

    Returns:
    --------
    None
    """
    # Copy and normalize the image array
    image_array = np.copy(image_array)
    image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())

    # Calculate mean values for rows and columns
    mean_rows = image_array.mean(axis=1)
    mean_cols = image_array.mean(axis=0)

    # Calculate thresholds using Otsu's method
    threshold_rows = filters.threshold_otsu(mean_rows)
    threshold_cols = filters.threshold_otsu(mean_cols)

    # Create a figure with subplots
    fig = plt.figure(figsize=(8, 8))
    grid = plt.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], hspace=0, wspace=0)

    # Main image subplot
    ax_image = fig.add_subplot(grid[1, 0])
    ax_image.imshow(image_array, aspect='auto')
    ax_image.axis('off')  # Turn off axis labels for the main image

    # Plot mean of columns (top)
    ax_cols = fig.add_subplot(grid[0, 0], sharex=ax_image)
    ax_cols.plot(np.arange(len(mean_cols)), (mean_cols > threshold_cols), color='black', label='Above Threshold')
    ax_cols.plot(np.arange(len(mean_cols)), mean_cols, color='red', label='Mean Columns')
    ax_cols.axhline(y=threshold_cols, color='green', linestyle='--', label=f'Threshold = {threshold_cols:.2f}')
    ax_cols.set_xticks([])
    ax_cols.set_yticks([0, 0.25, 0.5, 0.75, 1])  # Set ticks for normalized values
    ax_cols.set_yticklabels([0, 0.25, 0.5, 0.75, 1])
    ax_cols.grid(True)
    ax_cols.legend()

    # Plot mean of rows (right)
    ax_rows = fig.add_subplot(grid[1, 1], sharey=ax_image)
    ax_rows.plot((mean_rows > threshold_rows), np.arange(len(mean_rows)), color='black', label='Above Threshold')
    ax_rows.plot(mean_rows, np.arange(len(mean_rows)), color='blue', label='Mean Rows')
    ax_rows.axvline(x=threshold_rows, color='green', linestyle='--', label=f'Threshold = {threshold_rows:.2f}')
    ax_rows.set_xticks([0, 0.25, 0.5, 0.75, 1])  # Set ticks for normalized values
    ax_rows.set_xticklabels([0, 0.25, 0.5, 0.75, 1])
    ax_rows.set_yticks([])
    ax_rows.grid(True)
    ax_rows.legend()

    # Save to file if output path is provided, otherwise show plot
    plt.tight_layout()
    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_file), bbox_inches='tight')
        plt.close()
    else:
        plt.show()