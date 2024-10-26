import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


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
        plt.savefig(str(output_file), bbox_inches='tight')
        plt.close()
    else:
        plt.show()
