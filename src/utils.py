import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from skimage import filters

def show_images(images: list[np.ndarray], n_cols: int = 3, output_file: str | Path | None = None):
    """Displays a list of images in a grid or saves to a file."""
    n_rows = (len(images) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 3 * n_rows))
    axes = axes.ravel()

    for idx, image in enumerate(images):
        image = np.array(image)
        # Display in grayscale if the image is 2D or has only one channel
        axes[idx].imshow(image, cmap='gray' if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1) else None)
        axes[idx].axis('off')

    for idx in range(len(images), n_rows * n_cols):
        axes[idx].axis('off')

    plt.tight_layout()

    if output_file is not None:
        plt.savefig(str(output_file), bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def show_image_with_mean_values(image_array: np.ndarray) -> None:
    """Displays an image with mean values of its rows and columns."""
    image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())
    mean_rows = image_array.mean(axis=1)
    mean_cols = image_array.mean(axis=0)
    threshold_rows = filters.threshold_otsu(mean_rows)
    threshold_cols = filters.threshold_otsu(mean_cols)

    fig = plt.figure(figsize=(8, 8))
    grid = plt.GridSpec(2, 2, hspace=0, wspace=0)

    ax_image = fig.add_subplot(grid[1, 0])
    ax_image.imshow(image_array, aspect='auto', cmap='gray')  # Always in grayscale
    ax_image.axis('off')

    ax_cols = fig.add_subplot(grid[0, 0], sharex=ax_image)
    ax_cols.plot(np.arange(len(mean_cols)), (mean_cols > threshold_cols), color='black', label='Above Threshold')
    ax_cols.plot(np.arange(len(mean_cols)), mean_cols, color='red', label='Mean Columns')
    ax_cols.axhline(y=threshold_cols, color='green', linestyle='--', label=f'Threshold = {threshold_cols:.2f}')
    ax_cols.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax_cols.set_yticklabels([0, 0.25, 0.5, 0.75, 1])
    ax_cols.grid(True)
    ax_cols.legend()

    ax_rows = fig.add_subplot(grid[1, 1], sharey=ax_image)
    ax_rows.plot((mean_rows > threshold_rows), np.arange(len(mean_rows)), color='black', label='Above Threshold')
    ax_rows.plot(mean_rows, np.arange(len(mean_rows)), color='blue', label='Mean Rows')
    ax_rows.axvline(x=threshold_rows, color='green', linestyle='--', label=f'Threshold = {threshold_rows:.2f}')
    ax_rows.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax_rows.set_xticklabels([0, 0.25, 0.5, 0.75, 1])
    ax_rows.grid(True)
    ax_rows.legend()

    plt.tight_layout()
    plt.show()
