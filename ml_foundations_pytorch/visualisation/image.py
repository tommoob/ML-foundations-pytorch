""" Custom functions for ml_foundations_pytorch specific plots."""
from matplotlib import pyplot as plt
import numpy as np


def visualise_images(images: np.ndarray, num_rows: int = 4, num_cols: int = 4) -> None:
    """Visualizes a grid of images.

    Args:
        images: A numpy array of images to visualize.
        num_rows: Number of rows in the visualization grid.
        num_cols: Number of columns in the visualization grid.
    """
    num_images = min(num_rows * num_cols, images.shape[0])
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(1.5 * num_cols, 2 * num_rows))
    for i in range(num_images):
        ax = axes[i // num_cols, i % num_cols]
        ax.imshow(images[i], cmap="gray")
        ax.axis("off")
        ax.set_title(f"Image {i}")
    plt.tight_layout()
    plt.show()