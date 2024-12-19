""" Custom functions for ml_foundations_pytorch specific pre-processing."""
import numpy as np


def one_hot_encode_labels(self, labels: np.ndarray) -> np.ndarray:
    """One-hot encodes a set of labels.

    Args:
        labels: A numpy array of labels.

    Returns:
        A one-hot encoded 2D numpy array.
    """
    num_classes = 10
    one_hot = np.zeros((num_classes, labels.size), dtype=np.uint8)
    for label in range(num_classes):
        one_hot[label] = (labels == label).astype(np.uint8)
    return one_hot

