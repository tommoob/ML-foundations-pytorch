import torch
from torch.utils.data import Dataset
import numpy as np


class MNISTDataset(Dataset):
    """PyTorch-compatible dataset for MNIST."""

    def __init__(self, images: np.ndarray, labels: np.ndarray):
        """
        Args:
            images (np.ndarray): Array of images.
            labels (np.ndarray): Array of labels.
        """
        self.images = torch.tensor(images, dtype=torch.float32) / 255.0  # Normalize to [0, 1]
        self.labels = torch.tensor(labels, dtype=torch.long)  # Labels as integers

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.images)

    def __getitem__(self, idx):
        """Fetches an image-label pair by index."""
        image = self.images[idx]
        label = self.labels[idx]

        return image, label