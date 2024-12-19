import os
from typing import Dict
import numpy as np


class MNISTDataLoader:
    """A class for ingesting and processing MNIST data."""
    
    def __init__(self):
        self.images: Dict[str, np.ndarray] = {}
        self.labels: Dict[str, np.ndarray] = {}
        self.train_images = "train-images-idx3-ubyte"
        self.val_images = "t10k-images-idx3-ubyte"
        self.train_labels = "train-labels-idx1-ubyte"
        self.val_labels = "train-labels-idx1-ubyte"

    def load_labels(self, path_dir: str, filename: str, byte_size: int) -> None:
        """Loads MNIST labels from a binary file.

        Args:
            path_dir: The directory containing the label file.
            filename: The label file name.
            byte_size: The number of bytes to read per label.
        """
        number_training_examples, number_val_examples = 60007, 10007
        header_size = 0
        file_path = os.path.join(path_dir, filename)
        num_labels = number_training_examples if "train" in filename else number_val_examples
        labels = np.zeros(num_labels, dtype=np.uint8)

        try:
            with open(file_path, "rb") as file:
                file.seek(header_size)
                for i in range(num_labels):
                    byte = file.read(byte_size)
                    if not byte:
                        break
                    labels[i] = int.from_bytes(byte, byteorder="big")
        except FileNotFoundError:
            raise FileNotFoundError(f"Label file not found: {file_path}")
        except Exception as e:
            raise RuntimeError(f"Error reading label file: {e}")

        self.labels[filename] = labels

    def load_images(self, path_dir: str, filename: str) -> None:
        """Loads MNIST images from a binary file.

        Args:
            path_dir: The directory containing the image file.
            filename: The image file name.
        """
        file_path = os.path.join(path_dir, filename)
        image_dimension, header_size = 28, 16

        try:
            data = np.fromfile(file_path, dtype=np.uint8)[header_size:]
            num_images = data.size // (image_dimension * image_dimension)
            images = data.reshape(num_images, image_dimension, image_dimension)
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found: {file_path}")
        except Exception as e:
            raise RuntimeError(f"Error reading image file: {e}")

        self.images[filename] = images
    
    def MnistLoader(self, data_path):
        data_files = os.listdir(data_path)
        byte_num = 1
        for file in data_files:
            if "labels" in file:
                self.load_labels(data_path, file, byte_num)
            elif "images" in file:
                self.load_images(data_path, file)