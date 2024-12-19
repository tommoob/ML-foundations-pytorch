import logging
from ml_foundations_pytorch.data import MNISTdataloader
from ml_foundations_pytorch.visualisation import image
from models.MNIST import MNISTTwoLayerModel
import torch
import torch.nn as nn


def train() -> None:
    """Function for running custom training code.

    Returns: None

    """
    logging.info("Training ML model...")

    dataloader = MNISTdataloader.MNISTDataLoader()
    data_path = "/Users/thomas.obrien/dev/src/ML-Foundations-Pytorch/data/interim/MNIST"
    dataloader.MnistLoader(data_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model, loss function, and optimizer
    model = MNISTTwoLayerModel().to(device)
    criterion = nn.NLLLoss()  # Negative Log-Likelihood Loss for classification
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    


if __name__ == "__main__":
    train()
