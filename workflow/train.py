import logging
from ml_foundations_pytorch.data import MNISTdataloader
from ml_foundations_pytorch.data import MNISTdataset
from ml_foundations_pytorch.visualisation import image
from models.MNIST.hidden_layer import MNISTTwoLayerModel
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from ml_foundations_pytorch.utils.file_utils import (
    get_next_experiment_dir,
    save_model
    )


def train() -> None:
    """Function for running custom training code.

    Returns: None

    """
    logging.info("Training ML model...")

    dataloader = MNISTdataloader.MNISTDataLoader()
    data_path = "/Users/thomas.obrien/dev/src/ML-Foundations-Pytorch/data/interim/MNIST"
    dataloader.MnistLoader(data_path)

    trainDatasetMNIST = MNISTdataset.MNISTDataset(
        images=dataloader.images[dataloader.train_images],
        labels=dataloader.labels[dataloader.train_labels]
        )

    train_loader = DataLoader(trainDatasetMNIST, batch_size=64, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model, loss function, and optimizer
    model = MNISTTwoLayerModel().to(device)
    criterion = nn.NLLLoss()  # Negative Log-Likelihood Loss for classification
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    

    epochs = 10

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

    model_save_root = "runs/train"
    exp_model_dir = get_next_experiment_dir(model_save_root)
    save_model(model, save_dir=exp_model_dir)


if __name__ == "__main__":
    train()
