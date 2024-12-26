import logging
from models.MNIST.hidden_layer import MNISTTwoLayerModel
from models.MNIST.convolutional_network import ConvolutionalNetwork
from torchvision import datasets, transforms
import torch
import click
import torch.nn as nn
from box import ConfigBox
from ml_foundations_pytorch.utils.file_utils import (
    get_next_experiment_dir,
    save_model,
    load_yaml_as_box,
)


def train(epochs: int, config_values: ConfigBox) -> None:
    """Function for running custom training code.

    Parameters
    ----------
    epochs
        The number of passes over the dataset training will include
    config_values
        The contents of the data.yaml file

    """
    logging.info("Training ML model...")

    # data_path = "/Users/thomas.obrien/dev/src/ML-Foundations-Pytorch/data/interim/MNIST"
    # dataloader = MNISTdataloader.MNISTDataLoader()
    # train_loader = dataloader.orchestrate_MNIST_dataloading(
    #    data_path, dataloader.train_images, dataloader.train_labels
    # )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Convert PIL.Image to torch.Tensor
            transforms.Normalize((0.5,), (0.5,)),  # Normalize images to [-1, 1]
        ]
    )
    if config_values.dataset == "MNIST":
        train_dataset = datasets.MNIST(
            root="./data", train=True, transform=transform, download=True
        )
    elif config_values.dataset == "imagenet":
        train_dataset = datasets.ImageNet(
            root="./data", train=True, transform=transform, download=False
        )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model, loss function, and optimizer
    if config_values.model_type == "hidden_layer":
        model = MNISTTwoLayerModel().to(device)
        criterion = nn.NLLLoss()
    elif config_values.model_type == "convolution":
        model = ConvolutionalNetwork(kernel_size=3).to(device)
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    training_loop(epochs, model, device, train_loader, optimizer, criterion)

    model_save_root = "experiments/runs/train"
    exp_model_dir = get_next_experiment_dir(model_save_root)
    save_model(model, save_dir=exp_model_dir)


def training_loop(
    epochs: int,
    model: torch.nn.Module,
    device: str,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
):
    """
    Run training loop

    Parameters
    ----------
    epochs
        The number of passes over the dataset training will include
    model
        The model weights object being trained
    device
        The device on which to do training (cpu vs gpu)
    train_loader
        The training data and labels organised into a batched dataloader object
    optimizer
        Used to adjust model weights during training
    criterion
        The function used to calculate the loss in the system


    """
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


@click.command()
@click.option(
    "--epochs",
    type=int,
    default="10",
    help="Numbers of iterations to train for.",
)
@click.option(
    "--config",
    type=str,
    default="config/yolov7/data.yaml",
    help="COnfig where information about the dataset and model type are stored.",
)
def run_training(epochs: int, config: str):
    config_values = load_yaml_as_box(config)

    train(epochs, config_values)


if __name__ == "__main__":
    run_training()
