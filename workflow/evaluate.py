import logging
from ml_foundations_pytorch.utils.statistical_tools import mnist_detection_statistics
from ml_foundations_pytorch.utils.file_utils import load_model, load_yaml_as_box
import os
import click
from torchvision import datasets, transforms
import torch
from box import ConfigBox
from models.MNIST.hidden_layer import MNISTTwoLayerModel
from models.MNIST.convolutional_network import ConvolutionalNetwork


def evaluate(model_dir: str, config_values: ConfigBox) -> None:
    """Function for running custom model evaluation code.

    Returns: None

    """
    logging.info("Evaluating ML model...")
    model_path = os.path.join(model_dir, "model.pth")

    # Load Model
    if config_values.model_type == "hidden_layer":
        model = MNISTTwoLayerModel()
    elif config_values.model_type == "convolution":
        model = ConvolutionalNetwork()

    model = load_model(model_path, model)

    confidenceThreshold = 0.4

    # data_path = "/Users/thomas.obrien/dev/src/ML-Foundations-Pytorch/data/interim/MNIST"
    # dataloader = MNISTdataloader.MNISTDataLoader()
    # test_loader = dataloader.orchestrate_MNIST_dataloading(
    #    data_path, dataloader.val_images, dataloader.val_labels
    # )
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Convert PIL.Image to torch.Tensor
            transforms.Normalize((0.5,), (0.5,)),  # Normalize images to [-1, 1]
        ]
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, transform=transform, download=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False
    )

    mnist_detection_statistics(model, test_loader, confidenceThreshold)


@click.command()
@click.option(
    "--model_num",
    type=int,
    default="3",
    help="Which training run to use.",
)
@click.option(
    "--config",
    type=str,
    default="config/yolov7/data.yaml",
    help="COnfig where information about the dataset and model type are stored.",
)
def run_evaluation(model_num: int, config: str):
    config_values = load_yaml_as_box(config)
    model_dir = f"experiments/runs/train/exp{model_num}"
    evaluate(model_dir, config_values)


if __name__ == "__main__":
    run_evaluation()
