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
    """Create PR_curves of the model perormance on the testset

    model_dir
        The save directory for the model
    config_values
        The contents of the data.yaml file
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
    """Runs evaluation script (to create PR_curves) with click arguments

    model_num
        The number of the model being tested (for exp6/model.pth, model_num=6)
    config
        The path to the data.yaml file
    """
    config_values = load_yaml_as_box(config)
    model_dir = f"experiments/runs/train/exp{model_num}"
    evaluate(model_dir, config_values)


if __name__ == "__main__":
    run_evaluation()
