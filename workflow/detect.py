import os
import click
from ml_foundations_pytorch.utils.file_utils import load_model, load_yaml_as_box
from ml_foundations_pytorch.utils.model_utils import (
    display_sample_predictions,
)
from torchvision import datasets, transforms
import torch
from box import ConfigBox


def detect(model_dir: str, config_values: ConfigBox):
    """Runs detection on testset

    model_dir
        The save directory for the model
    config_values
        The contents of the data.yaml file
    """
    # Paths
    model_path = os.path.join(model_dir, "model.pth")

    # Load Model
    model = load_model(model_path)

    # Prepare Test Loader
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Convert PIL.Image to torch.Tensor
            transforms.Normalize((0.5,), (0.5,)),  # Normalize images to [-1, 1]
        ]
    )
    if config_values.dataset == "MNIST":
        test_dataset = datasets.MNIST(
            root="./data", train=False, transform=transform, download=True
        )
    elif config_values.dataset == "imagenet":
        test_dataset = datasets.ImageNet(
            root="./data", train=False, transform=transform, download=False
        )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False
    )
    # data_path = "/Users/thomas.obrien/dev/src/ML-Foundations-Pytorch/data/interim/MNIST"
    # dataloader = MNISTdataloader.MNISTDataLoader()
    # test_loader = dataloader.orchestrate_MNIST_dataloading(
    #    data_path,
    #    dataloader.val_images,
    #    dataloader.val_labels
    #    )

    # Run Inference
    # predictions = run_inference(model, test_loader)

    # Display Predictions
    display_sample_predictions(model, test_loader)


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
def run_detection(model_num: int, config: str):
    """Runs detection script with click arguments

    model_num
        The number of the model being tested (for exp6/model.pth, model_num=6)
    config
        The path to the data.yaml file
    """
    config_values = load_yaml_as_box(config)
    model_dir = f"experiments/runs/train/exp{model_num}"
    detect(model_dir, config_values)


if __name__ == "__main__":
    run_detection()
