import os
import click
from ml_foundations_pytorch.utils.file_utils import load_model
from ml_foundations_pytorch.utils.model_utils import (
    display_sample_predictions,
)
from torchvision import datasets, transforms
import torch


def detect(model_dir: str):
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
    test_dataset = datasets.MNIST(
        root="./data", train=False, transform=transform, download=True
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
def run_detection(model_num: int):
    model_dir = f"experiments/runs/train/exp{model_num}"
    detect(model_dir)


if __name__ == "__main__":
    run_detection()
