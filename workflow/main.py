from dotenv import load_dotenv
import logging

from preprocess import preprocess
from train import train
from test import test
from package import package
from evaluate import evaluate

from ml_foundations_pytorch.utils.gpu_utils import print_device_info
from ml_foundations_pytorch.utils.wandb_utils import (
    create_wandb_project,
    print_wandb_project_runs,
)

load_dotenv()  # take environment variables from .env.

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def workflow() -> None:
    print_device_info()
    create_wandb_project()
    print_wandb_project_runs()
    preprocess()
    train()
    test()
    package()
    evaluate()


if __name__ == "__main__":
    workflow()
