""" Custom functions for ml_foundations_pytorch specific WandB operations."""

from dotenv import load_dotenv
import wandb

import os
import logging

load_dotenv()  # take environment variables from .env.


def create_wandb_project() -> None:
    """Checks if a WandB project exists, if not, create it.

    Returns: None

    """
    api = wandb.Api()
    wandb_entity = os.environ.get("WANDB_ENTITY")
    wandb_project_name = os.environ.get("WANDB_PROJECT")
    try:
        # Check the project can be queried
        _ = len(api.runs(path=f'{os.environ.get("WANDB_ENTITY")}/{wandb_project_name}'))
        logging.info(
            f"WandB project '{wandb_project_name}' already exists for entity '{wandb_entity}'"
        )
    except ValueError:
        logging.info(
            f"WandB project '{wandb_project_name}' doesn't exist for entity '{wandb_entity}', creating..."
        )
        api.create_project(wandb_project_name, wandb_entity)
        logging.info("WandB project created! 🚀")


def print_wandb_project_runs() -> None:
    """Lists information about the current project defined from variables in .env

    Returns: None, logs the number of runs in the current project.

    """
    api = wandb.Api()
    wandb_project_name = os.environ.get("WANDB_PROJECT")
    try:
        runs = api.runs(path=f'{os.environ.get("WANDB_ENTITY")}/{wandb_project_name}')
        logging.info(
            f"WandB Project: '{wandb_project_name}', number of runs: {len(runs)}"
        )
    except ValueError:
        logging.warning(
            f"WandB Project '{wandb_project_name}' doesn't exist, ensure it is created"
            f"before logging runs."
        )
