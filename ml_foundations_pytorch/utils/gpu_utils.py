""" Custom functions for ml_foundations_pytorch specific GPU utilities"""

import logging
import torch


def print_device_info() -> None:
    """Show information for the current device, whether GPU or CPU

    Returns: None, logs info about the device to the console.

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    if device.type == "cuda":
        logging.info(f"Device name: {torch.cuda.get_device_name(0)}")
        logging.info("Memory Usage:")
        logging.info(
            f"Allocated: {round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1)}GB"
        )
        logging.info(f"Cached: {round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)}GB")
