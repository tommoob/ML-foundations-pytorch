import os
import torch
from models.MNIST.hidden_layer import MNISTTwoLayerModel

def get_next_experiment_dir(save_root: str, exp_prefix: str = "exp") -> str:
    """
    Finds the next available experiment directory in the save root.
    
    Args:
        save_root (str): The root directory where experiments are saved.
        exp_prefix (str): Prefix for experiment directories (default: "exp").
        
    Returns:
        str: Path to the next available experiment directory.
    """
    # Ensure the root directory exists
    os.makedirs(save_root, exist_ok=True)
    
    # Find the next available experiment folder
    exp_number = 1
    while True:
        exp_dir = os.path.join(save_root, f"{exp_prefix}{exp_number}")
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)  # Create the directory
            return exp_dir
        exp_number += 1

def save_model(model: torch.nn.Module, save_dir: str, model_name: str = "model.pth"):
    """
    Saves a PyTorch model to the specified directory.
    
    Args:
        model (torch.nn.Module): The PyTorch model to save.
        save_dir (str): Directory where the model will be saved.
        model_name (str): Name of the saved model file (default: "model.pth").
    """
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the model
    save_path = os.path.join(save_dir, model_name)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def load_model(model_path):
    model = MNISTTwoLayerModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model