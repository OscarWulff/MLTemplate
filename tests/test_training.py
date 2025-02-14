import os
from unittest.mock import MagicMock, patch

import torch

from machinelearningtemplate.model import FashionClassifierModel, ModelParams
from machinelearningtemplate.train import train


@patch("machinelearningtemplate.train.wandb.init")
@patch("machinelearningtemplate.train.wandb.log")
@patch("machinelearningtemplate.train.wandb.Artifact")
@patch("torch.utils.data.DataLoader")
@patch("machinelearningtemplate.train.corrupt_mnist")
def test_training_script(mock_corrupt_mnist, mock_dataloader, mock_wandb_artifact, mock_wandb_log, mock_wandb_init):
    """Test the training script."""
    # Mock the data loading function
    mock_train_set = MagicMock()
    mock_test_set = MagicMock()
    mock_corrupt_mnist.return_value = (mock_train_set, mock_test_set)

    # Mock DataLoader to return dummy data
    mock_train_loader = MagicMock()
    mock_train_loader.__iter__.return_value = iter([(torch.rand((1, 1, 28, 28)), torch.tensor([1]))])
    mock_dataloader.return_value = mock_train_loader

    # Mock wandb.init() to return a mock run object
    mock_run = MagicMock()
    mock_run.__enter__.return_value = mock_run  # Simulate context manager behavior
    mock_run.__exit__ = MagicMock()  # Simulate context exit
    mock_wandb_init.return_value = mock_run

    # Run the training script
    train()

    # Check if the model file is saved
    model_path = "models/model.pth"
    assert os.path.exists(model_path), f"Model file not found: {model_path}"

    # Load the saved model state dictionary
    model_state_dict = torch.load(model_path)

    # Define model parameters
    params = ModelParams(
        num_filters1=32, num_filters2=64, num_filters3=128, dropout_rate=0.5, num_fc_layers=2, ff_hidden_dim=256
    )

    # Create an instance of the model
    model = FashionClassifierModel(params)

    # Load the state dictionary into the model
    model.load_state_dict(model_state_dict)

    # Check if the model's state dictionary is loaded correctly
    for param_tensor in model.state_dict():
        assert param_tensor in model_state_dict, f"Missing parameter in state dictionary: {param_tensor}"

    print("Training script test passed.")


if __name__ == "__main__":
    test_training_script()
