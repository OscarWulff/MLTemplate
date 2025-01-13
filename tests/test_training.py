import os
import torch
from unittest.mock import patch, MagicMock
from machinelearningtemplate.train import train
from machinelearningtemplate.model import FashionClassifierModel, ModelParams

@patch("machinelearningtemplate.train.wandb.init")
@patch("machinelearningtemplate.train.wandb.log")
@patch("machinelearningtemplate.train.wandb.Artifact")
@patch("machinelearningtemplate.train.corrupt_mnist")
def test_training_script(mock_corrupt_mnist, mock_wandb_artifact, mock_wandb_log, mock_wandb_init):
    """Test the training script."""
    # Mock the data loading function
    mock_train_set = MagicMock()
    mock_test_set = MagicMock()
    mock_corrupt_mnist.return_value = (mock_train_set, mock_test_set)

    # Run the training script
    train()

    # Check if the model file is saved
    model_path = "models/model.pth"
    assert os.path.exists(model_path), f"Model file not found: {model_path}"

    # Load the saved model state dictionary
    model_state_dict = torch.load(model_path)

    # Define model parameters
    params = ModelParams(
        num_filters1=32,
        num_filters2=64,
        num_filters3=128,
        dropout_rate=0.5,
        num_fc_layers=2,
        ff_hidden_dim=256
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