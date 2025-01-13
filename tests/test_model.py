from dataclasses import dataclass
from pathlib import Path

import hydra
import torch
import typer
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn


@staticmethod
def create_model_from_model_params_yaml(params: dict) -> 'FashionClassifierModel':
    model_params = ModelParams(
        num_filters1=params["num_filters1"],
        num_filters2=params["num_filters2"],
        num_filters3=params["num_filters3"],
        dropout_rate=params["dropout_rate"],
        num_fc_layers=params["num_fc_layers"],
        ff_hidden_dim=params["ff_hidden_dim"]
    )
    return FashionClassifierModel(model_params)



@dataclass
class ModelParams:
    num_filters1: int = 32
    num_filters2: int = 64
    num_filters3: int = 128
    dropout_rate: float = 0.5
    num_fc_layers: int = 1
    ff_hidden_dim: int = 256

@dataclass
class Config:
    params: ModelParams

class FashionClassifierModel(nn.Module):
    """My awesome model with configurable parameters."""

    def __init__(self, params: ModelParams) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, params.num_filters1, 3, 1)
        self.conv2 = nn.Conv2d(params.num_filters1, params.num_filters2, 3, 1)
        self.conv3 = nn.Conv2d(params.num_filters2, params.num_filters3, 3, 1)
        self.dropout = nn.Dropout(params.dropout_rate)

        # Dynamically define fully connected layers
        fc_layers = []
        input_dim = params.num_filters3
        for _ in range(params.num_fc_layers - 1):
            fc_layers.append(nn.Linear(input_dim, params.ff_hidden_dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(params.dropout_rate))
            input_dim = params.ff_hidden_dim

        fc_layers.append(nn.Linear(input_dim, 10))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

project_root = Path(__file__).resolve().parents[2]  # Adjust as needed
config_path = str(project_root / "configs")

# Update the @hydra.main decorator in model.py
@hydra.main(config_path=config_path, config_name="default_config.yaml", version_base=None)
def main(cfg):
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    params = cfg.model_experiments.params
    # Usage
    model = create_model_from_model_params_yaml(params)
    model = FashionClassifierModel(params)
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Test the model with dummy input
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")



if __name__ == "__main__":
    main()