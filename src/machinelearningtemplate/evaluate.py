import os
from pathlib import Path

import hydra
import torch
import typer
from omegaconf import DictConfig

from machinelearningtemplate.data import corrupt_mnist
from machinelearningtemplate.model import FashionClassifierModel, ModelParams
from machinelearningtemplate.train import DEVICE

app = typer.Typer()
DEFAULT_MODEL_CHECKPOINT = "models/model.pth"

# Get the relative path to the config directory
CONFIG_PATH = "../../configs"


@app.command()
def evaluate(model_checkpoint: str = DEFAULT_MODEL_CHECKPOINT, experiment: str = "default") -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depended on it")
    print(f"Model checkpoint: {model_checkpoint}")
    print(f"Using experiment config: {experiment}")

    # Load configuration with relative path
    with hydra.initialize(config_path=CONFIG_PATH, version_base=None):
        cfg = hydra.compose(config_name="default_config")

    # Create parameters from model experiments config
    params = ModelParams(
        num_filters1=cfg.model_experiments.params.num_filters1,
        num_filters2=cfg.model_experiments.params.num_filters2,
        num_filters3=cfg.model_experiments.params.num_filters3,
        dropout_rate=cfg.model_experiments.params.dropout_rate,
        num_fc_layers=cfg.model_experiments.params.num_fc_layers,
        ff_hidden_dim=cfg.model_experiments.params.ff_hidden_dim,
    )

    # Initialize model and load weights
    model = FashionClassifierModel(params).to(DEVICE)
    checkpoint = torch.load(model_checkpoint, map_location=DEVICE)
    model.load_state_dict(checkpoint)

    _, test_set = corrupt_mnist()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    model.eval()
    correct, total = 0, 0
    for img, target in test_dataloader:
        img, target = img.to(DEVICE), target.to(DEVICE)
        y_pred = model(img)
        correct += (y_pred.argmax(dim=1) == target).float().sum().item()
        total += target.size(0)
    print(f"Test accuracy: {correct / total}")


if __name__ == "__main__":
    app()
