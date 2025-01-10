import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import typer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from train import DEVICE

from machinelearningtemplate.data import corrupt_mnist
from machinelearningtemplate.model import FashionClassifierModel


def visualize(model_checkpoint: str, figure_name: str = "embeddings.png") -> None:
    """Visualize model predictions."""
    model_path = Path("./models") / model_checkpoint
    model = FashionClassifierModel().to(DEVICE)
    model_path = Path("./models") / model_checkpoint
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.fc = torch.nn.Identity()

    _, test_dataset = corrupt_mnist()

    embeddings, targets = [], []
    with torch.inference_mode():
        for batch in torch.utils.data.DataLoader(test_dataset, batch_size=32):
            images, target = batch
            predictions = model(images)
            embeddings.append(predictions)
            targets.append(target)
        embeddings = torch.cat(embeddings).numpy()
        targets = torch.cat(targets).numpy()

    if embeddings.shape[1] > 500:  # Reduce dimensionality for large embeddings
        pca = PCA(n_components=100)
        embeddings = pca.fit_transform(embeddings)
    tsne = TSNE(n_components=2)
    embeddings = tsne.fit_transform(embeddings)

    if not os.path.exists("reports/figures"):
        os.makedirs("reports/figures")
    
    plt.figure(figsize=(10, 10))
    for i in range(10):
        mask = targets == i
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1], label=str(i))
    plt.legend()
    plt.savefig(f"reports/figures/{figure_name}")


if __name__ == "__main__":
    typer.run(visualize)