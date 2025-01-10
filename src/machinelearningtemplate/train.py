import os
import torch
import matplotlib.pyplot as plt
from machinelearningtemplate.data import corrupt_mnist
from machinelearningtemplate.model import FashionClassifierModel, ModelParams

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def train() -> None:
    """Train a model on MNIST."""
    print("Training day and night")

    params = ModelParams(
        num_filters1=32,
        num_filters2=64,
        num_filters3=128,
        dropout_rate=0.5,
        num_fc_layers=2,
        ff_hidden_dim=256
    )

    model = FashionClassifierModel(params).to(DEVICE)
    train_set, _ = corrupt_mnist()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=32)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(10):
        model.train()
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())
            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

    print("Training complete")

    # Ensure the models directory exists
    os.makedirs("models", exist_ok=True)

    # Save the model state dictionary directly
    torch.save(model.state_dict(), "models/model.pth")

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("reports/figures/training_statistics.png")

if __name__ == "__main__":
    train()