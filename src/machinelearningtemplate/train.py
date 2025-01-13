import os

import matplotlib.pyplot as plt
import torch

import wandb
from machinelearningtemplate.data import corrupt_mnist
from machinelearningtemplate.model import FashionClassifierModel, ModelParams

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def train() -> None:
    """Train a model on MNIST."""
    print("Training day and night")

    params = ModelParams(
        num_filters1=32, num_filters2=64, num_filters3=128, dropout_rate=0.5, num_fc_layers=2, ff_hidden_dim=256
    )

    # Ensure wandb.init() is properly scoped
    with wandb.init(
        project="corrupt_mnist",
        config={"params": params},
    ) as run:  # Use 'run' to emphasize the context
        model = FashionClassifierModel(params).to(DEVICE)
        train_set, _ = corrupt_mnist()

        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=32)

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        statistics = {"train_loss": [], "train_accuracy": []}
        for epoch in range(10):
            model.train()
            correct, total = 0, 0
            for i, (img, target) in enumerate(train_dataloader):
                img, target = img.to(DEVICE), target.to(DEVICE)
                optimizer.zero_grad()
                y_pred = model(img)
                loss = loss_fn(y_pred, target)
                loss.backward()
                optimizer.step()
                statistics["train_loss"].append(loss.item())

                # Calculate accuracy
                correct += (y_pred.argmax(dim=1) == target).float().sum().item()
                total += target.size(0)
                accuracy = correct / total
                statistics["train_accuracy"].append(accuracy)

                if i % 100 == 0:
                    print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}, accuracy: {accuracy}")

                # Log gradients to wandb
                run.log({"gradients": wandb.Histogram(model.conv1.weight.grad.cpu().detach().numpy())})

        print("Training complete")

        # Ensure the models directory exists
        os.makedirs("models", exist_ok=True)

        # Save the model state dictionary directly
        model_path = "models/model.pth"
        torch.save(model.state_dict(), model_path)

        # Log the model as an artifact within the context
        artifact = wandb.Artifact("fashion_classifier_model", type="model")
        artifact.add_file(model_path)
        run.log_artifact(artifact)  # Use 'run' to ensure it's within context

        # Log a sample image from the dataset
        sample_img, _ = next(iter(train_dataloader))
        run.log({"sample_image": wandb.Image(sample_img[0].cpu())})

        # Log a histogram of the model's weights
        run.log({"conv1_weights": wandb.Histogram(model.conv1.weight.cpu().detach().numpy())})

        # Log training loss and accuracy as a matplotlib figure
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        axs[0].plot(statistics["train_loss"])
        axs[0].set_title("Train loss")
        axs[1].plot(statistics["train_accuracy"])
        axs[1].set_title("Train accuracy")
        fig.savefig("reports/figures/training_statistics.png")
        wandb.log({"training_statistics": wandb.Image(fig)})


if __name__ == "__main__":
    train()
