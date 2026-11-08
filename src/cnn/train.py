"""Training loop for the Dataset B ResNet18 CNN baseline.

The training procedure follows the original notebook:
- weighted cross-entropy loss (weights are supplied by the caller)
- AdamW optimizer (supplied by the caller)
- validation selection using group-level macro-F1
- best-checkpoint saving and early stopping

This module deliberately does not create the data split or calculate class
weights. Those must be prepared from the training partition only, as in the
original experiment.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import torch

from .evaluate import evaluate_cnn


def train_cnn(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device: torch.device,
    max_epochs: int,
    patience: int,
    checkpoint_path: str,
    history_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Train the CNN and save the checkpoint with the best validation group F1.

    Args:
        model: Initialized CNN model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        criterion: Loss function, normally weighted CrossEntropyLoss.
        optimizer: Optimizer, normally AdamW.
        device: CUDA or CPU device.
        max_epochs: Maximum number of epochs.
        patience: Early-stopping patience.
        checkpoint_path: Destination for the best model checkpoint.
        history_dir: Optional directory for ``training_history.csv``.

    Returns:
        A DataFrame containing per-epoch training and validation metrics.
    """
    checkpoint_file = Path(checkpoint_path)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    history_path = None
    if history_dir is not None:
        history_directory = Path(history_dir)
        history_directory.mkdir(parents=True, exist_ok=True)
        history_path = history_directory / "training_history.csv"

    history = []
    best_score = -1.0
    epochs_without_improvement = 0
    best_epoch = 0

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        if total == 0:
            raise RuntimeError("The training DataLoader produced no samples.")

        train_loss = running_loss / total
        train_accuracy = correct / total

        # Model selection is based on subject–muscle–side group macro-F1.
        val_results = evaluate_cnn(model, val_loader, device)
        val_group_f1 = val_results["group_macro_f1"]

        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_image_accuracy": val_results["image_accuracy"],
            "val_image_macro_f1": val_results["image_macro_f1"],
            "val_group_accuracy": val_results["group_accuracy"],
            "val_group_macro_f1": val_group_f1,
            "val_group_balanced_accuracy": (
                val_results["group_balanced_accuracy"]
            ),
        }
        history.append(record)

        print(
            f"Epoch {epoch + 1}/{max_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Group Macro-F1: {val_group_f1:.4f}"
        )

        if val_group_f1 > best_score:
            best_score = val_group_f1
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_group_macro_f1": best_score,
                },
                checkpoint_file,
            )
            print("  Best model checkpoint saved.")
        else:
            epochs_without_improvement += 1

        if history_path is not None:
            pd.DataFrame(history).to_csv(history_path, index=False)

        if epochs_without_improvement >= patience:
            print("\\nEarly stopping triggered.")
            break

    history_df = pd.DataFrame(history)

    print("\\nTraining function completed.")
    print("Best epoch:", best_epoch)
    print("Best validation group macro-F1:", best_score)

    return history_df
