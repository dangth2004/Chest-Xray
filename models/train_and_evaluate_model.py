import os
import time
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def _train_one_epoch(model, optimizer, train_loader, device):
    """
    Performs a single training epoch for the given model.

    Iterates through the ``train_loader``, moves data to the specified device,
    performs a forward pass, calculates the ``loss``, and updates ``model weights``
    using the ``optimizer``. Displays a progress bar with the current ``loss``.

    :param model: The PyTorch model to train.
    :type model: torch.nn.Module
    :param optimizer: The optimizer used for updating model parameters.
    :type optimizer: torch.optim.Optimizer
    :param train_loader: DataLoader providing batches of training data.
    :type train_loader: torch.utils.data.DataLoader
    :param device: The device (e.g., 'cuda' or 'cpu') to which data should be moved.
    :type device: torch.device or str
    :return: The average training loss for the epoch.
    :rtype: float
    """
    model.train()
    train_loss = 0.0

    # Wrap train_loader with tqdm for a progress bar
    train_bar = tqdm(train_loader, desc=f"Training", unit="batch")

    for images, targets in train_bar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items() if isinstance(v, torch.Tensor)} for t in targets]

        optimizer.zero_grad()  # Reset gradient

        # Calculate loss
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        loss.backward()  # Backpropagation
        optimizer.step()  # Update model parameters
        train_loss += loss.item()

        # Update the progress bar with the current loss
        train_bar.set_postfix(loss=loss.item())

    avg_train_loss = train_loss / len(train_loader)
    return avg_train_loss


def _validate_one_epoch(model, valid_loader, device, map_metrics):
    """
    Performs a single validation epoch for the given model.

    Evaluates the model on the `valid_loader` without gradient computation,
    calculates and updates mAP metrics. Displays a progress bar.

    :param model: The PyTorch model to validate.
    :type model: torch.nn.Module
    :param valid_loader: DataLoader providing batches of validation data.
    :type valid_loader: torch.utils.data.DataLoader
    :param device: The device (e.g., 'cuda' or 'cpu') to which data should be moved.
    :type device: torch.device or str
    :param map_metrics: An object (e.g., from ``torchmetrics``) to compute mAP scores.
                        It should have ``reset()``, ``update()``, and ``compute()`` methods.
    :type map_metrics: object
    :return: A tuple containing:
             - map_score_50: The mAP score at IoU threshold 0.50.
             - map_score_50_95: The mAP score averaged over IoU thresholds from 0.50 to 0.95.
    :rtype: tuple[float, float, float]
    """
    model.eval()
    map_metrics.reset()

    # Wrap valid_loader with tqdm
    valid_bar = tqdm(valid_loader, desc=f"Validation", unit="batch")

    with torch.no_grad():
        for images, targets in valid_bar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items() if isinstance(v, torch.Tensor)} for t in targets]

            # Get model output for metric computation
            output = model(images)
            map_metrics.update(output, targets)

            # Update the progress bar with the current validation loss
            valid_bar.set_postfix(status="Processing")

    result = map_metrics.compute()
    map_score_50 = result["map_50"].item()
    map_score_50_95 = result["map"].item()

    return map_score_50, map_score_50_95


def train_model(epochs, model, device, optimizer, train_loader,
                valid_loader, result_file, model_save_path):
    """
    Trains a PyTorch model over a specified number of epochs.

    This function orchestrates the training process, including:

    - Iterating through training and validation epochs.
    - Calculating and logging training and validation losses.
    - Computing Mean Average Precision (mAP) metrics on the validation set.
    - Adjusting the learning rate using a ReduceLROnPlateau scheduler based on validation loss.
    - Saving the model's state dictionary if a new best mAP@0.5-0.95 score is achieved.
    - Logging epoch-wise metrics to the console and saving them to a CSV file.

    :param epochs: The total number of training epochs.
    :type epochs: int
    :param model: The PyTorch model to be trained.
    :type model: torch.nn.Module
    :param device: The device (e.g., 'cuda' or 'cpu') on which the model and data will operate.
    :type device: torch.device or str
    :param optimizer: The optimizer used for updating model parameters.
    :type optimizer: torch.optim.Optimizer
    :param train_loader: DataLoader providing batches of training data.
    :type train_loader: torch.utils.data.DataLoader
    :param valid_loader:  providing batches of validation data.
    :type valid_loader: torch.utils.data.DataLoader
    :param result_file: The base name for the CSV file where training metrics will be saved (e.g., 'training_log').
                        The file will be saved in a 'results' directory.
    :type result_file: str
    :param model_save_path: The full path including filename where the best model's state dictionary will be saved.
    :type model_save_path: str
    :return: None
    :rtype: None
    """
    map_metrics = MeanAveragePrecision(iou_thresholds=[0.5, 0.95])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=7)

    best_map = 0.0
    metrics_data = []

    os.makedirs("results", exist_ok=True)

    print(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        start_time = time.perf_counter()

        # Train model on training set
        avg_train_loss = _train_one_epoch(model, optimizer, train_loader, device)

        # Evaluate model on validation set
        map_score_50, map_score_50_95 = _validate_one_epoch(model, valid_loader, device, map_metrics)

        # Update learning rate
        scheduler.step(map_score_50_95)
        end_time = time.perf_counter()

        metrics_data.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'valid_map50': map_score_50,
            'valid_map50_95': map_score_50_95,
            'time': end_time - start_time
        })

        print(f"Epoch {epoch + 1}/{epochs} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
              f"Train loss: {avg_train_loss:.4f} | "
              f"Validation mAP@0.5: {map_score_50:.4f} | "
              f"Validation mAP@0.5-0.95: {map_score_50_95:.4f} | "
              f"Epoch time: {end_time - start_time:.2f}s")

        # Save best model state
        if map_score_50_95 > best_map:
            best_map = map_score_50_95
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_map': best_map,
            }, model_save_path)
            print(f"--> Saved new best model with mAP@0.5-0.95: {best_map:.4f} in epoch: {epoch + 1}")

    print("Training complete.")
    df = pd.DataFrame(metrics_data)
    df.to_csv(os.path.join("results", f"{result_file}.csv"), index=False)
    print(f"Training metrics saved to {result_file}.csv")


def evaluate_model(model, device, test_loader, results_file, model_path):
    """
    Evaluates a trained PyTorch model on a test dataset.

    This function loads a pre-trained model from a specified path, sets it to evaluation mode,
    and computes Mean Average Precision (mAP) metrics on the provided test data. The results
    (mAP@0.5 and mAP@0.5-0.95) are printed to the console and saved to a CSV file.

    :param model: The PyTorch model to be evaluated. Its state dictionary will be loaded.
    :type model: torch.nn.Module
    :param device: The device (e.g., 'cuda' or 'cpu') on which the model and data will operate.
    :type device: torch.device or str
    :param test_loader: DataLoader providing batches of test data.
    :type test_loader: torch.utils.data.DataLoader
    :param results_file: The base name for the CSV file where evaluation results will be saved
                         (e.g., 'test_results'). The file will be saved in a 'results' directory.
    :type results_file: str
    :param model_path: The full path to the saved model's state dictionary (.pth file).
    :type model_path: str
    :return: None
    :rtype: None
    """
    print(f"\nLoading best model from {model_path} for evaluation...")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    map_metric = MeanAveragePrecision(iou_thresholds=[0.5, 0.95])
    map_metric.reset()

    # Evaluate model on test set
    with torch.no_grad():
        for images, targets in test_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items() if isinstance(v, torch.Tensor)} for t in targets]
            outputs = model(images)
            map_metric.update(outputs, targets)

    # Calculate mAP metrics
    result = map_metric.compute()
    test_map_50 = result["map_50"].item()
    test_map_50_95 = result["map"].item()

    # Print result
    print(f"Test mAP@0.5: {test_map_50:.4f}")
    print(f"Test mAP@0.5-0.95: {test_map_50_95:.4f}")

    # Save result file
    evaluation_results = {
        'test_map_50': test_map_50,
        'test_map_50_95': test_map_50_95
    }

    df = pd.DataFrame([evaluation_results])
    df.to_csv(os.path.join("results", f"{results_file}.csv"), index=False)
    print(f"Evaluation results saved to {results_file}.csv")
