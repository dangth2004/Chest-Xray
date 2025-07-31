import os
import time
import torch
import pandas as pd
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def train_model(epochs, model, device, optimizer, train_loader,
                valid_loader, result_file, model_save_path):
    """

    :param epochs:
    :param model:
    :param device:
    :param optimizer:
    :param train_loader:
    :param valid_loader:
    :param result_file:
    :param model_save_path:
    :return:
    """
    map_metrics = MeanAveragePrecision(iou_thresholds=[0.5, 0.95])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=7)

    best_map = 0.0
    metrics_data = []

    os.makedirs("results", exist_ok=True)

    print(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        start_time = time.perf_counter()

        # TRAIN MODEL ON TRAINING SET
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

        # EVALUATE MODEL ON VALIDATION SET
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
    Evaluates a trained model on a test dataset.

    This function loads a pre-trained model, sets it to evaluation mode,
    and calculates the mean average precision (mAP) metrics on the
    provided test dataset. The evaluation results are then printed
    and saved to a CSV file.

    :param model: The neural network model to be evaluated.
    :type model: torch.nn.Module
    :param device: The device (e.g., 'cuda' or 'cpu') on which to perform evaluation.
    :type device: torch.device
    :param test_loader: DataLoader providing test images and targets.
    :type test_loader: torch.utils.data.DataLoader
    :param results_file: The base name for the CSV file where evaluation results will be saved.
    :type results_file: str
    :param model_path: The file path to the saved model checkpoint.
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
