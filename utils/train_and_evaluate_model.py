import os
import time
import numpy as np
import torch
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def train_detection_model(epochs, model, device, optimizer, train_loader,
                          valid_loader, result_file, model_save_path):
    map_metrics = MeanAveragePrecision(iou_thresholds=[0.5, 0.95])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=7)

    best_map = 0.0
    metrics_data = []

    os.makedirs("results", exist_ok=True)

    print(f"Training detection model for {epochs} epochs...")
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


def evaluate_detection_model(model, device, test_loader, results_file, model_path):
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


def train_classification_model(epochs, model, device, loss_func, optimizer, train_loader,
                               valid_loader, result_file, model_save_path):
    print(f"Training classification model for {epochs} epochs...")
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=7)
    best_val_acc = 0.0
    metrics_data = []
    os.makedirs("results", exist_ok=True)

    for epoch in range(epochs):
        start_time = time.perf_counter()

        # TRAIN MODEL ON TRAINING SET
        model.train()
        train_loss, train_corrects = 0.0, 0

        # Wrap train_loader with tqdm for a progress bar
        train_bar = tqdm(train_loader, desc=f"Training", unit="batch")

        for images, targets in train_bar:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()  # Reset gradient
            outputs = model(images)  # Predict output
            loss = loss_func(outputs, targets)
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_corrects += torch.sum(preds == targets)

            # Update the progress bar with the current loss
            train_bar.set_postfix(loss=loss.item())

        # Calculate train loss and train accuracy of epoch
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_corrects / len(train_loader.dataset)

        # EVALUATE MODEL ON VALIDATION SET
        model.eval()
        val_loss, val_corrects = 0.0, 0

        # Wrap valid_loader with tqdm for a progress bar
        valid_bar = tqdm(valid_loader, desc=f"Validation", unit="batch")

        with torch.no_grad():
            for images, targets in valid_bar:
                images, targets = images.to(device), targets.to(device)

                outputs = model(images)  # Predict output
                loss = loss_func(outputs, targets)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == targets)

                # Update the progress bar with the current loss
                valid_bar.set_postfix(loss=loss.item())

        # Calculate train loss and train accuracy of epoch
        val_loss = val_loss / len(valid_loader.dataset)
        val_acc = val_corrects / len(valid_loader.dataset)

        # Update learning rate
        scheduler.step(val_loss)
        end_time = time.perf_counter()

        # Save the best model state with the highest accuracy on validation set
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_val_acc
            }, model_save_path)
            print(f"--> Saved new best model with validation accuracy: {best_val_acc:.4f} in epoch: {epoch + 1}")

        # Print information of this epoch
        print(f"Epoch {epoch + 1}/{epochs} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
              f"Train loss: {train_loss:.4f} | "
              f"Train acc: {train_acc:.4f} | "
              f"Validation loss: {val_loss:.4f} | "
              f"Validation acc: {val_acc:.4f} | "
              f"Epoch time: {end_time - start_time:.2f}s")

        metrics_data.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc.item(),
            'val_loss': val_loss,
            'val_acc': val_acc.item(),
            'time': end_time - start_time,
            'learning_rate': optimizer.param_groups[0]['lr']
        })

    print("Training complete.")
    pd.DataFrame(metrics_data).to_csv(os.path.join("results", f"{result_file}.csv"), index=False)
    print(f"Training metrics saved to {result_file}.csv")


def evaluate_classification_model(model, device, test_loader, model_state):
    print(f"Evaluating classification model")
    checkpoint = torch.load(model_state)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Print accuracy, precision, recall and F1-score of model
    print(classification_report(all_labels, all_preds, target_names=['Has finding', 'No finding']))

    # Calculate roc_curve and roc_auc_score
    fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
    auc_score = roc_auc_score(all_labels, all_preds)

    print(f"AUC Score: {auc_score:.4f}")

    # Draw ROC Curve plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.grid()
    plt.show()
