import os
import time
import cv2
import pandas as pd
import torch
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from torchvision import tv_tensors
from torchvision.io import decode_image
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class CustomDataset(Dataset):
    def __init__(self, img_dir, annotation_file, transform=None):
        self.img_dir = img_dir
        self.labels_df = pd.read_csv(annotation_file)
        self.transform = transform
        self.image_ids = self.labels_df["image_id"].drop_duplicates().values

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, f"{self.image_ids[index]}.png")
        img = cv2.imread(img_path)
        x_min = self.labels_df["x_min"].values[index]
        x_max = self.labels_df["x_max"].values[index]
        y_min = self.labels_df["y_min"].values[index]
        y_max = self.labels_df["y_max"].values[index]
        bbox = [x_min, x_max, y_min, y_max]
        img = tv_tensors.Image(img)
        target = {"boxes": tv_tensors.BoundingBoxes(bbox, format="XYXY", canvas_size=(640, 640)),
                  "image_id": self.image_ids[index], "class_id": self.labels_df["class_id"].values[index],
                  "class_name": self.labels_df["class_name"].values[index]}

        if self.transform:
            img, target = self.transform(img, target)

        return img, target


def data_transform(img_size=(640, 640)):
    """
    Generates image transformation pipelines for training and evaluation.

    The training pipeline includes data augmentation techniques such as random
    horizontal/vertical flips and rotation, in addition to resizing and
    normalization. The evaluation pipeline only includes resizing and
    normalization.

    :param img_size: A tuple ``(height, width)`` specifying the target size for
                     resizing images. Defaults to ``(640, 640)``.
    :return: A tuple containing two torchvision.transforms.v2.Compose objects:
             - ``train_transforms``: The transformation pipeline for training data.
             - ``test_transforms``: The transformation pipeline for validation and
                                test data.
    :rtype: ``tuple[v2.Compose, v2.Compose]``
    """
    # Transform for training data (with data augmentation)
    train_transforms = v2.Compose([
        v2.Resize(size=img_size),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomRotation(degrees=30),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        v2.ToTensor(),
    ])

    # Transform for valid and test data (without data augmentation)
    test_transforms = v2.Compose([
        v2.Resize(size=img_size),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        v2.ToTensor(),
    ])

    return train_transforms, test_transforms


def load_data(train_folder, val_folder, test_folder,
              train_csv, val_csv, test_csv):
    """
    Loads image datasets and creates PyTorch DataLoader instances.

    This function first generates the necessary image transformations using
    ``data_transform``, then initializes ``CustomDataset`` objects for training,
    validation, and testing data, and finally wraps them in ``DataLoader``s
    for efficient batch processing.

    :param train_folder: Path to the directory containing training images.
    :param val_folder: Path to the directory containing validation images.
    :param test_folder: Path to the directory containing test images.
    :param train_csv: Path to the CSV file containing annotations for training data.
    :param val_csv: Path to the CSV file containing annotations for validation data.
    :param test_csv: Path to the CSV file containing annotations for test data.
    :return: A tuple containing three PyTorch DataLoader objects:
             - train_loader: DataLoader for the training dataset.
             - valid_loader: DataLoader for the validation dataset.
             - test_loader: DataLoader for the test dataset.
    :rtype: tuple[DataLoader, DataLoader, DataLoader]
    """
    train_transforms, test_transforms = data_transform()

    # Create dataset
    train_dataset = CustomDataset(img_dir=train_folder, annotation_file=train_csv, transform=train_transforms)
    valid_dataset = CustomDataset(img_dir=val_folder, annotation_file=val_csv, transform=test_transforms)
    test_dataset = CustomDataset(img_dir=test_folder, annotation_file=test_csv, transform=test_transforms)

    # Load dataset
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, valid_loader, test_loader


def train_model(epochs, model, device, optimizer, train_loader, valid_loader):
    valid_map50_history, valid_map50_95_history = [], []
    train_loss_history, valid_loss_history = [], []
    map_metrics = MeanAveragePrecision(iou_thresholds=[0.5, 0.95])

    print(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        start_time = time.perf_counter()

        # Train model on training set
        model.train()
        train_loss = 0.0

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items() if isinstance(v, torch.Tensor)} for t in targets]

            optimizer.zero_grad()  # Reset gradients to 0
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            loss.backward()  # Backpropagation
            optimizer.step()  # Update model weights
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # Evaluate model on validation set
        model.eval()
        map_metrics.reset()
        valid_loss = 0.0

        with torch.no_grad():
            for images, targets in valid_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items() if isinstance(v, torch.Tensor)} for t in targets]

                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
                output = model(images)
                map_metrics.update(output, targets)

                valid_loss += loss.item()

        avg_valid_loss = valid_loss / len(valid_loader)
        valid_loss_history.append(avg_valid_loss)

        # Calculate mAP metrics
        result = map_metrics.compute()
        map_score_50 = result["map_50"].item()
        map_score_50_95 = result["map"].item()
        valid_map50_history.append(map_score_50)
        valid_map50_95_history.append(map_score_50_95)
        end_time = time.perf_counter()

        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Train loss: {avg_train_loss:.4f} | "
              f"Validation loss: {avg_valid_loss:.4f} | "
              f"Validation mAP@0.5: {map_score_50:.4f}"
              f"Epoch time: {end_time - start_time:.4f}")

    print("Training complete.")
    return train_loss_history, valid_loss_history, valid_map50_history, valid_map50_95_history


def evaluate_model(model, device, test_loader):
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
    print(f"Test mAP@0.5: {test_map_50:.4f}")
    print(f"Test mAP@0.5-0.95: {test_map_50_95:.4f}")
