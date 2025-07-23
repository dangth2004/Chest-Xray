import os
import time
import cv2
import pandas as pd
import torch
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from torchvision import tv_tensors
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
        image_id = self.image_ids[index]  # Lấy image_id hiện tại
        img_path = os.path.join(self.img_dir, f"{image_id}.png")  # Sử dụng image_id đúng

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Xử lý trường hợp không đọc được ảnh (ví dụ: ảnh không tồn tại hoặc bị lỗi)
            print(f"Warning: Could not read image {img_path}. Skipping or handling error.")
            # Bạn có thể return một giá trị mặc định hoặc raise an error tùy vào yêu cầu
            # For simplicity, let's return None for now, and handle in DataLoader
            return None, None  # Hoặc bạn có thể bỏ qua và xử lý ở DataLoader nếu batch_sampler cho phép

        # Lọc DataFrame để lấy tất cả các bản ghi (bounding box) cho image_id này
        image_annotations = self.labels_df[self.labels_df["image_id"] == image_id]

        # Lấy bounding box, class_id, class_name từ các bản ghi đã lọc
        # Cần chuyển đổi thành list of lists hoặc tensor nếu có nhiều bbox
        boxes = image_annotations[["x_min", "y_min", "x_max", "y_max"]].values.tolist()
        class_ids = image_annotations["class_id"].values.tolist()
        class_names = image_annotations["class_name"].values.tolist()

        img = tv_tensors.Image(img)  # Đảm bảo ảnh là tensor của torchvision
        # Xử lý trường hợp ảnh không có bounding box nào
        if not boxes:
            # Tạo tensor rỗng nếu không có bounding box nào
            bbox_tensor = tv_tensors.BoundingBoxes([], format="XYXY",
                                                   canvas_size=img.shape[-2:])  # Sử dụng kích thước ảnh thực tế
            class_id_tensor = torch.tensor([], dtype=torch.int64)
        else:
            bbox_tensor = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=img.shape[-2:])
            class_id_tensor = torch.tensor(class_ids, dtype=torch.int64)  # Đảm bảo dtype là int64 cho class_id

        target = {
            "boxes": bbox_tensor,
            "image_id": image_id,
            "class_name": class_names,
            "labels": class_id_tensor  # Thường dùng "labels" thay vì "class_id" cho nhất quán
        }
        # "class_name" có thể không cần thiết trong target nếu bạn chỉ dùng class_id để huấn luyện

        if self.transform:
            img, target = self.transform(img, target)

        return img, target


def data_transform():
    # Transform for training data (with data augmentation)
    train_transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Grayscale(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomRotation(degrees=30),
        v2.Normalize(mean=[0.456], std=[0.224])
    ])

    # Transform for valid and test data (without data augmentation)
    test_transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Grayscale(),
        v2.Normalize(mean=[0.456], std=[0.224])
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
