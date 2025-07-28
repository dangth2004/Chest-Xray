import os
import cv2
import torch
import pandas as pd
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from torchvision import tv_tensors


class CustomDataset(Dataset):
    """
    Custom Dataset for loading images and their corresponding bounding box annotations.

    This dataset expects images to be grayscale PNGs and annotations to be provided
    in a CSV file. It handles reading images, parsing annotations, and applying
    transformations. If an image has no bounding boxes, it returns empty tensors
    for boxes and labels.

    :param img_dir: Path to the directory containing image files.
    :type img_dir: str
    :param annotation_file: Path to the CSV file containing annotation data ``(image_id, x_min, y_min, x_max, y_max, class_id, class_name)``.
    :type annotation_file: str
    :param transform: An optional callable transform to be applied to the image and target.
    :type transform: callable, optional
    """

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
    """
    Generates image transformation pipelines for training and evaluation of grayscale images.

    The training pipeline includes data augmentation techniques such as random
    horizontal/vertical flips and rotation, in addition to converting to image format,
    converting to ``float32``, converting to grayscale.
    The evaluation pipeline includes converting to image format, converting to ``float32``,
    converting to grayscale, without data augmentation.

    :return: A tuple containing two ``torchvision.transforms.v2.Compose`` objects:
             - ``train_transforms``: The transformation pipeline for training data.
             - ``test_transforms``: The transformation pipeline for validation and test data.
    :rtype: ``tuple[v2.Compose, v2.Compose]``
    """
    # Transform for training data (with data augmentation)
    train_transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Grayscale(),
        # DATA AUGMENTATION
        v2.RandomHorizontalFlip(p=0.5),
        v2.GaussianNoise(),
        v2.GaussianBlur(kernel_size=3)
    ])

    # Transform for valid and test data (without data augmentation)
    test_transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Grayscale()
    ])

    return train_transforms, test_transforms


def collate_fn(batch):
    """
    Collates a list of samples into a batch, typically used for custom DataLoader behavior.

    This function is designed to handle cases where individual samples within a batch
    might have different structures or sizes (e.g., varying number of bounding boxes
    per image in object detection). It transposes the batch, converting a list of
    ``(image, target)`` tuples into a tuple of ``(images, targets)``.

    :param batch: A list of individual samples, where each sample is typically a tuple
                  (e.g., ``(image, label)`` or ``(image, target_dict)``).
    :type batch: list
    :return: A tuple containing transposed elements of the batch. For example, if the
             input batch is ``[(img1, target1), (img2, target2)]``, the output will be
             ``((img1, img2), (target1, target2))``.
    :rtype: tuple
    """
    return tuple(zip(*batch))


def load_data(train_folder, val_folder, test_folder,
              train_csv, val_csv, test_csv, batch_size):
    """
    Loads image datasets and creates PyTorch DataLoader instances.

    This function first generates the necessary image transformations using
    ``data_transform``, then initializes ``CustomDataset`` objects for training,
    validation, and testing data. Finally, it wraps these datasets in ``DataLoader``s
    for efficient batch processing, utilizing a custom ``collate_fn`` for flexible
    batch handling.

    :param train_folder: Path to the directory containing training images.
    :type train_folder: str
    :param val_folder: Path to the directory containing validation images.
    :type val_folder: str
    :param test_folder: Path to the directory containing test images.
    :type test_folder: str
    :param train_csv: Path to the CSV file containing annotations for training data.
    :type train_csv: str
    :param val_csv: Path to the CSV file containing annotations for validation data.
    :type val_csv: str
    :param test_csv: Path to the CSV file containing annotations for test data.
    :type test_csv: str
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, valid_loader, test_loader
