import os
import cv2
import torch
import pandas as pd
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from torchvision import tv_tensors


class DetectionDataset(Dataset):
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
        v2.Resize((800, 800)),
        # DATA AUGMENTATION
        v2.RandomHorizontalFlip(p=0.5),
        v2.GaussianNoise(),
        v2.GaussianBlur(kernel_size=3)
    ])

    # Transform for valid and test data (without data augmentation)
    test_transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((800, 800)),
        v2.Grayscale()
    ])

    return train_transforms, test_transforms


def collate_fn(batch):
    return tuple(zip(*batch))


def load_detection_data(train_folder, val_folder, test_folder,
                        train_csv, val_csv, test_csv, batch_size):
    # Create data transform
    train_transforms, test_transforms = data_transform()

    # Create dataset
    train_dataset = DetectionDataset(img_dir=train_folder, annotation_file=train_csv, transform=train_transforms)
    valid_dataset = DetectionDataset(img_dir=val_folder, annotation_file=val_csv, transform=test_transforms)
    test_dataset = DetectionDataset(img_dir=test_folder, annotation_file=test_csv, transform=test_transforms)

    # Load dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, valid_loader, test_loader


def load_classification_data(train_folder, val_folder, test_folder, batch_size):
    # Create data transform
    train_transforms, test_transforms = data_transform()

    # Create dataset
    train_dataset = ImageFolder(root=train_folder, transform=train_transforms)
    valid_dataset = ImageFolder(root=val_folder, transform=test_transforms)
    test_dataset = ImageFolder(root=test_folder, transform=test_transforms)

    # Load dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader
