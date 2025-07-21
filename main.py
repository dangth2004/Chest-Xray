import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydicom

from utils import read_dicom_image, draw_bounding_boxes


def main():
    data_path = 'data/train/9b44eddd5b59cd65d92366f27290d949.dicom'
    bbox = [[1486.0, 1091.0, 1896.0, 1439.0], [942.0, 749.0, 1232.0, 807.0], [1564.0, 1093.0, 1919.0, 1543.0],
            [1093.0, 1899.0, 2471.0, 2321.0], [1486.0, 1075.0, 1883.0, 1507.0]]
    img_raw = read_image(data_path)
    img_raw_box = draw_bounding_boxes(img_raw, bbox)
    img, adjusted_bbox = read_dicom_image(data_path, bounding_box=bbox)
    img_box = draw_bounding_boxes(img, adjusted_bbox)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0][0].imshow(img_raw, cmap=plt.cm.bone)
    axes[0][0].set_title('Original Image')
    axes[0][1].imshow(img_raw_box, cmap=plt.cm.bone)
    axes[0][1].set_title('Original Image Bounding Box')
    axes[1][0].imshow(img, cmap=plt.cm.bone)
    axes[1][0].set_title('Resized Image')
    axes[1][1].imshow(img_box, cmap=plt.cm.bone)
    axes[1][1].set_title('Resized Image Bounding Box')
    plt.tight_layout()
    plt.show()


def read_image(data_path):
    data = pydicom.dcmread(data_path)
    img = data.pixel_array

    # Convert to float for normalization to avoid clipping issues with large pixel values
    img = img.astype(np.float32)

    # Handle MONOCHROME1 photometric interpretation
    # Invert image if MONOCHROME1, so higher pixel values mean brighter regions
    if 'PhotometricInterpretation' in data and data.PhotometricInterpretation == 'MONOCHROME1':
        img = np.max(img) - img

    # Normalize the images to 0-255 and convert to uint8
    img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # Apply the CLAHE technique to enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe_enhanced = clahe.apply(img_normalized)

    return img_clahe_enhanced


if __name__ == '__main__':
    main()
