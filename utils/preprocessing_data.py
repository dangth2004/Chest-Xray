import pydicom
import cv2
import numpy as np


def print_dicom_data(data_path):
    """
    Reads a DICOM file and prints its full metadata (DICOM tags and their values) to the console.

    This function does not return any value; its purpose is to display the
    DICOM dataset information directly.

    :param data_path: Path to the DICOM file (e.g., ``path/to/image.dicom``).
    :type data_path: str
    :return: None. This function does not return any value.
    :rtype: None
    """
    data = pydicom.dcmread(data_path)
    print(data)


def read_dicom_image(data_path):
    """
    Reads chest X-ray images from DICOM files and applies a series of
    preprocessing techniques suitable for deep learning models. It also
    transforms any provided bounding box coordinates to match the processed image.

    Preprocessing steps include:

    -   **Pixel Array Extraction**: Extracts raw pixel data from the DICOM file.
    -   **Type Conversion**: Converts pixel data to ``float32`` to handle varying bit depths.
    -   **MONOCHROME1 Inversion**: Inverts pixel values for `MONOCHROME1`
        photometric interpretation to ensure higher values represent brighter regions.
    -   **Normalization**: Scales pixel values to the 0-255 range and converts to ``uint8``.
    -   **Bounding Box Adjustment**: Transforms bounding box coordinates (if provided)
        to correspond to the resized and padded image.
    -   **Contrast Enhancement (CLAHE)**: Applies Contrast Limited Adaptive Histogram Equalization
        to improve local contrast.

    :param data_path: Path to the DICOM file (e.g., ``path/to/image.dicom``).
    :return: A tuple containing:
             - **img_clahe_enhanced** (numpy.ndarray): The fully preprocessed image (uint8, 1-channel grayscale).
             - **adjusted_bounding_boxes** (list): A list of transformed bounding box coordinates.
               This will be an empty list if `bounding_box` was None or empty in the input.
    :rtype: tuple[numpy.ndarray, list]
    """
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


def resize_image_with_padding(img, bounding_boxes=None,
                              target_shape=(640, 640), pad_color=(0, 0, 0)):
    """
    Resizes an image to a target shape while maintaining its aspect ratio
    by adding padding. It also updates the coordinates of any provided
    bounding boxes to match the new image dimensions and padding.

    :param img: The input image (NumPy array). Can be grayscale (2D) or color (3D).
    :param bounding_boxes: An optional list of bounding box coordinates.
                           Each bounding box should be in the format ``[x_min, y_min, x_max, y_max]``.
                           If None, no bounding boxes will be processed. Defaults to None.
    :param target_shape: A tuple ``(width, height)`` representing the desired output image size.
                         Defaults to ``(640, 640)``.
    :param pad_color: A tuple ``(B, G, R)`` specifying the color of the padding pixels for color images.
                      For grayscale images, a single integer value (0-255) can be provided.
                      Defaults to black ``(0, 0, 0)``.
    :return: A tuple containing:
             - **padded_img** (numpy.ndarray): The resized and padded image,
               which will have the ``target_shape``.
             - **updated_bounding_boxes** (list): A list of transformed bounding box coordinates.
               If `bounding_boxes` was None or empty, this will be an empty list.
    :rtype: tuple[numpy.ndarray, list]
    """
    h_orig, w_orig = img.shape[:2]
    target_h, target_w = target_shape

    # Tỷ lệ co dãn
    scale_h, scale_w = target_h / h_orig, target_w / w_orig
    scale = min(scale_h, scale_w)

    # Kích thước mới sau khi co giãn
    new_h, new_w = int(h_orig * scale), int(w_orig * scale)
    resized_img = cv2.resize(img, (new_w, new_h))

    # Đệm
    pad_h, pad_w = target_h - new_h, target_w - new_w
    # Chia đều đệm cho cả hai bên/trên/dưới
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    padded_img = cv2.copyMakeBorder(resized_img,
                                    pad_top, pad_bottom,
                                    pad_left, pad_right,
                                    cv2.BORDER_CONSTANT,
                                    value=pad_color)

    # Điều chỉnh bounding box
    updated_bounding_boxes = []  # Initialize an empty list
    if bounding_boxes is not None:
        for box in bounding_boxes:
            x_min, y_min, x_max, y_max = box

            # Co giãn tọa độ
            new_x_min = int(x_min * scale)
            new_y_min = int(y_min * scale)
            new_x_max = int(x_max * scale)
            new_y_max = int(y_max * scale)

            # Dịch chuyển tọa độ do thêm đệm
            new_x_min += pad_left
            new_y_min += pad_top
            new_x_max += pad_left
            new_y_max += pad_top

            updated_bounding_boxes.append([new_x_min, new_y_min, new_x_max, new_y_max])

    return padded_img, updated_bounding_boxes


def draw_bounding_boxes(img, bounding_boxes):
    """
    Draws bounding boxes on a copy of the input image.

    This function iterates through a list of bounding box coordinates and
    draws a rectangle for each box on the image. The original image
    remains unchanged. If bounding boxes are None, the function return
    the original image.

    :param img: The input image (NumPy array). Can be grayscale (2D) or color (3D).
                For grayscale images, the drawn boxes will appear in the specified color
                if the image is converted to 3 channels by OpenCV for drawing,
                otherwise, they might be drawn using intensity values.
    :param bounding_boxes: A list of bounding box coordinates. Each bounding box
                           should be in the format ``[x_min, y_min, x_max, y_max]`` (integers or floats).
                           These coordinates are typically pixel values.
    :return: **img_box** (numpy.ndarray): A copy of the input image with the bounding boxes drawn on it.
                                          The output image will have the same dimensions and data type
                                          as the input image, but with drawn rectangles.
    :rtype: numpy.ndarray
    """
    img_box = img.copy()

    if bounding_boxes is None or bounding_boxes == []:
        return img_box

    for box in bounding_boxes:
        x_min, y_min, x_max, y_max = box
        cv2.rectangle(img_box, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=(255, 0, 0), thickness=5)

    return img_box
