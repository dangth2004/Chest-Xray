import os
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import read_dicom_image


def save_images_and_csv(path_to_dicom_folder, destination_dir, df_full_labels, image_id_list):
    """
    Saves DICOM images as PNGs and creates an annotation CSV for a given list of image IDs.

    This function takes a list of image IDs, locates their corresponding DICOM files,
    reads them, converts them to PNG format, and saves them in a specified destination directory.
    It also filters the full DataFrame of labels to include only the annotations for the
    provided image IDs and saves this subset as an ``annotation.csv`` file in the same
    destination directory.

    :param path_to_dicom_folder: Path to the directory containing the DICOM image files.
                                 Each DICOM file is expected to be named as ``{image_id}.dicom``.
    :param destination_dir: The directory where the ``image`` folder (for PNGs) and
                            ``annotation.csv`` will be saved.
    :param df_full_labels: A pandas DataFrame containing all annotations. This DataFrame
                           is expected to have an ``image_id`` column.
    :param image_id_list: A list of image IDs (strings or integers) to be processed.
                          Only images with IDs in this list will be saved, and only
                          their annotations will be included in the ``annotation.csv``.
    """
    img_folder = os.path.join(destination_dir, 'image')
    os.makedirs(img_folder, exist_ok=True)

    # Split annotation files
    split_df = df_full_labels.loc[df_full_labels['image_id'].isin(image_id_list)].copy()
    split_csv_path = os.path.join(destination_dir, 'annotation.csv')
    split_df.to_csv(split_csv_path, index=False)

    # Split images into train - valid - test folder
    for img_id in image_id_list:
        dicom_path = os.path.join(path_to_dicom_folder, f"{img_id}.dicom")
        img_path = os.path.join(img_folder, f"{img_id}.png")
        cv2.imwrite(img_path, read_dicom_image(dicom_path))


def split_data(path_to_dicom_folder, path_to_csv,
               output_folder='dataset', train_size=0.7,
               val_size=0.15, test_size=0.15):
    """
    Reads DICOM image data and their corresponding annotations, then splits them into
    training, validation, and test datasets. The split datasets are saved into
    designated subfolders within the ``output_folder``, with DICOM images converted to PNGs
    and annotations saved as CSV files.

    This function performs the following steps:

    - Creates ``train``, ``valid``, and ``test`` subdirectories within the specified ``output_folder``.
    - Reads the full dataset from the provided CSV file.
    - Identifies unique image IDs and verifies the existence of their corresponding DICOM files.
    - Splits the verified image IDs into training, validation, and test sets based on the
       specified ``train_size``, ``val_size``, and ``test_size`` proportions.
    - Calls ``save_images_and_csv`` for each split to process and save the images and their
       respective annotations into their designated subfolders.

    :param path_to_dicom_folder: Path to the directory containing all original DICOM image files.
    :param path_to_csv: Path to the CSV file containing the full dataset annotations. This CSV
                        must have an ``image_id`` column.
    :param output_folder: The root directory where the ``train``, ``valid``, and ``test``
                          subfolders will be created. Defaults to ``dataset``.
    :param train_size: The proportion of the dataset to be used for training. Defaults to 0.7.
    :param val_size: The proportion of the dataset to be used for validation. Defaults to 0.15.
                     Note: The actual split for validation and test is performed on the
                     remaining data after the training split, using a 50/50 split of the remainder
                     if ``val_size`` and ``test_size`` are equal.
    :param test_size: The proportion of the dataset to be used for testing. Defaults to 0.15.
                      Note: The actual split for validation and test is performed on the
                      remaining data after the training split, using a 50/50 split of the remainder
                      if ``val_size`` and ``test_size`` are equal.
    """
    # Create train, valid and test folder
    train_folder = os.path.join(output_folder, 'train')
    val_folder = os.path.join(output_folder, 'valid')
    test_folder = os.path.join(output_folder, 'test')

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Split data
    print("Splitting data...")
    dataset = pd.read_csv(path_to_csv)
    unique_img_id = dataset['image_id'].unique().tolist()

    existing_id = []
    for img_id in unique_img_id:
        dicom_path = os.path.join(path_to_dicom_folder, f"{img_id}.dicom")
        if os.path.exists(dicom_path):
            existing_id.append(img_id)
        else:
            print(f"Dicom file {dicom_path} does not exist.")

    train_ids, temp_ids = train_test_split(existing_id, train_size=train_size, random_state=42)
    valid_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

    print(f"Found {len(existing_id)} images in {path_to_dicom_folder} folder.")
    print(f"Number of training images: {len(train_ids)}")
    print(f"Number of validation images: {len(valid_ids)}")
    print(f"Number of test images: {len(test_ids)}")

    # Read image from DICOM files and save as PNG
    print("Processing training folder...")
    save_images_and_csv(path_to_dicom_folder=path_to_dicom_folder, destination_dir=train_folder,
                        df_full_labels=dataset, image_id_list=train_ids)

    print("Processing validation folder...")
    save_images_and_csv(path_to_dicom_folder=path_to_dicom_folder, destination_dir=val_folder,
                        df_full_labels=dataset, image_id_list=valid_ids)

    print("Processing test folder...")
    save_images_and_csv(path_to_dicom_folder=path_to_dicom_folder, destination_dir=test_folder,
                        df_full_labels=dataset, image_id_list=test_ids)
