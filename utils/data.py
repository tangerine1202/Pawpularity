import logging
from pathlib import Path

import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


def load_data(data_dir, check=False):
    """
    Load the training and testing data from CSV files

    Args:
        data_dir: Directory containing the data files

    Returns:
        train_df, test_df: Pandas DataFrames with data and image paths
    """
    data_path = Path(data_dir)

    # Load the CSV files
    train_df = pd.read_csv(data_path / 'train.csv')
    test_df = pd.read_csv(data_path / 'test.csv')

    # Add image paths
    train_df['image_path'] = train_df['Id'].apply(
        lambda x: str(data_path / 'train' / f'{x}.jpg')
    )
    test_df['image_path'] = test_df['Id'].apply(
        lambda x: str(data_path / 'test' / f'{x}.jpg')
    )

    if check:
        check_data_integrity(train_df, data_dir)
        check_data_integrity(test_df, data_dir)

    return train_df, test_df


def split_train_val(train_df, val_size=0.2):
    """
    Split training data into training and validation sets

    Args:
        train_df: Training DataFrame
        val_size: Size of validation set (default 0.2)

    Returns:
        train_subset, val_subset: Split DataFrames
    """
    # Get indices and shuffle them
    indices = np.random.choice(train_df.index, size=len(train_df), replace=False)

    # Calculate split point
    val_count = int(len(train_df) * val_size)
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]

    # Create the subsets
    train_subset = train_df.iloc[train_indices].reset_index(drop=True)
    val_subset = train_df.iloc[val_indices].reset_index(drop=True)

    return train_subset, val_subset


def check_data_integrity(df, base_dir='data'):
    """
    Check if all images exist in the specified directory

    Args:
        df: DataFrame containing image IDs
        base_dir: Base directory for images

    Returns:
        bool: True if all images exist, False otherwise
    """
    base_path = Path(base_dir)
    missing_images = []

    for idx, row in df.iterrows():
        if 'image_path' in df.columns:
            img_path = Path(row['image_path'])
        else:
            img_dir = 'train' if 'Pawpularity' in df.columns else 'test'
            img_path = base_path / img_dir / f"{row['Id']}.jpg"

        if not img_path.exists():
            missing_images.append(row['Id'])

    if missing_images:
        log.error(f"Missing {len(missing_images)} images: {missing_images[:5]}...")
        return False

    return True


class CustomDataset(Dataset):
    def __init__(self, df, transform=None, has_label=True):
        """
        Args:
            df: DataFrame containing image paths and labels
            transform: Optional transform to be applied on a sample
        """
        self.df = df
        self.transform = transform
        self.has_label = has_label

        self.img_size = (224, 224)
        log.warning(
            f"Image size is hardcoded to {self.img_size}. Consider making it configurable."
        )

        self.label_col = 'Pawpularity'
        self.img_path_col = 'image_path'
        self.metadata_cols = [col for col in self.df.columns if col not in ['Id', self.label_col, self.img_path_col]]

        if self.has_label and self.label_col not in df.columns:
            raise ValueError(f"Label column '{self.label_col}' not found in DataFrame {df.columns}")
        if self.img_path_col not in df.columns:
            raise ValueError(f"Image path column '{self.img_path_col}' not found in DataFrame {df.columns}")

        self.img_paths = self.df[self.img_path_col].values
        self.metadata = self.df[self.metadata_cols].values.astype(np.float32)

        if self.has_label:
            self.labels = self.df[self.label_col].values[..., None].astype(np.float32)
            labels_normalized = self.normalize_labels(self.labels)
            labels_denormalized = self.denormalize_labels(labels_normalized)
            assert np.allclose(self.labels, labels_denormalized), "Normalization and denormalization mismatch"
            self.labels = labels_normalized
        else:
            self.labels = None

    @staticmethod
    def normalize_labels(labels):
        return labels / 100

    @staticmethod
    def denormalize_labels(labels):
        return labels * 100

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        data = self.metadata[idx]
        label = self.labels[idx] if self.has_label else None

        image = self.load_image(img_path)

        if self.transform:
            image = self.transform(image)

        if self.has_label:
            return image, data, label
        else:
            return image, data

    def load_image(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            log.error(f"Failed to load image at {img_path}")
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = cv2.resize(img, self.img_size)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # change to channel-first
        img = np.transpose(img, (2, 0, 1))
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        return img
