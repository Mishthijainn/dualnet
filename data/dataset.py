# ================================
# 1. DATASET CLASS
# ================================
import random
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from data.preprocessing import preprocess_fnirs_data


class fNIRSDataset(Dataset):
    """
    Dataset class for fNIRS data
    """

    def __init__(self, data, labels, task_type, transform=None):
        """
        Args:
            data: fNIRS signals (samples × channels × time_points)
            labels: Binary labels (0 or 1)
            task_type: 'pain' or 'stress'
            transform: Optional data augmentation
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.FloatTensor(labels).view(-1)
        self.task_type = task_type
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return {
            'signal': x,
            'label': y,
            'task': self.task_type
        }


class fNIRSDataAugmentation:
    """
    Data augmentation for fNIRS signals
    """

    def __init__(self, noise_level=0.05, shift_max=10, scaling_factor_range=(0.8, 1.2)):
        self.noise_level = noise_level
        self.shift_max = shift_max
        self.scaling_factor_range = scaling_factor_range

    def __call__(self, x):
        """
        Apply random transformations to fNIRS signal

        Args:
            x: fNIRS signal of shape (channels, time_points)

        Returns:
            Augmented signal
        """
        # Make a copy to avoid modifying the original
        x_aug = x.clone()

        # Add random noise
        if random.random() > 0.5:
            noise = torch.randn_like(x_aug) * self.noise_level
            x_aug = x_aug + noise

        # Random scaling
        if random.random() > 0.5:
            scaling_factor = random.uniform(*self.scaling_factor_range)
            x_aug = x_aug * scaling_factor

        # Random temporal shift (circular)
        if random.random() > 0.5:
            shift = random.randint(-self.shift_max, self.shift_max)
            if shift != 0:
                x_aug = torch.roll(x_aug, shifts=shift, dims=1)

        return x_aug


def load_and_preprocess_data(pain_data_path, stress_data_path, test_size=0.2, val_size=0.1):
    """
    Load and preprocess pain and stress fNIRS datasets

    Args:
        pain_data_path: Path to pain dataset
        stress_data_path: Path to stress dataset
        test_size: Proportion of data for testing
        val_size: Proportion of training data for validation

    Returns:
        DataLoaders for training, validation, and testing
    """
    # Load pain dataset
    pain_data = np.load(pain_data_path, allow_pickle=True).item()
    # Shape: (samples, channels, time_points)
    pain_signals = pain_data['signals']
    pain_labels = pain_data['labels']    # Shape: (samples,)

    # Load stress dataset
    stress_data = np.load(stress_data_path, allow_pickle=True).item()
    # Shape: (samples, channels, time_points)
    stress_signals = stress_data['signals']
    stress_labels = stress_data['labels']    # Shape: (samples,)

    # Preprocess signals
    pain_signals_processed = np.zeros_like(pain_signals)
    for i in range(pain_signals.shape[0]):
        pain_signals_processed[i] = preprocess_fnirs_data(pain_signals[i])

    stress_signals_processed = np.zeros_like(stress_signals)
    for i in range(stress_signals.shape[0]):
        stress_signals_processed[i] = preprocess_fnirs_data(stress_signals[i])

    # Split data into train, validation, and test sets
    pain_train_val_x, pain_test_x, pain_train_val_y, pain_test_y = train_test_split(
        pain_signals_processed, pain_labels, test_size=test_size, random_state=42, stratify=pain_labels
    )

    stress_train_val_x, stress_test_x, stress_train_val_y, stress_test_y = train_test_split(
        stress_signals_processed, stress_labels, test_size=test_size, random_state=42, stratify=stress_labels
    )

    # Further split training data into training and validation
    val_ratio = val_size / (1 - test_size)

    pain_train_x, pain_val_x, pain_train_y, pain_val_y = train_test_split(
        pain_train_val_x, pain_train_val_y, test_size=val_ratio, random_state=42, stratify=pain_train_val_y
    )

    stress_train_x, stress_val_x, stress_train_y, stress_val_y = train_test_split(
        stress_train_val_x, stress_train_val_y, test_size=val_ratio, random_state=42, stratify=stress_train_val_y
    )

    # Create datasets
    transform = fNIRSDataAugmentation()

    pain_train_dataset = fNIRSDataset(
        pain_train_x, pain_train_y, 'pain', transform=transform)
    pain_val_dataset = fNIRSDataset(pain_val_x, pain_val_y, 'pain')
    pain_test_dataset = fNIRSDataset(pain_test_x, pain_test_y, 'pain')

    stress_train_dataset = fNIRSDataset(
        stress_train_x, stress_train_y, 'stress', transform=transform)
    stress_val_dataset = fNIRSDataset(stress_val_x, stress_val_y, 'stress')
    stress_test_dataset = fNIRSDataset(stress_test_x, stress_test_y, 'stress')

    # Create data loaders
    batch_size = 32

    pain_train_loader = DataLoader(
        pain_train_dataset, batch_size=batch_size, shuffle=True)
    pain_val_loader = DataLoader(
        pain_val_dataset, batch_size=batch_size, shuffle=False)
    pain_test_loader = DataLoader(
        pain_test_dataset, batch_size=batch_size, shuffle=False)

    stress_train_loader = DataLoader(
        stress_train_dataset, batch_size=batch_size, shuffle=True)
    stress_val_loader = DataLoader(
        stress_val_dataset, batch_size=batch_size, shuffle=False)
    stress_test_loader = DataLoader(
        stress_test_dataset, batch_size=batch_size, shuffle=False)

    return {
        'pain': {
            'train': pain_train_loader,
            'val': pain_val_loader,
            'test': pain_test_loader
        },
        'stress': {
            'train': stress_train_loader,
            'val': stress_val_loader,
            'test': stress_test_loader
        }
    }
