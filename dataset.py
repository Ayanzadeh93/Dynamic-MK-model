"""
Dataset loader for medical image binary classification
Applies general augmentations suitable for medical imaging
"""

import os
import platform
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from typing import Tuple, List
import torch


class MedicalImageDataset(Dataset):
    """
    Custom Dataset for loading medical images from directory structure:
    
    New structure (with Train/Test split):
        dataset2/
            Train/
                0/  (class 0 training images)
                1/  (class 1 training images)
            Test/
                0/  (class 0 test images)
                1/  (class 1 test images)
    
    Old structure (backward compatibility):
        dataset/
            0/  (class 0 images)
            1/  (class 1 images)
    """
    
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        """
        Args:
            image_paths: List of paths to images
            labels: List of corresponding labels
            transform: Optional transform to be applied on images
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def get_image_paths_and_labels(data_dir: str, subset: str = None) -> Tuple[List[str], List[int]]:
    """
    Get all image paths and their corresponding labels from dataset directory
    
    Args:
        data_dir: Root directory containing Train/Test subdirectories
        subset: Optional subset name ('Train' or 'Test'). If None, assumes data_dir 
                contains class subdirectories directly (backward compatibility)
        
    Returns:
        Tuple of (image_paths, labels)
    """
    image_paths = []
    labels = []
    
    # If subset is specified, use the new structure: dataset2/Train or dataset2/Test
    if subset:
        data_dir = os.path.join(data_dir, subset)
    
    # Class 0
    class0_dir = os.path.join(data_dir, '0')
    if os.path.exists(class0_dir):
        for img_name in os.listdir(class0_dir):
            if img_name.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(class0_dir, img_name))
                labels.append(0)
    
    # Class 1
    class1_dir = os.path.join(data_dir, '1')
    if os.path.exists(class1_dir):
        for img_name in os.listdir(class1_dir):
            if img_name.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(class1_dir, img_name))
                labels.append(1)
    
    return image_paths, labels


def load_train_test_datasets(data_dir: str) -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    Load train and test datasets from separate directories
    
    Dataset structure:
        data_dir/
            Train/
                0/  (class 0 training images)
                1/  (class 1 training images)
            Test/
                0/  (class 0 test images)
                1/  (class 1 test images)
    
    Args:
        data_dir: Root directory containing Train and Test subdirectories
        
    Returns:
        Tuple of (train_paths, train_labels, test_paths, test_labels)
    """
    train_paths, train_labels = get_image_paths_and_labels(data_dir, subset='Train')
    test_paths, test_labels = get_image_paths_and_labels(data_dir, subset='Test')
    
    return train_paths, train_labels, test_paths, test_labels


def get_train_transforms(image_size: int = 224):
    """
    Get training data augmentation transforms
    Enhanced augmentations matching high-performance training:
    - Resize to target size
    - Random horizontal flip
    - Random rotation (10 degrees)
    - Color jitter (brightness and contrast)
    - Normalization (ImageNet statistics)
    
    Args:
        image_size: Target image size
        
    Returns:
        Composed transforms
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])


def get_val_transforms(image_size: int = 224):
    """
    Get validation/test data transforms (no augmentation)
    
    Args:
        image_size: Target image size
        
    Returns:
        Composed transforms
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])


def create_dataloaders(train_paths: List[str], train_labels: List[int],
                       val_paths: List[str], val_labels: List[int],
                       batch_size: int = 16, num_workers: int = 0,
                       image_size: int = 224) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders
    
    Args:
        train_paths: List of training image paths
        train_labels: List of training labels
        val_paths: List of validation image paths
        val_labels: List of validation labels
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading (0 for Windows compatibility)
        image_size: Target image size
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Windows compatibility: num_workers=0 to avoid multiprocessing issues
    if platform.system() == 'Windows':
        num_workers = 0
    
    train_dataset = MedicalImageDataset(
        train_paths, 
        train_labels, 
        transform=get_train_transforms(image_size)
    )
    
    val_dataset = MedicalImageDataset(
        val_paths, 
        val_labels, 
        transform=get_val_transforms(image_size)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader


def get_class_weights(labels: List[int], device: torch.device) -> torch.Tensor:
    """
    Calculate class weights for handling class imbalance
    
    Args:
        labels: List of all labels
        device: Device to place weights tensor on
        
    Returns:
        Tensor of class weights
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    # Calculate weights inversely proportional to class frequencies
    weights = torch.tensor([total / (len(unique) * count) for count in counts], 
                          dtype=torch.float32).to(device)
    
    return weights


def print_dataset_info(train_labels: List[int], val_labels: List[int], fold: int = None):
    """
    Print dataset statistics
    
    Args:
        train_labels: Training labels
        val_labels: Validation/Test labels
        fold: Current fold number (optional, for backward compatibility with k-fold)
    """
    fold_str = f" - Fold {fold}" if fold is not None else ""
    
    train_unique, train_counts = np.unique(train_labels, return_counts=True)
    val_unique, val_counts = np.unique(val_labels, return_counts=True)
    
    print(f"\n{'='*60}")
    print(f"Dataset Statistics{fold_str}")
    print(f"{'='*60}")
    print(f"\nTraining Set:")
    for label, count in zip(train_unique, train_counts):
        print(f"  Class {label}: {count} images ({count/len(train_labels)*100:.1f}%)")
    print(f"  Total: {len(train_labels)} images")
    
    print(f"\nTest/Validation Set:")
    for label, count in zip(val_unique, val_counts):
        print(f"  Class {label}: {count} images ({count/len(val_labels)*100:.1f}%)")
    print(f"  Total: {len(val_labels)} images")
    print(f"{'='*60}\n")

