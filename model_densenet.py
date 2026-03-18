"""
DenseNet-121 model implementation with ImageNet pretrained backbone
for binary classification of medical images
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class DenseNet121BinaryClassifier(nn.Module):
    """
    DenseNet-121 architecture with ImageNet pretrained backbone
    Modified for binary classification
    """
    
    def __init__(self, pretrained: bool = True, num_classes: int = 2, dropout: float = 0.5):
        """
        Initialize DenseNet-121 model
        
        Args:
            pretrained: Whether to use ImageNet pretrained weights
            num_classes: Number of output classes (2 for binary classification)
            dropout: Dropout rate for regularization
        """
        super(DenseNet121BinaryClassifier, self).__init__()
        
        # Load DenseNet-121 with ImageNet pretrained weights
        if pretrained:
            print("Loading DenseNet-121 with ImageNet pretrained weights...")
            try:
                # Try new weights API (torchvision >= 0.13)
                weights = models.DenseNet121_Weights.IMAGENET1K_V1
                self.densenet = models.densenet121(weights=weights)
            except (AttributeError, TypeError):
                # Fallback to old pretrained parameter
                self.densenet = models.densenet121(pretrained=True)
        else:
            print("Initializing DenseNet-121 from scratch...")
            self.densenet = models.densenet121(pretrained=False)
        
        # Get the number of input features to the classifier
        num_features = self.densenet.classifier.in_features
        
        # Replace the classifier for binary classification
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, num_classes)
        )
        
        self.num_classes = num_classes
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        return self.densenet(x)
    
    def freeze_backbone(self):
        """
        Freeze the feature extraction layers (all layers except classifier)
        Useful for fine-tuning on small datasets
        """
        for param in self.densenet.features.parameters():
            param.requires_grad = False
        print("DenseNet-121 backbone frozen. Only classifier will be trained.")
    
    def unfreeze_backbone(self):
        """
        Unfreeze all layers for full fine-tuning
        """
        for param in self.densenet.parameters():
            param.requires_grad = True
        print("DenseNet-121 backbone unfrozen. All layers will be trained.")
    
    def get_trainable_params(self):
        """
        Get the number of trainable parameters
        
        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self):
        """
        Get the total number of parameters
        
        Returns:
            Total number of parameters
        """
        return sum(p.numel() for p in self.parameters())


def create_densenet_model(pretrained: bool = True, 
                          num_classes: int = 2,
                          dropout: float = 0.5,
                          freeze_backbone: bool = False,
                          device: Optional[torch.device] = None) -> DenseNet121BinaryClassifier:
    """
    Factory function to create and configure DenseNet-121 model
    
    Args:
        pretrained: Whether to use ImageNet pretrained weights
        num_classes: Number of output classes
        dropout: Dropout rate
        freeze_backbone: Whether to freeze the backbone initially
        device: Device to place model on
        
    Returns:
        Configured DenseNet-121 model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = DenseNet121BinaryClassifier(
        pretrained=pretrained,
        num_classes=num_classes,
        dropout=dropout
    )
    
    if freeze_backbone:
        model.freeze_backbone()
    
    model = model.to(device)
    
    # Print model information
    print(f"\n{'='*60}")
    print("Model Information")
    print(f"{'='*60}")
    print(f"Architecture: DenseNet-121")
    print(f"Pretrained: {pretrained} (ImageNet)")
    print(f"Number of Classes: {num_classes}")
    print(f"Dropout Rate: {dropout}")
    print(f"Backbone Frozen: {freeze_backbone}")
    print(f"Total Parameters: {model.get_total_params():,}")
    print(f"Trainable Parameters: {model.get_trainable_params():,}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    return model


def get_optimizer(model: nn.Module, learning_rate: float = 0.001, 
                 weight_decay: float = 1e-4, optimizer_type: str = 'adam'):
    """
    Create optimizer for model training
    
    Args:
        model: PyTorch model
        learning_rate: Initial learning rate
        weight_decay: L2 regularization coefficient
        optimizer_type: Type of optimizer ('adam', 'sgd', 'adamw')
        
    Returns:
        Optimizer
    """
    if optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizer


def get_scheduler(optimizer, scheduler_type: str = 'cosine', 
                 num_epochs: int = 50, eta_min: float = 1e-6):
    """
    Create learning rate scheduler
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler ('cosine', 'step', 'plateau')
        num_epochs: Total number of epochs
        eta_min: Minimum learning rate for cosine scheduler
        
    Returns:
        Learning rate scheduler
    """
    if scheduler_type.lower() == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=eta_min
        )
    elif scheduler_type.lower() == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=num_epochs // 3,
            gamma=0.1
        )
    elif scheduler_type.lower() == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler







