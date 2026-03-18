"""
MobileNetV3 model implementation with ImageNet pretrained backbone
for binary classification of medical images
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class MobileNetV3BinaryClassifier(nn.Module):
    """
    MobileNetV3 architecture with ImageNet pretrained backbone
    Modified for binary classification
    """
    
    def __init__(self, pretrained: bool = True, num_classes: int = 2, 
                 dropout: float = 0.5, model_size: str = 'small'):
        """
        Initialize MobileNetV3 model
        
        Args:
            pretrained: Whether to use ImageNet pretrained weights
            num_classes: Number of output classes (2 for binary classification)
            dropout: Dropout rate for regularization
            model_size: 'small' or 'large' (default: 'small' for efficiency)
        """
        super(MobileNetV3BinaryClassifier, self).__init__()
        
        # Load MobileNetV3 with ImageNet pretrained weights
        if pretrained:
            print(f"Loading MobileNetV3-{model_size} with ImageNet pretrained weights...")
            if model_size.lower() == 'small':
                self.mobilenet = models.mobilenet_v3_small(pretrained=True)
            else:
                self.mobilenet = models.mobilenet_v3_large(pretrained=True)
        else:
            print(f"Initializing MobileNetV3-{model_size} from scratch...")
            if model_size.lower() == 'small':
                self.mobilenet = models.mobilenet_v3_small(pretrained=False)
            else:
                self.mobilenet = models.mobilenet_v3_large(pretrained=False)
        
        # Get the number of input features to the classifier
        # MobileNetV3's classifier structure varies by size
        if model_size.lower() == 'small':
            num_features = self.mobilenet.classifier[3].in_features
            # Replace classifier for binary classification
            self.mobilenet.classifier = nn.Sequential(
                nn.Linear(576, 1024),
                nn.Hardswish(),
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(1024, num_classes)
            )
        else:  # large
            num_features = self.mobilenet.classifier[3].in_features
            # Replace classifier for binary classification
            self.mobilenet.classifier = nn.Sequential(
                nn.Linear(960, 1280),
                nn.Hardswish(),
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(1280, num_classes)
            )
        
        self.num_classes = num_classes
        self.model_size = model_size
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        return self.mobilenet(x)
    
    def freeze_backbone(self):
        """
        Freeze the feature extraction layers (all layers except classifier)
        Useful for fine-tuning on small datasets
        """
        for param in self.mobilenet.features.parameters():
            param.requires_grad = False
        print(f"MobileNetV3-{self.model_size} backbone frozen. Only classifier will be trained.")
    
    def unfreeze_backbone(self):
        """
        Unfreeze all layers for full fine-tuning
        """
        for param in self.mobilenet.parameters():
            param.requires_grad = True
        print(f"MobileNetV3-{self.model_size} backbone unfrozen. All layers will be trained.")
    
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


def create_mobilenetv3_model(pretrained: bool = True, 
                             num_classes: int = 2,
                             dropout: float = 0.5,
                             freeze_backbone: bool = False,
                             model_size: str = 'small',
                             device: Optional[torch.device] = None) -> MobileNetV3BinaryClassifier:
    """
    Factory function to create and configure MobileNetV3 model
    
    Args:
        pretrained: Whether to use ImageNet pretrained weights
        num_classes: Number of output classes
        dropout: Dropout rate
        freeze_backbone: Whether to freeze the backbone initially
        model_size: 'small' or 'large' (default: 'small')
        device: Device to place model on
        
    Returns:
        Configured MobileNetV3 model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = MobileNetV3BinaryClassifier(
        pretrained=pretrained,
        num_classes=num_classes,
        dropout=dropout,
        model_size=model_size
    )
    
    if freeze_backbone:
        model.freeze_backbone()
    
    model = model.to(device)
    
    # Print model information
    print(f"\n{'='*60}")
    print("Model Information")
    print(f"{'='*60}")
    print(f"Architecture: MobileNetV3-{model_size}")
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







