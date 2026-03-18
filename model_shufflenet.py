"""
ShuffleNetV2 model implementation with ImageNet pretrained backbone
for binary classification of medical images
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class ShuffleNetV2BinaryClassifier(nn.Module):
    """
    ShuffleNetV2 architecture with ImageNet pretrained backbone
    Modified for binary classification
    """
    
    def __init__(self, pretrained: bool = True, num_classes: int = 2, 
                 model_size: str = 'x1.0'):
        """
        Initialize ShuffleNetV2 model
        
        Args:
            pretrained: Whether to use ImageNet pretrained weights
            num_classes: Number of output classes (2 for binary classification)
            model_size: Model size ('x0.5', 'x1.0', 'x1.5', 'x2.0')
        """
        super(ShuffleNetV2BinaryClassifier, self).__init__()
        
        # Load ShuffleNetV2 with ImageNet pretrained weights
        if pretrained:
            print(f"Loading ShuffleNetV2-{model_size} with ImageNet pretrained weights...")
            try:
                # Try new weights API (torchvision >= 0.13)
                if model_size == 'x0.5':
                    weights = models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1
                    self.shufflenet = models.shufflenet_v2_x0_5(weights=weights)
                elif model_size == 'x1.0':
                    weights = models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1
                    self.shufflenet = models.shufflenet_v2_x1_0(weights=weights)
                elif model_size == 'x1.5':
                    weights = models.ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1
                    self.shufflenet = models.shufflenet_v2_x1_5(weights=weights)
                elif model_size == 'x2.0':
                    weights = models.ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1
                    self.shufflenet = models.shufflenet_v2_x2_0(weights=weights)
                else:
                    raise ValueError(f"Unknown model size: {model_size}")
            except (AttributeError, TypeError):
                # Fallback to old pretrained parameter
                if model_size == 'x0.5':
                    self.shufflenet = models.shufflenet_v2_x0_5(pretrained=True)
                elif model_size == 'x1.0':
                    self.shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
                elif model_size == 'x1.5':
                    self.shufflenet = models.shufflenet_v2_x1_5(pretrained=True)
                elif model_size == 'x2.0':
                    self.shufflenet = models.shufflenet_v2_x2_0(pretrained=True)
                else:
                    raise ValueError(f"Unknown model size: {model_size}")
        else:
            print(f"Initializing ShuffleNetV2-{model_size} from scratch...")
            if model_size == 'x0.5':
                self.shufflenet = models.shufflenet_v2_x0_5(pretrained=False)
            elif model_size == 'x1.0':
                self.shufflenet = models.shufflenet_v2_x1_0(pretrained=False)
            elif model_size == 'x1.5':
                self.shufflenet = models.shufflenet_v2_x1_5(pretrained=False)
            elif model_size == 'x2.0':
                self.shufflenet = models.shufflenet_v2_x2_0(pretrained=False)
            else:
                raise ValueError(f"Unknown model size: {model_size}")
        
        # Get the number of input features to the classifier
        num_features = self.shufflenet.fc.in_features
        
        # Replace the classifier for binary classification
        self.shufflenet.fc = nn.Linear(num_features, num_classes)
        
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
        return self.shufflenet(x)
    
    def freeze_backbone(self):
        """
        Freeze the feature extraction layers (all layers except classifier)
        Useful for fine-tuning on small datasets
        """
        for param in self.shufflenet.parameters():
            param.requires_grad = False
        # Unfreeze classifier
        for param in self.shufflenet.fc.parameters():
            param.requires_grad = True
        print(f"ShuffleNetV2-{self.model_size} backbone frozen. Only classifier will be trained.")
    
    def unfreeze_backbone(self):
        """
        Unfreeze all layers for full fine-tuning
        """
        for param in self.shufflenet.parameters():
            param.requires_grad = True
        print(f"ShuffleNetV2-{self.model_size} backbone unfrozen. All layers will be trained.")
    
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


def create_shufflenet_model(pretrained: bool = True, 
                            num_classes: int = 2,
                            freeze_backbone: bool = False,
                            model_size: str = 'x1.0',
                            device: Optional[torch.device] = None) -> ShuffleNetV2BinaryClassifier:
    """
    Factory function to create and configure ShuffleNetV2 model
    
    Args:
        pretrained: Whether to use ImageNet pretrained weights
        num_classes: Number of output classes
        freeze_backbone: Whether to freeze the backbone initially
        model_size: Model size ('x0.5', 'x1.0', 'x1.5', 'x2.0')
        device: Device to place model on
        
    Returns:
        Configured ShuffleNetV2 model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ShuffleNetV2BinaryClassifier(
        pretrained=pretrained,
        num_classes=num_classes,
        model_size=model_size
    )
    
    if freeze_backbone:
        model.freeze_backbone()
    
    model = model.to(device)
    
    # Print model information
    print(f"\n{'='*60}")
    print("Model Information")
    print(f"{'='*60}")
    print(f"Architecture: ShuffleNetV2-{model_size}")
    print(f"Pretrained: {pretrained} (ImageNet)")
    print(f"Number of Classes: {num_classes}")
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







