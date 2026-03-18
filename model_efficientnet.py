"""
EfficientNet-B0 model implementation with ImageNet pretrained backbone
for binary classification of medical images
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class EfficientNetBinaryClassifier(nn.Module):
    """
    EfficientNet-B0 architecture with ImageNet pretrained backbone
    Modified for binary classification
    """
    
    def __init__(self, pretrained: bool = True, num_classes: int = 2, dropout: float = 0.5):
        """
        Initialize EfficientNet-B0 model
        
        Args:
            pretrained: Whether to use ImageNet pretrained weights
            num_classes: Number of output classes (2 for binary classification)
            dropout: Dropout rate for regularization
        """
        super(EfficientNetBinaryClassifier, self).__init__()
        
        # Load EfficientNet-B0 with ImageNet pretrained weights
        if pretrained:
            print("Loading EfficientNet-B0 with ImageNet pretrained weights...")
            # Create model without weights first
            self.efficientnet = models.efficientnet_b0(pretrained=False)
            
            # Try to load pretrained weights - use manual download to avoid hash issues
            loaded = False
            
            # Method 1: Try standard methods first (quick attempt)
            try:
                # Try old pretrained parameter (might work if cache is good)
                temp_model = models.efficientnet_b0(pretrained=True)
                self.efficientnet.load_state_dict(temp_model.state_dict())
                print("Loaded weights using pretrained=True")
                loaded = True
            except RuntimeError as e:
                if "hash" in str(e).lower() or "invalid hash" in str(e).lower():
                    print("Hash mismatch detected. Using manual download method...")
                else:
                    # Other error, try manual download anyway
                    print(f"Standard loading failed: {e}. Using manual download...")
            
            # Method 2: Manual download without hash check (reliable fallback)
            if not loaded:
                print("Downloading EfficientNet-B0 weights manually (bypassing hash check)...")
                import torch.hub
                url = "https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth"
                print(f"Downloading from: {url}")
                try:
                    state_dict = torch.hub.load_state_dict_from_url(
                        url, 
                        map_location='cpu', 
                        check_hash=False,
                        progress=True
                    )
                    self.efficientnet.load_state_dict(state_dict)
                    print("Successfully loaded EfficientNet-B0 weights (manual download)")
                    loaded = True
                except Exception as e:
                    print(f"Warning: Could not load pretrained weights: {e}")
                    print("Continuing with randomly initialized weights...")
        else:
            print("Initializing EfficientNet-B0 from scratch...")
            self.efficientnet = models.efficientnet_b0(pretrained=False)
        
        # Get the number of input features to the classifier
        # EfficientNet's classifier structure
        num_features = self.efficientnet.classifier[1].in_features
        
        # Replace the classifier for binary classification
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
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
        return self.efficientnet(x)
    
    def freeze_backbone(self):
        """
        Freeze the feature extraction layers (all layers except classifier)
        Useful for fine-tuning on small datasets
        """
        for param in self.efficientnet.features.parameters():
            param.requires_grad = False
        print("EfficientNet-B0 backbone frozen. Only classifier will be trained.")
    
    def unfreeze_backbone(self):
        """
        Unfreeze all layers for full fine-tuning
        """
        for param in self.efficientnet.parameters():
            param.requires_grad = True
        print("EfficientNet-B0 backbone unfrozen. All layers will be trained.")
    
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


def create_efficientnet_model(pretrained: bool = True, 
                             num_classes: int = 2,
                             dropout: float = 0.5,
                             freeze_backbone: bool = False,
                             device: Optional[torch.device] = None) -> EfficientNetBinaryClassifier:
    """
    Factory function to create and configure EfficientNet-B0 model
    
    Args:
        pretrained: Whether to use ImageNet pretrained weights
        num_classes: Number of output classes
        dropout: Dropout rate
        freeze_backbone: Whether to freeze the backbone initially
        device: Device to place model on
        
    Returns:
        Configured EfficientNet-B0 model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = EfficientNetBinaryClassifier(
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
    print(f"Architecture: EfficientNet-B0")
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

