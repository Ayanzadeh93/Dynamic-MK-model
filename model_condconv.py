"""
CondConv toy classifier for binary classification on medical images.
Uses conditionally-parameterized convolutions (CondConv) without
ImageNet pretraining (no public weights). Backbone remains trainable
by default; freezing is supported but not recommended here.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RouteFunc(nn.Module):
    """Routing function that produces expert weights for CondConv."""

    def __init__(self, c_in: int, num_experts: int):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(c_in, num_experts)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


class CondConv2d(nn.Module):
    """
    Conditionally-parameterized convolution.
    Based on: https://papers.nips.cc/paper/8412-condconv-conditionally-parameterized-convolutions-for-efficient-inference.pdf
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        num_experts: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_experts = num_experts

        self.weight = nn.Parameter(
            torch.Tensor(num_experts, out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_experts, out_channels))
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, routing_weight: torch.Tensor) -> torch.Tensor:
        b, c_in, h, w = x.size()
        k, c_out, c_in, kh, kw = self.weight.size()  # noqa: F841  # c_in shadowed for clarity
        x = x.view(1, -1, h, w)
        weight = self.weight.view(k, -1)
        combined_weight = torch.mm(routing_weight, weight).view(-1, c_in, kh, kw)
        if self.bias is not None:
            combined_bias = torch.mm(routing_weight, self.bias).view(-1)
            out = F.conv2d(
                x,
                weight=combined_weight,
                bias=combined_bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups * b,
            )
        else:
            out = F.conv2d(
                x,
                weight=combined_weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups * b,
            )
        out = out.view(b, c_out, out.size(-2), out.size(-1))
        return out


class CondConvBinaryClassifier(nn.Module):
    """
    Lightweight CondConv-based classifier (no pretrained weights available).
    """

    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.5,
        num_experts: int = 4,
    ):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Routing for CondConv blocks
        self.route1 = RouteFunc(32, num_experts)
        self.cond1 = CondConv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True, num_experts=num_experts)
        self.bn1 = nn.BatchNorm2d(64)

        self.route2 = RouteFunc(64, num_experts)
        self.cond2 = CondConv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True, num_experts=num_experts)
        self.bn2 = nn.BatchNorm2d(128)

        # Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        w1 = self.route1(x)
        x = self.cond1(x, w1)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        w2 = self.route2(x)
        x = self.cond2(x, w2)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)

        x = self.head(x)
        return x

    def freeze_backbone(self):
        """
        Freeze feature layers (stem + condconv blocks). Not recommended for this custom model.
        """
        for module in [self.stem, self.cond1, self.bn1, self.route1, self.cond2, self.bn2, self.route2]:
            for p in module.parameters():
                p.requires_grad = False
        print("CondConv backbone frozen. Only classifier will be trained.")

    def unfreeze_backbone(self):
        """Unfreeze all layers."""
        for p in self.parameters():
            p.requires_grad = True
        print("CondConv backbone unfrozen. All layers will be trained.")

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def create_condconv_model(
    pretrained: bool = False,  # No pretrained weights available
    num_classes: int = 2,
    dropout: float = 0.5,
    freeze_backbone: bool = False,
    num_experts: int = 4,
    device: Optional[torch.device] = None,
) -> CondConvBinaryClassifier:
    """
    Factory to create CondConv model. Pretrained flag is kept for API parity but ignored.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if pretrained:
        print("Warning: pretrained=True requested, but CondConv has no pretrained weights. Using random init.")

    model = CondConvBinaryClassifier(
        num_classes=num_classes,
        dropout=dropout,
        num_experts=num_experts,
    )

    if freeze_backbone:
        model.freeze_backbone()

    model = model.to(device)

    # Print model information
    print(f"\n{'='*60}")
    print("Model Information")
    print(f"{'='*60}")
    print("Architecture: CondConv (custom)")
    print(f"Pretrained: {pretrained} (no weights available, using random init)")
    print(f"Number of Classes: {num_classes}")
    print(f"Dropout Rate: {dropout}")
    print(f"Backbone Frozen: {freeze_backbone}")
    print(f"Experts per CondConv: {num_experts}")
    print(f"Total Parameters: {model.get_total_params():,}")
    print(f"Trainable Parameters: {model.get_trainable_params():,}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    return model


def get_optimizer(model: nn.Module, learning_rate: float = 0.001, weight_decay: float = 1e-4, optimizer_type: str = "adam"):
    """Create optimizer (same API as other models)."""
    opt_type = optimizer_type.lower()
    if opt_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if opt_type == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if opt_type == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def get_scheduler(optimizer, scheduler_type: str = "cosine", num_epochs: int = 50, eta_min: float = 1e-6):
    """Create LR scheduler (matches existing helper API)."""
    stype = scheduler_type.lower()
    if stype == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=eta_min)
    if stype == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs // 3, gamma=0.1)
    if stype == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5, verbose=True)
    raise ValueError(f"Unknown scheduler type: {scheduler_type}")







