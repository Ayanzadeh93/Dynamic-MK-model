"""
CondConv MobileNetV2-style classifier (no pretrained weights available).
Implements inverted residual blocks with CondConv depthwise layers.
"""

import math
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RouteFunc(nn.Module):
    """Routing function to produce expert weights."""

    def __init__(self, c_in: int, num_experts: int):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c_in, num_experts)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


class CondConv2d(nn.Module):
    """Conditionally-parameterized convolution."""

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
        b, _, h, w = x.size()
        k, c_out, c_in_per_group, kh, kw = self.weight.size()
        x = x.view(1, -1, h, w)
        weight = self.weight.view(k, -1)
        combined_weight = torch.mm(routing_weight, weight).view(-1, c_in_per_group, kh, kw)
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


def _make_divisible(v: int, divisor: int = 8) -> int:
    return int(math.ceil(v / divisor) * divisor)


class InvertedResidualCondConv(nn.Module):
    """MobileNetV2 inverted residual block with CondConv depthwise."""

    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int, num_experts: int):
        super().__init__()
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            layers.extend(
                [
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                ]
            )

        # CondConv depthwise
        self.route = RouteFunc(hidden_dim, num_experts)
        self.condconv = CondConv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=hidden_dim,
            bias=False,
            num_experts=num_experts,
        )
        self.bn_depthwise = nn.BatchNorm2d(hidden_dim)
        self.relu_depthwise = nn.ReLU6(inplace=True)

        # Pointwise linear
        self.pointwise = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

        self.expand_ratio = expand_ratio
        self.hidden_dim = hidden_dim
        self.out_channels = oup

        self.pre_expand = nn.Sequential(*layers) if layers else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = x
        if self.pre_expand is not None:
            out = self.pre_expand(out)

        w = self.route(out)
        out = self.condconv(out, w)
        out = self.bn_depthwise(out)
        out = self.relu_depthwise(out)

        out = self.pointwise(out)

        if self.use_res_connect:
            out = identity + out
        return out


class CondConvMobileNetV2(nn.Module):
    """
    MobileNetV2-style network with CondConv depthwise layers.
    No pretrained weights are available for this configuration.
    """

    def __init__(self, num_classes: int = 2, width_mult: float = 1.0, dropout: float = 0.2, num_experts: int = 4):
        super().__init__()
        input_channel = _make_divisible(32 * width_mult, 8)
        last_channel = _make_divisible(1280 * width_mult, 8) if width_mult > 1.0 else 1280

        # (t, c, n, s)
        inverted_residual_setting: List[Tuple[int, int, int, int]] = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]

        features: List[nn.Module] = [
            nn.Conv2d(3, input_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True),
        ]

        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidualCondConv(input_channel, output_channel, stride, t, num_experts))
                input_channel = output_channel

        features.extend(
            [
                nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False),
                nn.BatchNorm2d(last_channel),
                nn.ReLU6(inplace=True),
            ]
        )

        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(last_channel, num_classes),
        )

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def freeze_backbone(self):
        for name, module in self.named_children():
            if name != "classifier":
                for p in module.parameters():
                    p.requires_grad = False
        print("CondConv-MobileNet backbone frozen. Only classifier will be trained.")

    def unfreeze_backbone(self):
        for p in self.parameters():
            p.requires_grad = True
        print("CondConv-MobileNet backbone unfrozen. All layers will be trained.")

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def create_condconv_mobilenet_model(
    pretrained: bool = False,
    num_classes: int = 2,
    dropout: float = 0.2,
    freeze_backbone: bool = False,
    num_experts: int = 4,
    width_mult: float = 1.0,
    device: Optional[torch.device] = None,
) -> CondConvMobileNetV2:
    """
    Factory for CondConv MobileNetV2-style model. Pretrained flag is accepted for API parity but not used.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if pretrained:
        print("Warning: pretrained=True requested, but no CondConv-MobileNet pretrained weights exist. Using random init.")

    model = CondConvMobileNetV2(
        num_classes=num_classes,
        width_mult=width_mult,
        dropout=dropout,
        num_experts=num_experts,
    )

    if freeze_backbone:
        model.freeze_backbone()

    model = model.to(device)

    print(f"\n{'='*60}")
    print("Model Information")
    print(f"{'='*60}")
    print("Architecture: CondConv-MobileNetV2-style")
    print(f"Pretrained: {pretrained} (no weights available, random init)")
    print(f"Number of Classes: {num_classes}")
    print(f"Dropout Rate: {dropout}")
    print(f"Backbone Frozen: {freeze_backbone}")
    print(f"Experts per CondConv: {num_experts}")
    print(f"Width Multiplier: {width_mult}")
    print(f"Total Parameters: {model.get_total_params():,}")
    print(f"Trainable Parameters: {model.get_trainable_params():,}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    return model


def get_optimizer(model: nn.Module, learning_rate: float = 0.001, weight_decay: float = 1e-4, optimizer_type: str = "adam"):
    opt_type = optimizer_type.lower()
    if opt_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if opt_type == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if opt_type == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def get_scheduler(optimizer, scheduler_type: str = "cosine", num_epochs: int = 50, eta_min: float = 1e-6):
    stype = scheduler_type.lower()
    if stype == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=eta_min)
    if stype == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs // 3, gamma=0.1)
    if stype == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5, verbose=True)
    raise ValueError(f"Unknown scheduler type: {scheduler_type}")







