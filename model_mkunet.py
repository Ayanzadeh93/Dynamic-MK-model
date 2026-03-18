"""
MK-UNet Medical Image Classification Models
============================================
Adapted from MK-UNet (ICCV 2025 CVAMD Oral) segmentation architecture
for binary/multi-class classification of medical images.

Four variants for clean ablation study:

  M1 — MKUNetClassifier
         Original MK-UNet encoder, static equal-weight kernel aggregation.
         Baseline. Direct port of paper encoder + GAP + classifier head.

  M2 — DynamicMKUNetClassifier
         Adds Context-Adaptive Dynamic Routing (our Contribution 1).
         Lightweight router per MKDC block learns α_k(x) weights via
         GAP → FC → ReLU → FC → Softmax, replacing static equal weights.

  M3 — FreqMKUNetClassifier
         Adds Frequency-Aware Branch Conditioning (our Contribution 2).
         Decomposes feature maps into high/low frequency components via FFT.
         Routes small kernels (1×1) to high-freq (fine texture/margins),
         large kernels (5×5) to low-freq (gross morphology/shape).
         Uses static equal-weight aggregation — isolates freq contribution.

  M4 — FreqDynamicMKUNetClassifier  ← FULL MODEL (both contributions)
         Combines Contribution 1 + Contribution 2.
         Frequency-conditioned branch inputs + dynamic routing weights.
         Expected best performance.

Ablation table this enables:
  M1 → M2: effect of dynamic routing alone          (+Contribution 1)
  M1 → M3: effect of frequency conditioning alone   (+Contribution 2)
  M1 → M4: combined effect                          (+Both)
  M3 → M4: adding routing on top of freq            (+Contribution 1 | freq)
  M2 → M4: adding freq on top of routing            (+Contribution 2 | dynamic)

Classification motivation for frequency decomposition:
  High-frequency features → irregular texture, spiculated margins,
                            fine boundary patterns (malignancy indicators)
  Low-frequency features  → gross shape, size, overall mass morphology
                            (structural indicators)
  Dynamic routing learns  → which spatial scale matters most per image,
                            per encoder stage, per pathology type.

Reference: Rahman & Marculescu, "MK-UNet: Multi-kernel Lightweight CNN for
Medical Image Segmentation", ICCV 2025 CVAMD Workshop.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


# =============================================================================
# Utility functions
# =============================================================================

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _init_weights(module, name='', scheme=''):
    if isinstance(module, nn.Conv2d):
        fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        fan_out //= module.groups
        nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    """Activation layer factory."""
    act = act.lower()
    if act == 'relu':
        return nn.ReLU(inplace)
    elif act == 'relu6':
        return nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        return nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        return nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        return nn.GELU()
    elif act == 'hswish':
        return nn.Hardswish(inplace)
    else:
        raise NotImplementedError(f'activation layer [{act}] is not found')


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


# =============================================================================
# Shared attention blocks (used by all four models)
# =============================================================================

class ChannelAttention(nn.Module):
    """
    CBAM-style channel attention.
    Applies both average and max pooling, combines via sigmoid.
    Recalibrates which channels carry discriminative medical features.
    """
    def __init__(self, in_planes, out_planes=None, ratio=16, activation='relu'):
        super().__init__()
        self.in_planes  = in_planes
        self.out_planes = out_planes if out_planes is not None else in_planes
        ratio           = min(ratio, in_planes)
        reduced         = max(in_planes // ratio, 1)

        self.avg_pool  = nn.AdaptiveAvgPool2d(1)
        self.max_pool  = nn.AdaptiveMaxPool2d(1)
        self.activation = act_layer(activation, inplace=True)
        self.fc1       = nn.Conv2d(in_planes, reduced, 1, bias=False)
        self.fc2       = nn.Conv2d(reduced, self.out_planes, 1, bias=False)
        self.sigmoid   = nn.Sigmoid()
        self.apply(partial(_init_weights, scheme=''))

    def forward(self, x):
        avg_out = self.fc2(self.activation(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.activation(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """
    CBAM-style spatial attention.
    Highlights spatially important regions — critical when lesion
    occupies a small fraction of the medical image.
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.apply(partial(_init_weights, scheme=''))

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


# Global cache for frequency masks to accelerate training (Contribution 2)
_FFT_MASK_CACHE = {}

def fft_decompose(x, low_freq_ratio=0.25):
    """
    Decompose feature map x into high and low frequency components via 2D FFT.
    Optimised with mask caching to prevent redundant calculations.
    """
    B, C, H, W = x.shape
    device = x.device
    
    # Check cache for existing mask
    cache_key = (H, W, low_freq_ratio, str(device))
    if cache_key in _FFT_MASK_CACHE:
        mask_low, mask_high = _FFT_MASK_CACHE[cache_key]
    else:
        # 2D FFT Shift Reference
        cx, cy = H // 2, W // 2
        r      = int(min(H, W) * low_freq_ratio)
        
        # Build masks - only happens once per resolution
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        dist        = ((yy - cx) ** 2 + (xx - cy) ** 2).float().sqrt()
        mask_low    = (dist <= r).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        mask_high   = 1.0 - mask_low
        _FFT_MASK_CACHE[cache_key] = (mask_low, mask_high)

    # 2D FFT
    fft = torch.fft.fft2(x, norm='ortho')
    fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))

    # Apply mask and invert for LOW frequency
    # We only need ONE IFFT call because high_freq = x - low_freq (linearity)
    low_freq = torch.fft.ifft2(
        torch.fft.ifftshift(fft_shifted * mask_low, dim=(-2, -1)),
        norm='ortho'
    ).real

    high_freq = x - low_freq
    return high_freq, low_freq


# =============================================================================
# M1 — Original static MKDC + MKIR  (paper baseline)
# =============================================================================

class MultiKernelDepthwiseConv(nn.Module):
    """
    Static parallel depthwise convolutions.
    All kernel branches contribute equally: sum(B_k(x)) for k in K.
    Identical to the original MK-UNet paper (Eq. 2).
    """
    def __init__(self, in_channels, kernel_sizes, stride,
                 activation='relu6', dw_parallel=True):
        super().__init__()
        self.dw_parallel = dw_parallel
        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, ks, stride, ks // 2,
                          groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                act_layer(activation, inplace=True)
            )
            for ks in kernel_sizes
        ])
        self.apply(partial(_init_weights, scheme=''))

    def forward(self, x):
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if not self.dw_parallel:
                x = x + dw_out
        return outputs


class MultiKernelInvertedResidualBlock(nn.Module):
    """
    Original MKIR block from MK-UNet paper.
    PWC1 → MKDC (static) → channel_shuffle → PWC2 → skip.
    """
    def __init__(self, in_c, out_c, stride, expansion_factor=2,
                 dw_parallel=True, add=True,
                 kernel_sizes=[1, 3, 5], activation='relu6'):
        super().__init__()
        assert stride in [1, 2]
        self.in_c             = in_c
        self.out_c            = out_c
        self.add              = add
        self.n_scales         = len(kernel_sizes)
        self.use_skip         = (stride == 1)
        self.ex_c             = int(in_c * expansion_factor)

        self.pconv1 = nn.Sequential(
            nn.Conv2d(in_c, self.ex_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_c),
            act_layer(activation, inplace=True)
        )
        self.mkdc = MultiKernelDepthwiseConv(
            self.ex_c, kernel_sizes, stride, activation, dw_parallel
        )
        combined              = self.ex_c if add else self.ex_c * self.n_scales
        self.combined_channels = combined
        self.pconv2 = nn.Sequential(
            nn.Conv2d(combined, out_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_c),
        )
        if self.use_skip and (in_c != out_c):
            self.conv1x1 = nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)
        self.apply(partial(_init_weights, scheme=''))

    def forward(self, x):
        pout1 = self.pconv1(x)
        outs  = self.mkdc(pout1)
        dout  = sum(outs) if self.add else torch.cat(outs, dim=1)
        dout  = channel_shuffle(dout, gcd(self.combined_channels, self.out_c))
        out   = self.pconv2(dout)
        if self.use_skip:
            if self.in_c != self.out_c:
                x = self.conv1x1(x)
            return x + out
        return out


def mk_irb_bottleneck(in_c, out_c, n, s, expansion_factor=2,
                      dw_parallel=True, add=True,
                      kernel_sizes=[1, 3, 5], activation='relu6'):
    blocks = [MultiKernelInvertedResidualBlock(
        in_c, out_c, s, expansion_factor, dw_parallel, add, kernel_sizes, activation
    )]
    for _ in range(1, n):
        blocks.append(MultiKernelInvertedResidualBlock(
            out_c, out_c, 1, expansion_factor, dw_parallel, add, kernel_sizes, activation
        ))
    return nn.Sequential(*blocks)


# =============================================================================
# M2 — Dynamic MKDC + MKIR  (Contribution 1: context-adaptive routing)
# =============================================================================

class DynamicMultiKernelDepthwiseConv(nn.Module):
    """
    Contribution 1: Context-Adaptive Dynamic Multi-Kernel Selection.

    Instead of equal static weights, a lightweight router network computes
    input-dependent scalar weights α_k(x) ∈ (0,1) for each kernel branch k,
    with Softmax ensuring they sum to 1 and compete against each other.

    Formulation (paper Eq. 2 generalised):
        MKDC_dynamic(x) = CS( Σ_{k∈K} α_k(x) · DWCBk(x) )

    Router: GAP(x) → Conv1×1 → ReLU → Conv1×1 → Softmax over K

    Medical classification rationale:
        A fine-texture malignant lesion should upweight the 1×1 branch.
        A large well-defined benign mass should upweight the 5×5 branch.
        The router learns this per image, per encoder stage, automatically.
    """
    def __init__(self, in_channels, kernel_sizes, stride,
                 activation='relu6', dw_parallel=True):
        super().__init__()
        self.dw_parallel  = dw_parallel
        self.num_branches = len(kernel_sizes)

        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, ks, stride, ks // 2,
                          groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                act_layer(activation, inplace=True)
            )
            for ks in kernel_sizes
        ])

        # Lightweight router — negligible parameter overhead
        squeeze_dim = max(in_channels // 4, 8)
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, squeeze_dim, 1, bias=False),
            act_layer(activation, inplace=True),
            nn.Conv2d(squeeze_dim, self.num_branches, 1, bias=False)
        )
        self.apply(partial(_init_weights, scheme=''))

    def forward(self, x):
        # α_k(x): shape [B, K, 1, 1], Softmax forces competition across K
        alpha = F.softmax(self.router(x), dim=1)

        outputs = []
        for i, dwconv in enumerate(self.dwconvs):
            dw_out      = dwconv(x)
            weight      = alpha[:, i:i+1, :, :]   # [B,1,1,1] → broadcasts
            outputs.append(dw_out * weight)
            if not self.dw_parallel:
                x = x + dw_out * weight
        return outputs


class DynamicMultiKernelInvertedResidualBlock(nn.Module):
    """MKIR with dynamic routing (Contribution 1)."""
    def __init__(self, in_c, out_c, stride, expansion_factor=2,
                 dw_parallel=True, add=True,
                 kernel_sizes=[1, 3, 5], activation='relu6'):
        super().__init__()
        assert stride in [1, 2]
        self.in_c             = in_c
        self.out_c            = out_c
        self.add              = add
        self.n_scales         = len(kernel_sizes)
        self.use_skip         = (stride == 1)
        self.ex_c             = int(in_c * expansion_factor)

        self.pconv1 = nn.Sequential(
            nn.Conv2d(in_c, self.ex_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_c),
            act_layer(activation, inplace=True)
        )
        self.mkdc = DynamicMultiKernelDepthwiseConv(
            self.ex_c, kernel_sizes, stride, activation, dw_parallel
        )
        combined               = self.ex_c if add else self.ex_c * self.n_scales
        self.combined_channels = combined
        self.pconv2 = nn.Sequential(
            nn.Conv2d(combined, out_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_c),
        )
        if self.use_skip and (in_c != out_c):
            self.conv1x1 = nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)
        self.apply(partial(_init_weights, scheme=''))

    def forward(self, x):
        pout1 = self.pconv1(x)
        outs  = self.mkdc(pout1)
        dout  = sum(outs) if self.add else torch.cat(outs, dim=1)
        dout  = channel_shuffle(dout, gcd(self.combined_channels, self.out_c))
        out   = self.pconv2(dout)
        if self.use_skip:
            if self.in_c != self.out_c:
                x = self.conv1x1(x)
            return x + out
        return out


def dynamic_mk_irb_bottleneck(in_c, out_c, n, s, expansion_factor=2,
                               dw_parallel=True, add=True,
                               kernel_sizes=[1, 3, 5], activation='relu6'):
    blocks = [DynamicMultiKernelInvertedResidualBlock(
        in_c, out_c, s, expansion_factor, dw_parallel, add, kernel_sizes, activation
    )]
    for _ in range(1, n):
        blocks.append(DynamicMultiKernelInvertedResidualBlock(
            out_c, out_c, 1, expansion_factor, dw_parallel, add, kernel_sizes, activation
        ))
    return nn.Sequential(*blocks)


# =============================================================================
# M3 — Frequency-Aware MKDC + MKIR  (Contribution 2: freq conditioning)
# =============================================================================

class FreqMultiKernelDepthwiseConv(nn.Module):
    """
    Contribution 2: Frequency-Aware Branch Conditioning.

    Each kernel branch receives a frequency-conditioned version of the input:
      Branch 0 (smallest kernel, 1×1): x + β_h · high_freq(x)
                → boosted sensitivity to fine texture, irregular margins
      Branch 1 (medium kernel,  3×3): x  (unchanged — full spectrum)
      Branch 2 (largest kernel, 5×5): x + β_l · low_freq(x)
                → boosted sensitivity to gross morphology, lesion shape

    β_h, β_l are learnable scalar gates (initialised to 1.0) that allow
    the model to learn how strongly to condition each branch on its
    assigned frequency band. Aggregation is static (equal weights) —
    isolating the frequency contribution for ablation.

    Medical classification rationale:
      Radiologists simultaneously evaluate texture (high-freq cues like
      spiculations, heterogeneity) and structure (low-freq cues like mass
      shape and size). This block gives each kernel branch a pre-filtered
      view that matches what it is geometrically best suited to capture.
    """
    def __init__(self, in_channels, kernel_sizes, stride,
                 activation='relu6', dw_parallel=True,
                 low_freq_ratio=0.25):
        super().__init__()
        assert len(kernel_sizes) == 3, \
            "FreqMKDC requires exactly 3 kernel sizes [small, medium, large]"
        self.dw_parallel    = dw_parallel
        self.low_freq_ratio = low_freq_ratio

        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, ks, stride, ks // 2,
                          groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                act_layer(activation, inplace=True)
            )
            for ks in kernel_sizes
        ])

        # Learnable frequency gates — one per conditioned branch
        self.beta_high = nn.Parameter(torch.ones(1))   # for small kernel
        self.beta_low  = nn.Parameter(torch.ones(1))   # for large kernel

        self.apply(partial(_init_weights, scheme=''))

    def forward(self, x):
        high_freq, low_freq = fft_decompose(x, self.low_freq_ratio)

        # Frequency-conditioned branch inputs
        branch_inputs = [
            x + self.beta_high * high_freq,   # small kernel: detail-boosted
            x,                                  # medium kernel: unchanged
            x + self.beta_low  * low_freq,     # large kernel: structure-boosted
        ]

        outputs = []
        for i, (dwconv, x_in) in enumerate(zip(self.dwconvs, branch_inputs)):
            dw_out = dwconv(x_in)
            outputs.append(dw_out)
            if not self.dw_parallel:
                x = x + dw_out
        return outputs


class FreqMultiKernelInvertedResidualBlock(nn.Module):
    """MKIR with frequency-aware branch conditioning (Contribution 2)."""
    def __init__(self, in_c, out_c, stride, expansion_factor=2,
                 dw_parallel=True, add=True,
                 kernel_sizes=[1, 3, 5], activation='relu6',
                 low_freq_ratio=0.25):
        super().__init__()
        assert stride in [1, 2]
        self.in_c             = in_c
        self.out_c            = out_c
        self.add              = add
        self.n_scales         = len(kernel_sizes)
        self.use_skip         = (stride == 1)
        self.ex_c             = int(in_c * expansion_factor)

        self.pconv1 = nn.Sequential(
            nn.Conv2d(in_c, self.ex_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_c),
            act_layer(activation, inplace=True)
        )
        self.mkdc = FreqMultiKernelDepthwiseConv(
            self.ex_c, kernel_sizes, stride, activation,
            dw_parallel, low_freq_ratio
        )
        combined               = self.ex_c if add else self.ex_c * self.n_scales
        self.combined_channels = combined
        self.pconv2 = nn.Sequential(
            nn.Conv2d(combined, out_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_c),
        )
        if self.use_skip and (in_c != out_c):
            self.conv1x1 = nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)
        self.apply(partial(_init_weights, scheme=''))

    def forward(self, x):
        pout1 = self.pconv1(x)
        outs  = self.mkdc(pout1)
        dout  = sum(outs) if self.add else torch.cat(outs, dim=1)
        dout  = channel_shuffle(dout, gcd(self.combined_channels, self.out_c))
        out   = self.pconv2(dout)
        if self.use_skip:
            if self.in_c != self.out_c:
                x = self.conv1x1(x)
            return x + out
        return out


def freq_mk_irb_bottleneck(in_c, out_c, n, s, expansion_factor=2,
                            dw_parallel=True, add=True,
                            kernel_sizes=[1, 3, 5], activation='relu6',
                            low_freq_ratio=0.25):
    blocks = [FreqMultiKernelInvertedResidualBlock(
        in_c, out_c, s, expansion_factor, dw_parallel, add,
        kernel_sizes, activation, low_freq_ratio
    )]
    for _ in range(1, n):
        blocks.append(FreqMultiKernelInvertedResidualBlock(
            out_c, out_c, 1, expansion_factor, dw_parallel, add,
            kernel_sizes, activation, low_freq_ratio
        ))
    return nn.Sequential(*blocks)


# =============================================================================
# M4 — Freq-Dynamic MKDC + MKIR  (Contribution 1 + 2 combined)
# =============================================================================

class FreqDynamicMultiKernelDepthwiseConv(nn.Module):
    """
    Contributions 1 + 2 combined: Frequency-Aware + Dynamic Routing.

    Each branch receives a frequency-conditioned input (Contribution 2),
    AND the aggregation weights α_k(x) are dynamically learned (Contribution 1).

    Full formulation:
        x_0 = x + β_h · high_freq(x)       ← small kernel input
        x_1 = x                              ← medium kernel input
        x_2 = x + β_l · low_freq(x)        ← large kernel input
        α   = Softmax(Router(x))             ← dynamic weights [B, K]
        out = CS( Σ_k α_k(x) · DWCBk(x_k) )

    This is the most expressive block. The router now makes its routing
    decision knowing that each branch has already been pre-conditioned
    to its frequency band — the routing effectively says:
    "given this image, should we prioritise the fine-detail branch,
    the structural branch, or balance them?"
    """
    def __init__(self, in_channels, kernel_sizes, stride,
                 activation='relu6', dw_parallel=True,
                 low_freq_ratio=0.25):
        super().__init__()
        assert len(kernel_sizes) == 3, \
            "FreqDynamicMKDC requires exactly 3 kernel sizes [small, medium, large]"
        self.dw_parallel    = dw_parallel
        self.num_branches   = len(kernel_sizes)
        self.low_freq_ratio = low_freq_ratio

        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, ks, stride, ks // 2,
                          groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                act_layer(activation, inplace=True)
            )
            for ks in kernel_sizes
        ])

        # Learnable frequency gates (Contribution 2)
        self.beta_high = nn.Parameter(torch.ones(1))
        self.beta_low  = nn.Parameter(torch.ones(1))

        # Dynamic router (Contribution 1)
        squeeze_dim = max(in_channels // 4, 8)
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, squeeze_dim, 1, bias=False),
            act_layer(activation, inplace=True),
            nn.Conv2d(squeeze_dim, self.num_branches, 1, bias=False)
        )
        self.apply(partial(_init_weights, scheme=''))

    def forward(self, x):
        # Frequency decomposition
        high_freq, low_freq = fft_decompose(x, self.low_freq_ratio)

        # Frequency-conditioned branch inputs
        branch_inputs = [
            x + self.beta_high * high_freq,
            x,
            x + self.beta_low  * low_freq,
        ]

        # Dynamic routing weights over K branches
        alpha = F.softmax(self.router(x), dim=1)   # [B, K, 1, 1]

        outputs = []
        for i, (dwconv, x_in) in enumerate(zip(self.dwconvs, branch_inputs)):
            dw_out = dwconv(x_in)
            weight = alpha[:, i:i+1, :, :]
            outputs.append(dw_out * weight)
            if not self.dw_parallel:
                x = x + dw_out * weight
        return outputs


class FreqDynamicMultiKernelInvertedResidualBlock(nn.Module):
    """MKIR with frequency conditioning + dynamic routing (M4, full model)."""
    def __init__(self, in_c, out_c, stride, expansion_factor=2,
                 dw_parallel=True, add=True,
                 kernel_sizes=[1, 3, 5], activation='relu6',
                 low_freq_ratio=0.25):
        super().__init__()
        assert stride in [1, 2]
        self.in_c             = in_c
        self.out_c            = out_c
        self.add              = add
        self.n_scales         = len(kernel_sizes)
        self.use_skip         = (stride == 1)
        self.ex_c             = int(in_c * expansion_factor)

        self.pconv1 = nn.Sequential(
            nn.Conv2d(in_c, self.ex_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_c),
            act_layer(activation, inplace=True)
        )
        self.mkdc = FreqDynamicMultiKernelDepthwiseConv(
            self.ex_c, kernel_sizes, stride, activation,
            dw_parallel, low_freq_ratio
        )
        combined               = self.ex_c if add else self.ex_c * self.n_scales
        self.combined_channels = combined
        self.pconv2 = nn.Sequential(
            nn.Conv2d(combined, out_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_c),
        )
        if self.use_skip and (in_c != out_c):
            self.conv1x1 = nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)
        self.apply(partial(_init_weights, scheme=''))

    def forward(self, x):
        pout1 = self.pconv1(x)
        outs  = self.mkdc(pout1)
        dout  = sum(outs) if self.add else torch.cat(outs, dim=1)
        dout  = channel_shuffle(dout, gcd(self.combined_channels, self.out_c))
        out   = self.pconv2(dout)
        if self.use_skip:
            if self.in_c != self.out_c:
                x = self.conv1x1(x)
            return x + out
        return out


def freq_dynamic_mk_irb_bottleneck(in_c, out_c, n, s, expansion_factor=2,
                                    dw_parallel=True, add=True,
                                    kernel_sizes=[1, 3, 5], activation='relu6',
                                    low_freq_ratio=0.25):
    blocks = [FreqDynamicMultiKernelInvertedResidualBlock(
        in_c, out_c, s, expansion_factor, dw_parallel, add,
        kernel_sizes, activation, low_freq_ratio
    )]
    for _ in range(1, n):
        blocks.append(FreqDynamicMultiKernelInvertedResidualBlock(
            out_c, out_c, 1, expansion_factor, dw_parallel, add,
            kernel_sizes, activation, low_freq_ratio
        ))
    return nn.Sequential(*blocks)


# =============================================================================
# Shared encoder backbone builder
# =============================================================================

def _build_encoder(bottleneck_fn, in_channels, channels, depths,
                   kernel_sizes, expansion_factor, activation, **kwargs):
    """Build 5-stage encoder using the given bottleneck function."""
    enc = nn.ModuleList()
    in_c = in_channels
    for i, (out_c, depth) in enumerate(zip(channels, depths)):
        stage = bottleneck_fn(
            in_c, out_c, depth, s=1,
            expansion_factor=expansion_factor,
            kernel_sizes=kernel_sizes,
            activation=activation,
            **kwargs
        )
        enc.append(stage)
        in_c = out_c
    return enc


# =============================================================================
# M1 — MKUNetClassifier  (original, static)
# =============================================================================

class MKUNetClassifier(nn.Module):
    """
    M1 — Baseline classifier.
    Original MK-UNet encoder with static equal-weight multi-kernel aggregation.
    No routing, no frequency conditioning.
    """
    def __init__(self, num_classes=2, in_channels=3,
                 channels=[16, 32, 64, 96, 160],
                 depths=[1, 1, 1, 1, 1],
                 kernel_sizes=[1, 3, 5],
                 expansion_factor=2,
                 dropout=0.5):
        super().__init__()
        self.num_classes = num_classes
        act = 'relu6'

        self.encoder1 = mk_irb_bottleneck(in_channels,  channels[0], depths[0], 1, expansion_factor, kernel_sizes=kernel_sizes, activation=act)
        self.encoder2 = mk_irb_bottleneck(channels[0],  channels[1], depths[1], 1, expansion_factor, kernel_sizes=kernel_sizes, activation=act)
        self.encoder3 = mk_irb_bottleneck(channels[1],  channels[2], depths[2], 1, expansion_factor, kernel_sizes=kernel_sizes, activation=act)
        self.encoder4 = mk_irb_bottleneck(channels[2],  channels[3], depths[3], 1, expansion_factor, kernel_sizes=kernel_sizes, activation=act)
        self.encoder5 = mk_irb_bottleneck(channels[3],  channels[4], depths[4], 1, expansion_factor, kernel_sizes=kernel_sizes, activation=act)

        self.CA = ChannelAttention(channels[4], ratio=16)
        self.SA = SpatialAttention()

        self.pool       = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(channels[4], num_classes)
        )
        self.apply(partial(_init_weights, scheme=''))

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        out = F.max_pool2d(self.encoder1(x),   2, 2)
        out = F.max_pool2d(self.encoder2(out),  2, 2)
        out = F.max_pool2d(self.encoder3(out),  2, 2)
        out = F.max_pool2d(self.encoder4(out),  2, 2)
        out = F.max_pool2d(self.encoder5(out),  2, 2)
        out = self.CA(out) * out
        out = self.SA(out) * out
        out = self.pool(out).flatten(1)
        return self.classifier(out)

    def get_total_params(self):
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# M2 — DynamicMKUNetClassifier  (+ Contribution 1: dynamic routing)
# =============================================================================

class DynamicMKUNetClassifier(nn.Module):
    """
    M2 — Adds Context-Adaptive Dynamic Routing (Contribution 1).
    Everything else identical to M1.
    Ablation: M1 → M2 isolates the effect of dynamic routing alone.
    """
    def __init__(self, num_classes=2, in_channels=3,
                 channels=[16, 32, 64, 96, 160],
                 depths=[1, 1, 1, 1, 1],
                 kernel_sizes=[1, 3, 5],
                 expansion_factor=2,
                 dropout=0.5):
        super().__init__()
        self.num_classes = num_classes
        act = 'relu6'

        self.encoder1 = dynamic_mk_irb_bottleneck(in_channels,  channels[0], depths[0], 1, expansion_factor, kernel_sizes=kernel_sizes, activation=act)
        self.encoder2 = dynamic_mk_irb_bottleneck(channels[0],  channels[1], depths[1], 1, expansion_factor, kernel_sizes=kernel_sizes, activation=act)
        self.encoder3 = dynamic_mk_irb_bottleneck(channels[1],  channels[2], depths[2], 1, expansion_factor, kernel_sizes=kernel_sizes, activation=act)
        self.encoder4 = dynamic_mk_irb_bottleneck(channels[2],  channels[3], depths[3], 1, expansion_factor, kernel_sizes=kernel_sizes, activation=act)
        self.encoder5 = dynamic_mk_irb_bottleneck(channels[3],  channels[4], depths[4], 1, expansion_factor, kernel_sizes=kernel_sizes, activation=act)

        self.CA = ChannelAttention(channels[4], ratio=16)
        self.SA = SpatialAttention()

        self.pool       = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(channels[4], num_classes)
        )
        self.apply(partial(_init_weights, scheme=''))

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        out = F.max_pool2d(self.encoder1(x),   2, 2)
        out = F.max_pool2d(self.encoder2(out),  2, 2)
        out = F.max_pool2d(self.encoder3(out),  2, 2)
        out = F.max_pool2d(self.encoder4(out),  2, 2)
        out = F.max_pool2d(self.encoder5(out),  2, 2)
        out = self.CA(out) * out
        out = self.SA(out) * out
        out = self.pool(out).flatten(1)
        return self.classifier(out)

    def get_total_params(self):
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# M3 — FreqMKUNetClassifier  (+ Contribution 2: frequency conditioning)
# =============================================================================

class FreqMKUNetClassifier(nn.Module):
    """
    M3 — Adds Frequency-Aware Branch Conditioning (Contribution 2).
    Uses static equal-weight aggregation — dynamic routing is OFF.
    Ablation: M1 → M3 isolates the effect of frequency conditioning alone.
    """
    def __init__(self, num_classes=2, in_channels=3,
                 channels=[16, 32, 64, 96, 160],
                 depths=[1, 1, 1, 1, 1],
                 kernel_sizes=[1, 3, 5],
                 expansion_factor=2,
                 dropout=0.5,
                 low_freq_ratio=0.25):
        super().__init__()
        self.num_classes = num_classes
        act = 'relu6'

        self.encoder1 = freq_mk_irb_bottleneck(in_channels,  channels[0], depths[0], 1, expansion_factor, kernel_sizes=kernel_sizes, activation=act, low_freq_ratio=low_freq_ratio)
        self.encoder2 = freq_mk_irb_bottleneck(channels[0],  channels[1], depths[1], 1, expansion_factor, kernel_sizes=kernel_sizes, activation=act, low_freq_ratio=low_freq_ratio)
        self.encoder3 = freq_mk_irb_bottleneck(channels[1],  channels[2], depths[2], 1, expansion_factor, kernel_sizes=kernel_sizes, activation=act, low_freq_ratio=low_freq_ratio)
        self.encoder4 = freq_mk_irb_bottleneck(channels[2],  channels[3], depths[3], 1, expansion_factor, kernel_sizes=kernel_sizes, activation=act, low_freq_ratio=low_freq_ratio)
        self.encoder5 = freq_mk_irb_bottleneck(channels[3],  channels[4], depths[4], 1, expansion_factor, kernel_sizes=kernel_sizes, activation=act, low_freq_ratio=low_freq_ratio)

        self.CA = ChannelAttention(channels[4], ratio=16)
        self.SA = SpatialAttention()

        self.pool       = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(channels[4], num_classes)
        )
        self.apply(partial(_init_weights, scheme=''))

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        out = F.max_pool2d(self.encoder1(x),   2, 2)
        out = F.max_pool2d(self.encoder2(out),  2, 2)
        out = F.max_pool2d(self.encoder3(out),  2, 2)
        out = F.max_pool2d(self.encoder4(out),  2, 2)
        out = F.max_pool2d(self.encoder5(out),  2, 2)
        out = self.CA(out) * out
        out = self.SA(out) * out
        out = self.pool(out).flatten(1)
        return self.classifier(out)

    def get_total_params(self):
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# M4 — FreqDynamicMKUNetClassifier  (Contribution 1 + 2, full model)
# =============================================================================

class FreqDynamicMKUNetClassifier(nn.Module):
    """
    M4 — Full model combining both contributions.
    Frequency-Aware Branch Conditioning + Context-Adaptive Dynamic Routing.
    Expected best performance across all metrics.
    Ablation:
      M3 → M4: effect of adding routing on top of frequency conditioning
      M2 → M4: effect of adding frequency conditioning on top of routing
    """
    def __init__(self, num_classes=2, in_channels=3,
                 channels=[16, 32, 64, 96, 160],
                 depths=[1, 1, 1, 1, 1],
                 kernel_sizes=[1, 3, 5],
                 expansion_factor=2,
                 dropout=0.5,
                 low_freq_ratio=0.25):
        super().__init__()
        self.num_classes = num_classes
        act = 'relu6'

        self.encoder1 = freq_dynamic_mk_irb_bottleneck(in_channels,  channels[0], depths[0], 1, expansion_factor, kernel_sizes=kernel_sizes, activation=act, low_freq_ratio=low_freq_ratio)
        self.encoder2 = freq_dynamic_mk_irb_bottleneck(channels[0],  channels[1], depths[1], 1, expansion_factor, kernel_sizes=kernel_sizes, activation=act, low_freq_ratio=low_freq_ratio)
        self.encoder3 = freq_dynamic_mk_irb_bottleneck(channels[1],  channels[2], depths[2], 1, expansion_factor, kernel_sizes=kernel_sizes, activation=act, low_freq_ratio=low_freq_ratio)
        self.encoder4 = freq_dynamic_mk_irb_bottleneck(channels[2],  channels[3], depths[3], 1, expansion_factor, kernel_sizes=kernel_sizes, activation=act, low_freq_ratio=low_freq_ratio)
        self.encoder5 = freq_dynamic_mk_irb_bottleneck(channels[3],  channels[4], depths[4], 1, expansion_factor, kernel_sizes=kernel_sizes, activation=act, low_freq_ratio=low_freq_ratio)

        self.CA = ChannelAttention(channels[4], ratio=16)
        self.SA = SpatialAttention()

        self.pool       = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(channels[4], num_classes)
        )
        self.apply(partial(_init_weights, scheme=''))

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        out = F.max_pool2d(self.encoder1(x),   2, 2)
        out = F.max_pool2d(self.encoder2(out),  2, 2)
        out = F.max_pool2d(self.encoder3(out),  2, 2)
        out = F.max_pool2d(self.encoder4(out),  2, 2)
        out = F.max_pool2d(self.encoder5(out),  2, 2)
        out = self.CA(out) * out
        out = self.SA(out) * out
        out = self.pool(out).flatten(1)
        return self.classifier(out)

    def get_total_params(self):
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Factory functions
# =============================================================================

def _print_model_info(name, tag, method, model, device):
    print(f"\n{'='*65}")
    print(f"  Model : {name}")
    print(f"  Tag   : {tag}")
    print(f"  Method: {method}")
    print(f"  Params: {model.get_total_params():,}")
    print(f"  Device: {device}")
    print(f"{'='*65}\n")


def create_mkunet_original(num_classes=2, dropout=0.5, device=None):
    """M1 — Original static MK-UNet classifier."""
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = MKUNetClassifier(num_classes=num_classes, dropout=dropout).to(device)
    _print_model_info(
        "MK-UNet Original (M1)", "static",
        "Static equal-weight multi-kernel aggregation", model, device
    )
    return model


def create_mkunet_dynamic(num_classes=2, dropout=0.5, device=None):
    """M2 — Dynamic routing classifier (Contribution 1)."""
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = DynamicMKUNetClassifier(num_classes=num_classes, dropout=dropout).to(device)
    _print_model_info(
        "MK-UNet Dynamic (M2)", "+dynamic_routing",
        "Context-Adaptive Routing via Softmax α_k(x)", model, device
    )
    return model


def create_mkunet_freq(num_classes=2, dropout=0.5, low_freq_ratio=0.25, device=None):
    """M3 — Frequency-aware classifier (Contribution 2)."""
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = FreqMKUNetClassifier(
        num_classes=num_classes, dropout=dropout,
        low_freq_ratio=low_freq_ratio
    ).to(device)
    _print_model_info(
        "MK-UNet Frequency-Aware (M3)", "+freq_conditioning",
        "FFT branch conditioning: high→small kernel, low→large kernel", model, device
    )
    return model


def create_mkunet_freq_dynamic(num_classes=2, dropout=0.5, low_freq_ratio=0.25, device=None):
    """M4 — Full model: frequency conditioning + dynamic routing."""
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = FreqDynamicMKUNetClassifier(
        num_classes=num_classes, dropout=dropout,
        low_freq_ratio=low_freq_ratio
    ).to(device)
    _print_model_info(
        "MK-UNet Freq-Dynamic (M4 — FULL)", "+freq+dynamic",
        "Frequency conditioning + Context-Adaptive Routing (both contributions)", model, device
    )
    return model


# =============================================================================
# Ablation summary printer
# =============================================================================

def print_ablation_summary(device=None):
    """Print parameter counts and ablation design for all four models."""
    device = device or torch.device('cpu')
    models = {
        'M1 — Original (baseline)' : create_mkunet_original(device=device),
        'M2 — + Dynamic routing'   : create_mkunet_dynamic(device=device),
        'M3 — + Freq conditioning' : create_mkunet_freq(device=device),
        'M4 — + Both (full model)' : create_mkunet_freq_dynamic(device=device),
    }
    print(f"\n{'='*65}")
    print("  ABLATION STUDY DESIGN")
    print(f"{'='*65}")
    print(f"  {'Model':<35} {'Params':>10}  {'Delta':>10}")
    print(f"  {'-'*55}")
    base = None
    for name, m in models.items():
        p = m.get_total_params()
        delta = f"+{p - base:,}" if base is not None else "—"
        print(f"  {name:<35} {p:>10,}  {delta:>10}")
        if base is None:
            base = p
    print(f"\n  Ablation comparisons:")
    print(f"    M1 → M2 : isolates dynamic routing        (Contribution 1)")
    print(f"    M1 → M3 : isolates frequency conditioning (Contribution 2)")
    print(f"    M1 → M4 : combined effect of both")
    print(f"    M2 → M4 : freq conditioning on top of routing")
    print(f"    M3 → M4 : dynamic routing on top of freq")
    print(f"{'='*65}\n")


# =============================================================================
# Quick test
# =============================================================================

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(2, 3, 224, 224).to(device)

    m1 = create_mkunet_original(device=device)
    m2 = create_mkunet_dynamic(device=device)
    m3 = create_mkunet_freq(device=device)
    m4 = create_mkunet_freq_dynamic(device=device)

    for tag, m in [('M1', m1), ('M2', m2), ('M3', m3), ('M4', m4)]:
        out = m(x)
        print(f"  {tag} output shape: {out.shape}  ✓")

    print_ablation_summary(device=torch.device('cpu'))
    print("All four models verified successfully.")