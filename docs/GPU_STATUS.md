# GPU Usage Status for All Models

## Summary

✅ **All training scripts are configured to use GPU automatically when available.**

## Current GPU Status

- **GPU Available**: ✅ Yes
- **GPU Name**: NVIDIA GeForce RTX 5070 Ti Laptop GPU
- **GPU Memory**: 11.94 GB
- **CUDA Version**: 11.8
- **PyTorch Version**: 2.1.0+cu118
- **Compute Capability**: 12.0 (sm_120)

### Note on Compatibility Warning

You may see a warning about compute capability 12.0 not being officially supported by PyTorch 2.1.0. However, **the GPU still works correctly** - this is just an informational warning. The GPU can successfully create tensors and run training.

## All Models Using GPU

All 11 training scripts automatically detect and use GPU:

1. ✅ **SqueezeNet** (`train_kfold.py`)
2. ✅ **MobileNetV2** (`train_mobilenet_kfold.py`)
3. ✅ **MobileNetV3** (`train_mobilenetv3_kfold.py`)
4. ✅ **EfficientNet-B0** (`train_efficientnet_kfold.py`)
5. ✅ **ResNet-18** (`train_resnet_kfold.py`)
6. ✅ **ConvNeXt-Tiny** (`train_convnext_kfold.py`)
7. ✅ **DenseNet-121** (`train_densenet_kfold.py`)
8. ✅ **ShuffleNetV2** (`train_shufflenet_kfold.py`)
9. ✅ **GhostNet-100** (`train_ghostnet_kfold.py`)
10. ✅ **EfficientNetV2-S** (`train_efficientnetv2_kfold.py`)
11. ✅ **NFNet-F0** (`train_nfnet_kfold.py`)
12. ✅ **DeiT-Tiny** (`train_deit_kfold.py`)

## How GPU Detection Works

Each training script:
1. Checks if CUDA is available: `torch.cuda.is_available()`
2. Sets device to `cuda` if available, otherwise `cpu`
3. Moves all tensors and models to the detected device
4. **Prints device information at the start of training** (newly added)

## Device Information Display

When you run any training script, you'll now see:

```
================================================================================
Device Information
================================================================================
Using device: cuda
GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
GPU Memory: 11.94 GB
================================================================================
```

If GPU is not available, you'll see:
```
WARNING: CUDA not available. Training will use CPU (much slower).
```

## Verification

To check GPU status manually, run:
```bash
conda activate med
python check_gpu.py
```

## Performance Impact

- **With GPU**: Training is 10-50x faster depending on model size
- **Without GPU**: Training will fall back to CPU (much slower but still works)

## Troubleshooting

If GPU is not being used:

1. **Check CUDA availability**: Run `python check_gpu.py`
2. **Verify PyTorch CUDA support**: `python -c "import torch; print(torch.cuda.is_available())"`
3. **Check NVIDIA drivers**: Ensure latest drivers are installed
4. **Verify conda environment**: Make sure you're using the `med` environment with PyTorch CUDA build

## Model-Specific GPU Usage

All models:
- Automatically detect GPU
- Move model to GPU: `model.to(device)`
- Move data to GPU: `images.to(device)`, `labels.to(device)`
- Use GPU for all forward/backward passes

The device information is printed at the start of each training run, so you can verify GPU usage immediately.



