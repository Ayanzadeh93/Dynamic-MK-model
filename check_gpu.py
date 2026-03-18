"""
Check GPU availability and PyTorch CUDA support
"""
import torch
import sys

print("="*80)
print("GPU/CUDA Availability Check")
print("="*80)

# PyTorch version
print(f"\nPyTorch version: {torch.__version__}")

# CUDA availability
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print(f"  Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
    
    # Test tensor creation on GPU
    try:
        test_tensor = torch.randn(10, 10).cuda()
        print(f"\n[OK] Successfully created tensor on GPU")
        print(f"  Tensor device: {test_tensor.device}")
        del test_tensor
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"\n[ERROR] Failed to create tensor on GPU: {e}")
else:
    print("\n[WARNING] CUDA is not available. Training will use CPU (much slower).")
    print("  To use GPU, ensure:")
    print("  1. You have an NVIDIA GPU")
    print("  2. CUDA drivers are installed")
    print("  3. PyTorch was installed with CUDA support")
    print("  4. GPU is properly detected by the system")

print("\n" + "="*80)

# Check what device will be used by training scripts
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training scripts will use: {device}")
print("="*80)

