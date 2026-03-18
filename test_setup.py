"""
Quick setup test to verify all components work correctly
"""

import os
import sys
import torch

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        import torchvision
        import numpy
        import pandas
        import matplotlib
        import seaborn
        import sklearn
        import PIL
        import tqdm
        print("[OK] All required packages imported successfully")
        return True
    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA...")
    
    if torch.cuda.is_available():
        print(f"[OK] CUDA is available")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
    else:
        print("[WARNING] CUDA not available, will use CPU")
        print("  Training will be slower on CPU")

def test_dataset():
    """Test if dataset directory exists and is properly structured"""
    print("\nTesting dataset...")
    
    dataset_dir = 'dataset'
    if not os.path.exists(dataset_dir):
        print(f"[ERROR] Dataset directory not found: {dataset_dir}")
        return False
    
    class0_dir = os.path.join(dataset_dir, '0')
    class1_dir = os.path.join(dataset_dir, '1')
    
    if not os.path.exists(class0_dir):
        print(f"[ERROR] Class 0 directory not found: {class0_dir}")
        return False
    
    if not os.path.exists(class1_dir):
        print(f"[ERROR] Class 1 directory not found: {class1_dir}")
        return False
    
    # Count images
    class0_images = len([f for f in os.listdir(class0_dir) 
                        if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))])
    class1_images = len([f for f in os.listdir(class1_dir) 
                        if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))])
    
    print(f"[OK] Dataset structure is correct")
    print(f"  Class 0: {class0_images} images")
    print(f"  Class 1: {class1_images} images")
    print(f"  Total: {class0_images + class1_images} images")
    
    if class0_images == 0 or class1_images == 0:
        print("[WARNING] One class has no images")
        return False
    
    return True

def test_modules():
    """Test if custom modules can be imported"""
    print("\nTesting custom modules...")
    
    try:
        from model import create_squeezenet_model
        print("[OK] model.py imported successfully")
    except Exception as e:
        print(f"[ERROR] Error importing model.py: {e}")
        return False
    
    try:
        from dataset import get_image_paths_and_labels
        print("[OK] dataset.py imported successfully")
    except Exception as e:
        print(f"[ERROR] Error importing dataset.py: {e}")
        return False
    
    try:
        from evaluation_metrics import BinaryClassificationMetrics
        print("[OK] evaluation_metrics.py imported successfully")
    except Exception as e:
        print(f"[ERROR] Error importing evaluation_metrics.py: {e}")
        return False
    
    return True

def test_model_creation():
    """Test if model can be created"""
    print("\nTesting model creation...")
    
    try:
        from model import create_squeezenet_model
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = create_squeezenet_model(
            pretrained=True,
            num_classes=2,
            device=device
        )
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        
        assert output.shape == (1, 2), f"Expected output shape (1, 2), got {output.shape}"
        
        print("[OK] Model created and tested successfully")
        print(f"  Output shape: {output.shape}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Error creating/testing model: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("SqueezeNet Setup Verification")
    print("="*60)
    
    all_passed = True
    
    # Run tests
    all_passed &= test_imports()
    test_cuda()  # CUDA test doesn't affect pass/fail
    all_passed &= test_dataset()
    all_passed &= test_modules()
    all_passed &= test_model_creation()
    
    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("[SUCCESS] ALL TESTS PASSED")
        print("="*60)
        print("\nYou can now run training with:")
        print("  python train_kfold.py")
    else:
        print("[FAILED] SOME TESTS FAILED")
        print("="*60)
        print("\nPlease fix the issues above before training")
        sys.exit(1)

if __name__ == '__main__':
    main()

