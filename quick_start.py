"""
Quick Start Script - Simplified training with default settings
For users who want to start training immediately
"""

import os
import sys

def check_requirements():
    """Check if all requirements are installed"""
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
        return True
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("\nPlease install requirements:")
        print("  pip install -r requirements.txt")
        return False

def check_dataset():
    """Check if dataset exists"""
    if not os.path.exists('dataset'):
        print("Error: 'dataset' directory not found")
        return False
    
    if not os.path.exists('dataset/0') or not os.path.exists('dataset/1'):
        print("Error: Dataset must have subdirectories '0' and '1' for each class")
        return False
    
    return True

def main():
    """Main quick start function"""
    print("="*70)
    print(" SqueezeNet K-Fold Cross Validation - Quick Start")
    print("="*70)
    
    # Check requirements
    print("\n[1/3] Checking requirements...")
    if not check_requirements():
        sys.exit(1)
    print("[OK] All requirements satisfied")
    
    # Check dataset
    print("\n[2/3] Checking dataset...")
    if not check_dataset():
        sys.exit(1)
    print("[OK] Dataset found")
    
    # Import and run training
    print("\n[3/3] Starting training...")
    print("-"*70)
    
    try:
        from train_kfold import main as train_main
        train_main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()

