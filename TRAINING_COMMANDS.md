# Training Commands for All Models

All training scripts have been updated to use the new dataset structure:
- **Dataset Path**: `C:\Tim\Taymaz\dataset2`
- **Structure**: 
  - `dataset2\Train\0` (class 0 training images)
  - `dataset2\Train\1` (class 1 training images)
  - `dataset2\Test\0` (class 0 test images)
  - `dataset2\Test\1` (class 1 test images)

## Quick Start

1. **Activate your conda environment:**
   ```bash
   conda activate med
   ```

2. **Run any model:**
   ```bash
   python train_[modelname]_kfold.py
   ```

## All Available Models

### 1. SqueezeNet
```bash
python train_kfold.py
```
**Results Directory**: `results_squeezenet`

### 2. ResNet-18
```bash
python train_resnet_kfold.py
```
**Results Directory**: `results_resnet18`

### 3. MobileNetV2
```bash
python train_mobilenet_kfold.py
```
**Results Directory**: `results_mobilenet`

### 4. MobileNetV3
```bash
python train_mobilenetv3_kfold.py
```
**Results Directory**: `results_mobilenetv3`

### 5. EfficientNet-B0
```bash
python train_efficientnet_kfold.py
```
**Results Directory**: `results_efficientnet`

### 6. EfficientNetV2-S
```bash
python train_efficientnetv2_kfold.py
```
**Results Directory**: `results_efficientnetv2`

### 7. ConvNeXt-Tiny
```bash
python train_convnext_kfold.py
```
**Results Directory**: `results_convnext`

### 8. DenseNet-121
```bash
python train_densenet_kfold.py
```
**Results Directory**: `results_densenet`

### 9. ShuffleNetV2
```bash
python train_shufflenet_kfold.py
```
**Results Directory**: `results_shufflenet`

### 10. GhostNet-100
```bash
python train_ghostnet_kfold.py
```
**Results Directory**: `results_ghostnet`

### 11. NFNet-F0
```bash
python train_nfnet_kfold.py
```
**Results Directory**: `results_nfnet`

### 12. DeiT-Tiny
```bash
python train_deit_kfold.py
```
**Results Directory**: `results_deit`

## Batch File Commands (Windows)

You can also create batch files for easier execution. Example for GhostNet:

```batch
@echo off
call conda activate med
python train_ghostnet_kfold.py
pause
```

## What Each Training Script Does

1. **Loads data** from `dataset2\Train` and `dataset2\Test`
2. **Trains the model** with early stopping
3. **Evaluates** on the test set
4. **Saves results** including:
   - `best_model.pth` - Best model checkpoint
   - `confusion_matrix.png` - Confusion matrix
   - `roc_curve.png` - ROC curve
   - `pr_curve.png` - Precision-Recall curve
   - `training_history.png` - Training history plots
   - `classification_report.txt` - Detailed classification report
   - `training_summary.txt` - Complete training summary

## Training Configuration

All models use the same configuration:
- **Dataset**: `dataset2` (Train/Test split)
- **Image Size**: 224x224
- **Batch Size**: 16
- **Epochs**: 50 (with early stopping, patience=15)
- **Learning Rate**: 0.01
- **Optimizer**: Adam
- **Scheduler**: Cosine annealing
- **Backbone**: Frozen (only classifier trained)

## Notes

- All scripts now use **Train/Test split** instead of k-fold cross validation
- Results are saved in model-specific directories (e.g., `results_resnet18`)
- Training time will be much faster since we're training only once instead of 20 folds
- Make sure your `dataset2` folder has the correct structure before running




