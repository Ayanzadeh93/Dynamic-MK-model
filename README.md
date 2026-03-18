
# Dynamic-MK-model

# SqueezeNet Binary Classification with K-Fold Cross Validation

A comprehensive deep learning framework for binary classification of medical images using SqueezeNet with ImageNet pretrained backbone and k-fold cross validation.

## Overview

This project implements a robust binary classification system specifically designed for medical imaging datasets with:

- **SqueezeNet Architecture**: Lightweight CNN with ImageNet pretrained weights
- **K-Fold Cross Validation**: 20-fold stratified cross-validation for robust evaluation
- **Medical Image Augmentation**: Conservative augmentations appropriate for medical imaging
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Specificity, and more
- **Reusable Evaluation Module**: Can be imported and used with other models

## Dataset Structure

```
dataset/
├── 0/              # Class 0 images (607 images)
│   ├── image1.tif
│   ├── image2.tif
│   └── ...
├── 1/              # Class 1 images (298 images)
│   ├── image1.tif
│   ├── image2.tif
│   └── ...
└── requirments.txt
```

Total: **905 medical images** (607 Class 0, 298 Class 1)

## Project Structure

```
.
├── train_kfold.py           # Main training script with k-fold CV
├── model.py                 # SqueezeNet model implementation
├── dataset.py               # Dataset loader and augmentations
├── evaluation_metrics.py    # Reusable evaluation metrics module
├── inference.py             # Inference script for predictions
├── README.md               # This file
└── requirements.txt        # Python dependencies
```

## Features

### Model Architecture
- **SqueezeNet1.1** with ImageNet pretrained backbone
- Binary classification (2 classes)
- Dropout regularization
- Adaptive classifier for medical imaging

### Data Augmentation
Conservative augmentations suitable for medical images:
- Random rotation (±15 degrees)
- Horizontal and vertical flips
- Color jitter (brightness, contrast, saturation)
- ImageNet normalization

### Training Features
- **K-Fold Cross Validation**: 20-fold stratified splitting
- **Class Balancing**: Weighted loss function for imbalanced classes
- **Early Stopping**: Configurable patience
- **Learning Rate Scheduling**: Cosine annealing, step decay, or plateau
- **Model Checkpointing**: Saves best model for each fold

### Evaluation Metrics
- Accuracy
- Precision
- Recall (Sensitivity)
- F1-Score
- Specificity
- ROC-AUC
- Average Precision

### Visualization
- Confusion matrices
- ROC curves
- Precision-Recall curves
- Training history plots
- K-fold metrics comparison
- Metrics boxplots

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Ensure your dataset is organized in the correct structure (see Dataset Structure above)

## Usage

### Training with K-Fold Cross Validation

Run the main training script:

```bash
python train_kfold.py
```

The training will:
1. Load the dataset from `dataset/` directory
2. Perform 20-fold stratified cross-validation
3. Train a SqueezeNet model for each fold
4. Evaluate and save metrics for each fold
5. Generate comprehensive plots and reports
6. Save all results to `results_squeezenet_kfold20/` directory

### Configuration

You can modify the configuration in `train_kfold.py`:

```python
config = {
    # Dataset
    'data_dir': 'dataset',
    'num_classes': 2,
    'image_size': 224,
    
    # K-Fold
    'num_folds': 20,
    'random_seed': 42,
    
    # Model
    'pretrained': True,
    'dropout': 0.5,
    'freeze_backbone': False,
    
    # Training
    'num_epochs': 50,
    'batch_size': 16,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'optimizer': 'adam',
    'scheduler': 'cosine',
    
    # Early stopping
    'early_stopping': True,
    'patience': 15,
}
```

### Inference

Run inference on new images:

**Single Image:**
```bash
python inference.py --model results_squeezenet_kfold20/best_model_fold_1.pth --input path/to/image.tif
```

**Directory of Images:**
```bash
python inference.py --model results_squeezenet_kfold20/best_model_fold_1.pth --input path/to/images/
```

**Options:**
- `--model`: Path to trained model checkpoint
- `--input`: Path to image file or directory
- `--output`: Output directory for results (default: inference_results)
- `--device`: Device to use (cuda or cpu)

### Using Evaluation Metrics with Other Models

The evaluation metrics module is designed to be reusable:

```python
from evaluation_metrics import BinaryClassificationMetrics, evaluate_model

# Create metrics calculator
metrics_calc = BinaryClassificationMetrics(save_dir='my_results')

# Calculate metrics
metrics = metrics_calc.calculate_metrics(y_true, y_pred, y_pred_proba)

# Generate plots
metrics_calc.plot_confusion_matrix(y_true, y_pred)
metrics_calc.plot_roc_curve(y_true, y_pred_proba)
metrics_calc.plot_precision_recall_curve(y_true, y_pred_proba)

# For k-fold results
metrics_calc.plot_kfold_metrics(fold_metrics)
metrics_calc.plot_metrics_boxplot(fold_metrics)
metrics_calc.save_metrics_summary(fold_metrics)
```

## Output Files

After training, the following files will be generated in `results_squeezenet_kfold20/`:

### Main Results
- `fold_metrics.csv`: Detailed metrics for each fold
- `metrics_statistics.csv`: Mean and standard deviation of all metrics
- `training_summary.txt`: Complete training summary
- `config.json`: Configuration used for training

### Visualizations
- `kfold_metrics_comparison.png`: Line plots of metrics across folds
- `metrics_boxplot.png`: Boxplot distribution of all metrics

### Per-Fold Results (in `fold_X/` subdirectories)
- `confusion_matrix.png`: Confusion matrix
- `roc_curve.png`: ROC curve
- `pr_curve.png`: Precision-Recall curve
- `training_history.png`: Training and validation loss/accuracy
- `classification_report.txt`: Detailed classification report
- `best_model_fold_X.pth`: Best model checkpoint for the fold

## Performance Metrics

The system tracks and reports:

- **Per-Fold Metrics**: Individual performance for each of the 20 folds
- **Aggregate Statistics**: Mean ± Standard Deviation across all folds
- **Best Model Selection**: Saves the best model based on validation accuracy

## Hardware Requirements

- **GPU**: CUDA-capable GPU recommended (but CPU works too)
- **RAM**: At least 8GB recommended
- **Storage**: ~500MB for model checkpoints and results

## Tips for Medical Imaging

1. **Small Dataset Handling**: The system uses:
   - Pretrained ImageNet weights for transfer learning
   - Class weighting to handle imbalance
   - Conservative augmentation to preserve medical features

2. **Validation Strategy**: 20-fold CV ensures:
   - Robust performance estimation
   - Better use of limited data
   - Reliable confidence intervals

3. **Model Selection**: SqueezeNet is chosen because:
   - Lightweight (fewer parameters)
   - Good performance with limited data
   - Fast training and inference
   - Proven ImageNet pretrained weights

## Extending to Other Models

To use this framework with different architectures:

1. Modify `model.py` to include your architecture
2. Update `create_model()` function in `train_kfold.py`
3. The rest of the pipeline (data loading, training, evaluation) remains the same

## Troubleshooting

**Out of Memory Error:**
- Reduce `batch_size` in config
- Reduce `image_size` to 128 or 112
- Use fewer `num_workers`

**Slow Training:**
- Ensure CUDA is available: `torch.cuda.is_available()`
- Reduce `num_folds` for faster experimentation
- Enable `freeze_backbone` to train only classifier

**Poor Performance:**
- Increase `num_epochs`
- Adjust `learning_rate` (try 0.0001 or 0.01)
- Enable/disable `freeze_backbone`
- Modify augmentation strength

## Citation

If you use this code, please cite:

```
SqueezeNet Binary Classification Framework
Medical Image Analysis with K-Fold Cross Validation
2024
```

## License

This project is open source and available under the MIT License.

## Contact

For questions or issues, please open an issue in the repository.

---

**Note**: This framework is specifically designed for binary classification of medical images with small datasets. The conservative augmentation strategy and robust cross-validation ensure reliable performance even with limited training data.


