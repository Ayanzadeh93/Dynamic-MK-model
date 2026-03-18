# System Architecture

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Medical Image Dataset                         │
│           (905 images: 607 Class 0, 298 Class 1)                │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Data Loading & Preprocessing                   │
│  - Load from directory structure (dataset/0/, dataset/1/)       │
│  - Apply general augmentations (rotation, flip, color jitter)   │
│  - Normalize with ImageNet statistics                           │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              20-Fold Stratified Cross Validation                 │
│  - Split data into 20 folds (stratified by class)              │
│  - Train and validate on each fold independently                │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SqueezeNet Model                              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  ImageNet Pretrained Backbone (Feature Extraction)       │  │
│  │  - SqueezeNet1.1 architecture                            │  │
│  │  - ~1.2M parameters                                       │  │
│  └───────────────────┬───────────────────────────────────────┘  │
│                      │                                           │
│                      ▼                                           │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Custom Binary Classifier                                 │  │
│  │  - Dropout (0.5)                                          │  │
│  │  - Conv2d (512 → 2)                                       │  │
│  │  - AdaptiveAvgPool2d                                      │  │
│  └───────────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Training Process                            │
│  - Loss: CrossEntropyLoss (class weighted)                      │
│  - Optimizer: Adam (lr=0.001, weight_decay=1e-4)                │
│  - Scheduler: CosineAnnealingLR                                 │
│  - Early Stopping: patience=15                                  │
│  - Max Epochs: 50 per fold                                      │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Evaluation & Metrics                             │
│  - Accuracy, Precision, Recall, F1-Score                        │
│  - Specificity, Sensitivity, ROC-AUC                            │
│  - Confusion Matrix, ROC Curve, PR Curve                        │
│  - Training history plots                                       │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Results & Outputs                             │
│  - 20 model checkpoints (best per fold)                         │
│  - Aggregate metrics (mean ± std)                               │
│  - Comprehensive visualizations                                 │
│  - Classification reports                                       │
└─────────────────────────────────────────────────────────────────┘
```

## Module Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                       train_kfold.py                            │
│                    (Main Training Script)                       │
│  - Orchestrates entire training pipeline                       │
│  - Manages k-fold cross validation loop                        │
│  - Handles checkpointing and logging                           │
└───┬─────────────┬─────────────┬─────────────┬─────────────────┘
    │             │             │             │
    ▼             ▼             ▼             ▼
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌────────────────────┐
│model.py │  │dataset  │  │evaluati-│  │    inference.py    │
│         │  │.py      │  │on_metri │  │                    │
│SqueezeN │  │         │  │cs.py    │  │Prediction Engine   │
│et Model │  │Data Load│  │         │  │- Single image      │
│         │  │& Augment│  │Reusable │  │- Batch inference   │
│- Create │  │         │  │Metrics  │  │- Directory scan    │
│- Train  │  │- Load   │  │         │  │                    │
│- Optim. │  │- Trans. │  │- Calc   │  │                    │
│- Sched. │  │- Batch  │  │- Plot   │  │                    │
└─────────┘  └─────────┘  │- Save   │  └────────────────────┘
                          └─────────┘
```

## Data Flow

```
Input Images (.tif)
       │
       ▼
┌─────────────────┐
│  Image Loader   │
│  (PIL/Pillow)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Augmentation    │
│ - Resize(224)   │
│ - RandomRotate  │
│ - RandomFlip    │
│ - ColorJitter   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Normalization   │
│ (ImageNet Stats)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Tensor(3,224,224│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ DataLoader      │
│ (Batching)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Forward   │
└────────┬────────┘
         │
         ▼
   Predictions
```

## Training Loop Architecture

```
For each fold (1 to 20):
  │
  ├─ Split data (train/val)
  │
  ├─ Create model (pretrained SqueezeNet)
  │
  ├─ Setup optimizer & scheduler
  │
  ├─ For each epoch (1 to 50):
  │   │
  │   ├─ Training Phase:
  │   │   ├─ For each batch:
  │   │   │   ├─ Forward pass
  │   │   │   ├─ Calculate loss
  │   │   │   ├─ Backward pass
  │   │   │   └─ Update weights
  │   │   └─ Calculate avg train loss & accuracy
  │   │
  │   ├─ Validation Phase:
  │   │   ├─ For each batch:
  │   │   │   ├─ Forward pass (no grad)
  │   │   │   └─ Calculate loss
  │   │   └─ Calculate avg val loss & accuracy
  │   │
  │   ├─ Update learning rate (scheduler)
  │   │
  │   ├─ Save checkpoint if best val accuracy
  │   │
  │   └─ Check early stopping
  │
  ├─ Load best model for fold
  │
  ├─ Evaluate on validation set
  │
  ├─ Generate plots & reports
  │
  └─ Save fold metrics

Aggregate all fold results
Generate summary plots & reports
Save final statistics
```

## File Organization

```
Project Root
│
├── Core Training
│   ├── train_kfold.py          (Main training orchestrator)
│   ├── model.py                (SqueezeNet architecture)
│   ├── dataset.py              (Data loading & augmentation)
│   └── evaluation_metrics.py   (Metrics calculation & plotting)
│
├── Utilities
│   ├── inference.py            (Run predictions)
│   ├── test_setup.py           (Verify installation)
│   ├── quick_start.py          (One-command launcher)
│   └── example_use_metrics.py  (Usage examples)
│
├── Documentation
│   ├── README.md               (Full documentation)
│   ├── GETTING_STARTED.md      (Quick start guide)
│   ├── PROJECT_SUMMARY.md      (Project overview)
│   ├── ARCHITECTURE.md         (This file)
│   └── requirements.txt        (Dependencies)
│
├── Data
│   └── dataset/
│       ├── 0/                  (Class 0 images)
│       └── 1/                  (Class 1 images)
│
└── Results (generated during training)
    └── results_squeezenet_kfold20/
        ├── config.json
        ├── fold_metrics.csv
        ├── metrics_statistics.csv
        ├── training_summary.txt
        ├── *.png (visualization plots)
        ├── best_model_fold_*.pth
        └── fold_*/
            ├── confusion_matrix.png
            ├── roc_curve.png
            ├── pr_curve.png
            ├── training_history.png
            └── classification_report.txt
```

## Model Architecture Detail

```
SqueezeNet1.1 Binary Classifier
│
├── Features (from pretrained ImageNet)
│   ├── Conv2d(3, 64, kernel=3, stride=2)
│   ├── ReLU
│   ├── MaxPool2d(kernel=3, stride=2)
│   │
│   ├── Fire Module 1
│   ├── Fire Module 2
│   ├── Fire Module 3
│   ├── MaxPool2d
│   │
│   ├── Fire Module 4
│   ├── Fire Module 5
│   ├── Fire Module 6
│   ├── Fire Module 7
│   ├── MaxPool2d
│   │
│   ├── Fire Module 8
│   └── Fire Module 9
│       └── Output: (batch, 512, h, w)
│
└── Classifier (custom for binary classification)
    ├── Dropout(p=0.5)
    ├── Conv2d(512, 2, kernel=1)
    ├── AdaptiveAvgPool2d((1, 1))
    └── Flatten → Output: (batch, 2)

Total Parameters: ~1,235,496
Trainable Parameters: ~1,235,496 (or ~10,000 if backbone frozen)
```

## Metrics Calculation Pipeline

```
Model Predictions
       │
       ▼
┌───────────────────┐
│ Get Probabilities │ ← softmax(outputs)
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Get Predictions   │ ← argmax(outputs)
└─────────┬─────────┘
          │
          ▼
┌───────────────────────────────────────┐
│ Calculate Metrics                     │
│ - Accuracy   = (TP + TN) / Total      │
│ - Precision  = TP / (TP + FP)         │
│ - Recall     = TP / (TP + FN)         │
│ - F1-Score   = 2 * (P * R) / (P + R)  │
│ - Specificity= TN / (TN + FP)         │
│ - ROC-AUC    = area under ROC curve   │
└─────────┬─────────────────────────────┘
          │
          ▼
┌───────────────────┐
│ Generate Plots    │
│ - Confusion Matrix│
│ - ROC Curve       │
│ - PR Curve        │
│ - Training History│
└───────────────────┘
```

## Key Design Decisions

### 1. Why SqueezeNet?
- Lightweight (small memory footprint)
- Fast inference
- Good accuracy with limited data
- Proven pretrained weights

### 2. Why 20-Fold CV?
- Small dataset (905 images)
- More folds = more robust evaluation
- Better use of limited data
- Reliable confidence intervals

### 3. Why ImageNet Pretraining?
- Transfer learning from large dataset
- Better feature extraction
- Faster convergence
- Higher accuracy with small data

### 4. Why These Augmentations?
- Medical imaging specific
- Preserve diagnostic features
- Minimal distortion
- Rotation-invariant (tissue orientation)

### 5. Why Class Weighting?
- Handle class imbalance (607 vs 298)
- Prevent bias toward majority class
- Better minority class performance

## Extensibility Points

To customize or extend the framework:

1. **Add New Model**: Modify `model.py`
2. **Change Augmentations**: Edit `dataset.py` → `get_train_transforms()`
3. **Add Metrics**: Extend `evaluation_metrics.py` → `calculate_metrics()`
4. **Modify Training**: Update `train_kfold.py` → `train_one_epoch()`
5. **Change Folds**: Edit config in `train_kfold.py` → `num_folds`

## Performance Optimization

```
Speed Optimization Path:
┌──────────────────────────────────┐
│ Use GPU (5-10x speedup)          │
├──────────────────────────────────┤
│ Increase batch_size (2x speedup) │
├──────────────────────────────────┤
│ Reduce num_folds (linear speedup)│
├──────────────────────────────────┤
│ Freeze backbone (2x speedup)     │
├──────────────────────────────────┤
│ Reduce num_epochs (linear)       │
└──────────────────────────────────┘

Memory Optimization Path:
┌──────────────────────────────────┐
│ Reduce batch_size                │
├──────────────────────────────────┤
│ Freeze backbone                  │
├──────────────────────────────────┤
│ Reduce image_size (224→128)      │
├──────────────────────────────────┤
│ Use mixed precision training     │
└──────────────────────────────────┘
```

---

**This architecture ensures:**
- Modularity: Each component is independent
- Reusability: Evaluation metrics work with any model
- Scalability: Easy to extend to more classes or models
- Maintainability: Clear separation of concerns
- Robustness: Comprehensive error handling and validation




