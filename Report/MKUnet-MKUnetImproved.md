# MK-UNet Binary Classification Results

Successfully adapted the **MK-UNet** (ICCV 2025 CVAMD Oral) architecture from medical image segmentation to a binary classification backbone. We created two variants and trained them on the `dataset2/` provided:

1. **MK-UNet Original (Classifier)**: A direct port of the MK-UNet multi-kernel encoder path followed by a Global Average Pooling and classification head.
2. **MK-UNet Improved**: Enhanced with wider kernels `[1,3,5,7]`, higher expansion factor, GELU activations, Squeeze-and-Excitation (SE) blocks, GeM pooling, Stochastic Depth (DropPath), Label Smoothing, and Mixup augmentation.

## Comparison Table

Both MK-UNet models significantly outperformed the existing baseline models (SqueezeNet ~89%, MobileNetV3 ~92%) on the dataset. The **Improved MK-UNet** demonstrated superior performance across almost all metrics, outperforming the original by **+0.67% F1-score**.

| Metric | MK-UNet Original | MK-UNet Improved | Delta |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 94.03% | **94.63%** | +0.60% |
| **Precision** | **98.80%** | 97.37% | -1.43% |
| **Recall (Sensitivity)**| 90.28% | **92.79%** | +2.51% |
| **F1-Score** | 94.35% | **95.02%** | +0.67% |
| **Specificity** | **98.65%** | 96.91% | -1.74% |
| **ROC-AUC** | 97.86% | **98.62%** | +0.76% |
| **Avg Precision** | 98.58% | **99.05%** | +0.47% |
| **Total Params** | **124K** | 583K | |
| **Best Epoch** | 49 | 73 | |
| **Training Time** | **16.27 min** | 44.41 min | |

## File Outputs

All the assets, training histories, and checkpoints were generated successfully and saved in the output directory:
- **`c:\Tim\Taymaz\final_results\mkunet_comparison\`**

<br>

### Original MK-UNet Confusion Matrix
![Original Confusion Matrix](file:///c:/Tim/Taymaz/final_results/mkunet_comparison/mkunet_original/confusion_matrix.png)

### Improved MK-UNet Confusion Matrix
![Improved Confusion Matrix](file:///c:/Tim/Taymaz/final_results/mkunet_comparison/mkunet_improved/confusion_matrix.png)

<br>

> [!NOTE]
> Training logic, mixup augmentations, and definitions for both models are located in `c:\Tim\Taymaz\model_mkunet.py` and `c:\Tim\Taymaz\train_mkunet_kfold.py`. The Improved model trades off slightly more parameters and longer training time for a tangible boost in Recall and overall F1/ROC-AUC. 
