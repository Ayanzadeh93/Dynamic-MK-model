@echo off
echo ============================================================
echo  DeiT-Tiny Training - Medical Image Classification (timm)
echo ============================================================
echo.
echo Dataset: 408 images (255 Class 0, 153 Class 1)
echo Configuration: 20-Fold Cross Validation
echo Model: DeiT-Tiny (Vision Transformer) with ImageNet pretrained weights (timm)
echo.
echo ============================================================
echo  Activating 'med' environment and starting training...
echo ============================================================
echo.

call conda activate med
python train_deit_kfold.py

echo.
echo ============================================================
echo  Training Complete!
echo ============================================================
echo.
echo Check results in: results_deit_kfold20\
echo.
pause

