@echo off
echo ============================================================
echo  NFNet-F0 Training - Medical Image Classification (timm)
echo ============================================================
echo.
echo Dataset: 408 images (255 Class 0, 153 Class 1)
echo Configuration: 20-Fold Cross Validation
echo Model: NFNet-F0 with ImageNet pretrained weights (timm)
echo.
echo ============================================================
echo  Activating 'med' environment and starting training...
echo ============================================================
echo.

call conda activate med
python train_nfnet_kfold.py

echo.
echo ============================================================
echo  Training Complete!
echo ============================================================
echo.
echo Check results in: results_nfnet_kfold20\
echo.
pause

