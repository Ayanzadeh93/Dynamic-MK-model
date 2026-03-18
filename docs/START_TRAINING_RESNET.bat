@echo off
echo ============================================================
echo  ResNet-18 Training - Medical Image Classification
echo ============================================================
echo.
echo Dataset: 408 images (255 Class 0, 153 Class 1)
echo Configuration: 20-Fold Cross Validation
echo Model: ResNet-18 with ImageNet pretrained weights
echo.
echo ============================================================
echo  Activating 'med' environment and starting training...
echo ============================================================
echo.

call conda activate med
python train_resnet_kfold.py

echo.
echo ============================================================
echo  Training Complete!
echo ============================================================
echo.
echo Check results in: results_resnet18_kfold20\
echo.
pause

