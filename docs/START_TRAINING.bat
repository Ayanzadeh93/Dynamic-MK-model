@echo off
echo ============================================================
echo  SqueezeNet Training - Medical Image Classification
echo ============================================================
echo.
echo Dataset: 408 images (255 Class 0, 153 Class 1)
echo Configuration: 10-Fold Cross Validation
echo Model: SqueezeNet with ImageNet pretrained weights
echo.
echo ============================================================
echo  Activating 'med' environment and starting training...
echo ============================================================
echo.

call conda activate med
python train_kfold.py

echo.
echo ============================================================
echo  Training Complete!
echo ============================================================
echo.
echo Check results in: results_squeezenet_kfold10\
echo.
pause

