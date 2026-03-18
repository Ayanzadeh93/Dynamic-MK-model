@echo off
echo ========================================
echo Setting up 'med' conda environment
echo ========================================

call conda activate med

echo.
echo Installing PyTorch with CUDA support...
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

echo.
echo Installing other required packages...
pip install numpy==1.24.3 pandas==2.0.3 matplotlib==3.7.2 seaborn==0.12.2 scikit-learn==1.3.0 Pillow==10.0.0 tqdm==4.66.1

echo.
echo ========================================
echo Installation complete!
echo ========================================
echo.
echo To use this environment, run:
echo   conda activate med
echo.
echo Then start training with:
echo   python train_kfold.py
echo ========================================

