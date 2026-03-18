@echo off
REM ============================================================================
REM Train All Models - Batch Script for Windows
REM This script runs all model training scripts sequentially on dataset2
REM Results will be saved in final_results/[model_name]/ directories
REM ============================================================================

echo ============================================================================
echo TRAINING ALL MODELS ON DATASET2
echo ============================================================================
echo.
echo This will train all 12 models sequentially.
echo Results will be saved in final_results/[model_name]/ directories
echo.
echo Press Ctrl+C to cancel, or
pause

REM Activate conda environment (adjust if your environment name is different)
call conda activate med
if errorlevel 1 (
    echo ERROR: Could not activate conda environment 'med'
    echo Please make sure conda is installed and the environment exists.
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo Starting training...
echo ============================================================================
echo.

REM Create final_results directory
if not exist "final_results" mkdir final_results


REM Model 6: MobileNet
echo [6/12] Training MobileNet...
echo ----------------------------------------------------------------------------
python train_mobilenet_kfold.py
if errorlevel 1 (
    echo ERROR: MobileNet training failed!
    pause
    exit /b 1
)
echo.
echo [6/12] MobileNet training completed!
echo.

REM Model 7: MobileNetV3
echo [7/12] Training MobileNetV3...
echo ----------------------------------------------------------------------------
python train_mobilenetv3_kfold.py
if errorlevel 1 (
    echo ERROR: MobileNetV3 training failed!
    pause
    exit /b 1
)
echo.
echo [7/12] MobileNetV3 training completed!
echo.

REM Model 9: DenseNet
echo [9/12] Training DenseNet...
echo ----------------------------------------------------------------------------
python train_densenet_kfold.py
if errorlevel 1 (
    echo ERROR: DenseNet training failed!
    pause
    exit /b 1
)
echo.
echo [9/12] DenseNet training completed!
echo.

REM Model 10: ShuffleNet
echo [10/12] Training ShuffleNet...
echo ----------------------------------------------------------------------------
python train_shufflenet_kfold.py
if errorlevel 1 (
    echo ERROR: ShuffleNet training failed!
    pause
    exit /b 1
)
echo.
echo [10/12] ShuffleNet training completed!
echo.

REM Model 11: NFNet
echo [11/12] Training NFNet...
echo ----------------------------------------------------------------------------
python train_nfnet_kfold.py
if errorlevel 1 (
    echo ERROR: NFNet training failed!
    pause
    exit /b 1
)
echo.
echo [11/12] NFNet training completed!
echo.

REM Model 12: DeiT
echo [12/12] Training DeiT...
echo ----------------------------------------------------------------------------
python train_deit_kfold.py
if errorlevel 1 (
    echo ERROR: DeiT training failed!
    pause
    exit /b 1
)
echo.
echo [12/12] DeiT training completed!
echo.

REM Model 8: ConvNext
echo [8/12] Training ConvNext...
echo ----------------------------------------------------------------------------
python train_convnext_kfold.py
if errorlevel 1 (
    echo ERROR: ConvNext training failed!
    pause
    exit /b 1
)
echo.
echo [8/12] ConvNext training completed!
echo.



echo ============================================================================
echo ALL MODELS TRAINING COMPLETED SUCCESSFULLY!
echo ============================================================================
echo.
echo Results saved in:
echo   final_results/resnet18/
echo   final_results/squeezenet/
echo   final_results/ghostnet/
echo   final_results/efficientnet/
echo   final_results/efficientnetv2/
echo   final_results/mobilenet/
echo   final_results/mobilenetv3/
echo   final_results/convnext/
echo   final_results/densenet/
echo   final_results/shufflenet/
echo   final_results/nfnet/
echo   final_results/deit/
echo.
pause




