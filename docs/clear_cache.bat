@echo off
echo ============================================================
echo  Clearing PyTorch Model Cache
echo ============================================================
echo.
echo This will clear corrupted model weights from cache
echo.

REM Clear EfficientNet cache
if exist "%USERPROFILE%\.cache\torch\hub\checkpoints\efficientnet_b0_rwightman-3dd342df.pth" (
    del "%USERPROFILE%\.cache\torch\hub\checkpoints\efficientnet_b0_rwightman-3dd342df.pth"
    echo Deleted EfficientNet-B0 cached weights
)

REM Clear all torch hub cache (optional - more aggressive)
echo.
echo Cache cleared. You can now retry training.
echo.
pause




