@echo off
REM Install PyTorch with CUDA support
REM For RTX 5090 (sm_120): Uses PyTorch nightly with CUDA 12.6

echo.
echo ================================================================================
echo Installing PyTorch Nightly with CUDA 12.6 Support
echo (Required for RTX 5090 / Blackwell architecture)
echo ================================================================================
echo.

echo Removing any existing PyTorch installations...
call uv pip uninstall -y torch torchaudio torchvision 2>nul

echo.
echo Installing PyTorch nightly with CUDA 12.6...
call uv pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu126 torch torchvision torchaudio

echo.
echo Verifying installation...
.venv\Scripts\python.exe -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

echo.
echo ================================================================================
echo Installation complete!
echo NOTE: You may see warnings about sm_120 compatibility - this is expected.
echo The GPU will work in compatibility mode.
echo ================================================================================
echo.
