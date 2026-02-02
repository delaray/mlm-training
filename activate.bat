@echo off
REM Activation script for MLM Training project
REM This ensures you're using the correct venv with CUDA-enabled PyTorch

echo.
echo ================================================================================
echo MLM Training Project - Environment Activation
echo ================================================================================
echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo.
echo Checking CUDA availability...
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

echo.
echo ================================================================================
echo Environment ready! You can now run:
echo   python test_setup.py
echo   python example_mlm_training.py
echo   python download_models.py
echo ================================================================================
echo.
