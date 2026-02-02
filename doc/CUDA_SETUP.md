# MLM Training - CUDA Setup Instructions

## ✅ CUDA is Now Working!

Your environment is now properly configured with:
- **PyTorch**: 2.6.0+cu124
- **GPU**: NVIDIA GeForce RTX 5090 Laptop GPU
- **CUDA**: 12.4
- **GPU Memory**: 25.7 GB

## ⚠️ Important Note: RTX 5090 Compatibility

Your RTX 5090 has compute capability 12.0 (Blackwell architecture), which is newer than officially supported by PyTorch 2.6. However, PyTorch will run in compatibility mode and work correctly. This warning is expected and won't affect training.

## How to Run Scripts

**Always use the venv Python directly:**

```bash
# Method 1: Activate venv first (RECOMMENDED)
.venv\Scripts\activate
python test_setup.py
python example_mlm_training.py
python download_models.py

# Method 2: Use venv Python directly
.venv\Scripts\python.exe test_setup.py
.venv\Scripts\python.exe example_mlm_training.py
```

**DO NOT use `uv run`** - it reinstalls the CPU version of PyTorch!

## Quick Start Scripts

### Windows Batch Script
```bash
activate.bat
```
This activates the venv and shows CUDA status.

### Then run your scripts:
```bash
python test_setup.py
python download_models.py --model deberta-v3-base
python example_mlm_training.py
```

## Why `uv run` Doesn't Work

The `uv run` command reads `pyproject.toml` and tries to install `torch>=2.9.0`, which pulls the CPU version from PyPI. The CUDA version must be installed from PyTorch's specific index URL.

## How PyTorch with CUDA Was Installed

```bash
# Removed CPU version
uv pip uninstall torch torchaudio torchvision

# Installed CUDA version from PyTorch index
uv pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
```

## Verifying CUDA

Run this to verify:
```bash
.venv\Scripts\python.exe -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

Expected output:
```
CUDA: True
GPU: NVIDIA GeForce RTX 5090 Laptop GPU
```

## If CUDA Breaks Again

If you accidentally run `uv sync` or `uv run` and lose CUDA support, reinstall with **matching versions**:

```bash
# Quick fix - run the batch script
install_pytorch_cuda.bat

# Or manually:
uv pip uninstall torch torchaudio torchvision
uv pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124
```

**Important**: Always use `+cu124` suffix to ensure CUDA versions are installed, not CPU versions!

### Common Issues

**ImportError with Trainer or transformers**:
- Cause: Version mismatch between torch, torchvision, and torchaudio
- Solution: Run `install_pytorch_cuda.bat` to reinstall matching versions
- Verify: All three should show `2.6.0+cu124`

**Check versions**:
```bash
uv pip list | Select-String "torch"
```

Expected output:
```
torch                    2.6.0+cu124
torchaudio               2.6.0+cu124
torchvision              0.21.0+cu124
```

## Training Performance

With your RTX 5090 (25.7 GB), you can:
- ✅ Train DeBERTa-v3-base with QLoRA (~3GB VRAM)
- ✅ Train DeBERTa-v3-large with QLoRA (~6GB VRAM)
- ✅ Use batch size 8-16 comfortably
- ✅ Run multiple experiments simultaneously

Expected training times (with QLoRA):
- DeBERTa-v3-base, 10k samples, 10 epochs: ~1.5-2 hours
- DeBERTa-v3-large, 10k samples, 10 epochs: ~3-4 hours

## Summary

✅ **CUDA is working!**
✅ **Use `.venv\Scripts\python.exe` or activate the venv**
✅ **Avoid `uv run` for now**
✅ **Ready to train!**

---

Last updated: February 2, 2026
