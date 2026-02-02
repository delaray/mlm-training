"""
Quick test to verify model setup works with RTX 5090
Tests the complete model loading and LoRA setup process
"""
import os
import logging
from src.mlm_trainer import setup_model_for_mlm_training, setup_logging

# Setup logging
setup_logging(log_file=None, level=logging.INFO)

print("="*80)
print("RTX 5090 Model Setup Test")
print("="*80)
print()

# Use BERT-base as it's quick to load
model_path = "models/bert-base-uncased"

if not os.path.exists(model_path):
    print(f"❌ Model not found at {model_path}")
    print("   Please download a model first:")
    print("   python download_models.py --model bert-base-uncased")
    exit(1)

print(f"Testing model setup from: {model_path}")
print()

try:
    # Test model setup without quantization (RTX 5090 compatible)
    model, is_quantized = setup_model_for_mlm_training(
        model_name=model_path,
        use_qlora=False,  # Disabled for RTX 5090
        use_lora=True,    # LoRA works great
        lora_r=8,         # Small rank for quick test
        lora_alpha=16,
        load_in_4bit=False,
        load_in_8bit=False
    )
    
    print()
    print("="*80)
    print("✅ SUCCESS! Model setup completed without errors")
    print("="*80)
    print()
    print(f"Model type: {type(model).__name__}")
    print(f"Quantized: {is_quantized}")
    print(f"Device: {next(model.parameters()).device}")
    print(f"Dtype: {next(model.parameters()).dtype}")
    print()
    print("⚠️  NOTE: RTX 5090 sm_120 requires CPU training")
    print("   GPU support will be available when PyTorch adds full sm_120 kernels")
    print()
    print("🎉 Your system is ready for CPU-based training!")
    print()
    
except Exception as e:
    print()
    print("="*80)
    print("❌ ERROR during model setup")
    print("="*80)
    print()
    print(f"Error: {e}")
    print()
    import traceback
    traceback.print_exc()
    exit(1)
