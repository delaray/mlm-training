"""Quick verification that all imports work correctly"""
import torch
from src.mlm_trainer import (
    prepare_mlm_dataset,
    setup_model_for_mlm_training,
    train_mlm_model,
    save_trained_model,
    generate_embeddings
)
from transformers import Trainer, TrainingArguments, AutoModelForMaskedLM
from peft import LoraConfig, get_peft_model

print('=' * 80)
print('IMPORT TEST - All Critical Functions')
print('=' * 80)
print()

print('PyTorch:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('GPU Memory:', round(torch.cuda.get_device_properties(0).total_memory/1e9, 1), 'GB')
print()

print('✓ All imports successful!')
print('✓ mlm_trainer module loaded')
print('✓ transformers library loaded')
print('✓ peft library loaded')
print()
print('🎉 Ready to run example_mlm_training.py!')
print('=' * 80)
