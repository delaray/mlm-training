# MLM Training Project

Complete pipeline for training encoder models using Masked Language Modeling (MLM) with PEFT optimization (LoRA/QLoRA).

## 🚀 Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Install PyTorch with CUDA support (IMPORTANT!)
install_pytorch_cuda.bat
# Or manually:
# uv pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124

# 3. Activate virtual environment
.venv\Scripts\activate

# 4. Verify setup
python test_imports.py

# 5. Download a model
python download_models.py --model deberta-v3-base

# 6. Add your PDF books to data/books/

# 7. Start training
python example_mlm_training.py
```

> ⚠️ **Critical**: Always use matching CUDA versions with `+cu124` suffix. Run `install_pytorch_cuda.bat` after any `uv sync` operation. See [CUDA_SETUP.md](CUDA_SETUP.md) for details.

## 📚 Documentation

- **[RTX_5090_COMPATIBILITY.md](RTX_5090_COMPATIBILITY.md)** - ⭐ RTX 5090 / Blackwell GPU guide
- **[CUDA_SETUP.md](CUDA_SETUP.md)** - GPU/CUDA setup and troubleshooting
- **[SUMMARY_AND_INSTRUCTIONS.md](SUMMARY_AND_INSTRUCTIONS.md)** - Complete documentation and guide
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Quick command reference
- **[MODEL_RECOMMENDATIONS.md](MODEL_RECOMMENDATIONS.md)** - Model selection guide

## 🎯 Features

✅ **Dataset Preparation** - Automatic PDF processing and tokenization  
✅ **PEFT Training** - Memory-efficient LoRA/QLoRA optimization  
✅ **Model Management** - Easy save/load of trained models  
✅ **Embedding Generation** - Generate text embeddings for downstream tasks  
✅ **Multiple Models** - Support for DeBERTa, ELECTRA, RoBERTa, and more  

## 📁 Project Structure

```
mlm-training/
├── src/
│   ├── mlm_trainer.py         # Main MLM training module ⭐
│   ├── ingest.py               # PDF document ingestion
│   └── ...
├── example_mlm_training.py    # Complete working example ⭐
├── download_models.py         # Model download utility ⭐
├── test_setup.py              # Setup verification script ⭐
├── data/books/                # Place your PDF books here
├── models/                    # Downloaded & trained models
└── results/                   # Training outputs & checkpoints
```

## 🔧 Requirements

- Python 3.12+
- CUDA-capable GPU (recommended: 24GB VRAM)
- Windows 11 / Linux / macOS

## 📦 Key Dependencies

- `transformers` - HuggingFace Transformers
- `peft` - Parameter-Efficient Fine-Tuning (LoRA/QLoRA)
- `bitsandbytes` - Quantization for QLoRA
- `torch` - PyTorch deep learning framework
- `datasets` - HuggingFace Datasets

## 🎓 What is MLM Training?

Masked Language Modeling is a self-supervised learning technique where:
1. Random tokens in text are masked (15%)
2. Model learns to predict masked tokens
3. Improves model's understanding of context and semantics

This is the same technique used to pre-train BERT, RoBERTa, and DeBERTa.

## 💡 Why Use This?

- **Domain Adaptation**: Fine-tune models on your specific domain (books, technical docs, etc.)
- **Better Embeddings**: Generate domain-specific embeddings for RAG, search, clustering
- **Memory Efficient**: QLoRA uses 4x less memory than full fine-tuning
- **Production Ready**: Professional code with logging, checkpointing, and error handling

## 📊 Supported Models

| Model | Parameters | Embedding Dim | Recommended |
|-------|-----------|---------------|-------------|
| DeBERTa-v3-base | 184M | 768 | ⭐ Yes |
| ELECTRA-base | 110M | 768 | ⭐ Yes |
| RoBERTa-base | 125M | 768 | ⭐ Yes |
| DeBERTa-v3-large | 434M | 1024 | Advanced |

## 🔬 Example Usage

### Train a Model
```python
from src.mlm_trainer import (
    prepare_mlm_dataset,
    setup_model_for_mlm_training,
    train_mlm_model,
    save_trained_model
)

# Prepare dataset from PDFs
datasets, tokenizer = prepare_mlm_dataset(
    data_dir="data/books",
    model_name="models/deberta-v3-base"
)

# Setup model with QLoRA
model, _ = setup_model_for_mlm_training(
    model_name="models/deberta-v3-base",
    use_qlora=True,
    use_lora=True
)

# Train
trainer = train_mlm_model(
    model=model,
    tokenizer=tokenizer,
    datasets=datasets,
    epochs=10
)

# Save
save_trained_model(
    model=model,
    tokenizer=tokenizer,
    save_path="models/my-trained-model",
    is_peft_model=True
)
```

### Generate Embeddings
```python
from src.mlm_trainer import generate_embeddings

texts = ["Your text here", "Another text"]

embeddings = generate_embeddings(
    text=texts,
    model_path="models/my-trained-model",
    base_model_name="models/deberta-v3-base",
    is_peft_model=True,
    normalize=True
)

# Use embeddings for similarity, clustering, RAG, etc.
```

## 🎯 Hardware Requirements

**Recommended**:
- GPU: 24GB VRAM (e.g., RTX 4090, A5000)
- RAM: 32GB+
- Storage: 50GB+ for models and datasets

**Minimum** (with QLoRA):
- GPU: 8GB VRAM (e.g., RTX 3060)
- RAM: 16GB
- Storage: 20GB

## 📈 Training Time Estimates

On 24GB GPU with QLoRA:
- DeBERTa-v3-base, 10k samples, 10 epochs: ~2-3 hours
- ELECTRA-base, 10k samples, 10 epochs: ~1.5-2 hours

## 🔧 Troubleshooting

**Out of Memory?**
```python
# Use these settings
use_qlora=True
load_in_4bit=True
batch_size=4
gradient_accumulation_steps=8
```

**Slow Training?**
```python
# Enable FP16
fp16=True
```

**Import Errors?**
```bash
uv sync --force
```

## 📖 Learn More

- [Complete Documentation](SUMMARY_AND_INSTRUCTIONS.md)
- [Model Selection Guide](MODEL_RECOMMENDATIONS.md)
- [Quick Reference](QUICK_REFERENCE.md)

## 🤝 Contributing

This is a personal project, but feel free to adapt it for your needs!

## 📝 License

See project license file.

## 🎉 Ready to Start?

```bash
uv run python test_setup.py     # Verify setup
uv run python example_mlm_training.py  # Start training!
```

---

**Project Status**: ✅ Production Ready  
**Last Updated**: February 2, 2026  
**Hardware Target**: Windows 11, 24GB GPU, 200GB RAM
