# MLM Training Project - Complete Summary & Instructions

## 📋 Project Overview

This project provides a complete pipeline for training encoder models using Masked Language Modeling (MLM) on custom PDF book datasets, with memory-efficient PEFT techniques (LoRA/QLoRA).

**Date**: February 2, 2026
**Status**: ✅ Complete and Ready to Use

---

## 🎯 What Was Accomplished

### 1. ✅ Dataset Preparation Function
**File**: [`src/mlm_trainer.py`](src/mlm_trainer.py) - `prepare_mlm_dataset()`

**Features**:
- Reads PDF books from a directory using your existing `read_files_directory()` function
- Automatically chunks and tokenizes text
- Creates train/test splits (default 90/10)
- Returns HuggingFace `DatasetDict` ready for Trainer
- Handles multiple files in nested directories
- Configurable chunk size, overlap, and max length

**Usage**:
```python
from src.mlm_trainer import prepare_mlm_dataset

datasets, tokenizer = prepare_mlm_dataset(
    data_dir="data/books",
    model_name="models/deberta-v3-base",
    max_length=512,
    chunk_size=2048,
    test_split=0.1
)
```

---

### 2. ✅ Model Recommendations
**File**: [`MODEL_RECOMMENDATIONS.md`](MODEL_RECOMMENDATIONS.md)

**Recommended Models** (all fit in your 24GB GPU with QLoRA):

| Model | Parameters | Quality | Speed | Best For |
|-------|-----------|---------|-------|----------|
| **DeBERTa-v3-base** ⭐ | 184M | ⭐⭐⭐⭐⭐ | ⚡⚡ | **Best overall quality** |
| ELECTRA-base | 110M | ⭐⭐⭐⭐ | ⚡⚡⚡ | Fast iteration |
| RoBERTa-base | 125M | ⭐⭐⭐⭐ | ⚡⚡⚡ | Reliable baseline |

**HuggingFace Links**:
- DeBERTa-v3-base: https://huggingface.co/microsoft/deberta-v3-base
- ELECTRA-base: https://huggingface.co/google/electra-base-discriminator
- RoBERTa-base: https://huggingface.co/FacebookAI/roberta-base

**Download Instructions**:
```bash
# Quick download script (recommended)
uv run python download_models.py --all

# Or download specific model
uv run python download_models.py --model deberta-v3-base
```

---

### 3. ✅ MLM Training Function with PEFT
**File**: [`src/mlm_trainer.py`](src/mlm_trainer.py) - `setup_model_for_mlm_training()` and `train_mlm_model()`

**Features**:
- **QLoRA Support**: 4-bit quantization reduces memory by ~75%
- **LoRA Support**: Parameter-efficient fine-tuning (trains only ~1% of parameters)
- **Automatic Device Mapping**: Utilizes GPU efficiently
- **Mixed Precision (FP16)**: Faster training on modern GPUs
- **Gradient Accumulation**: Simulates larger batch sizes
- **TensorBoard Logging**: Track training progress
- **Best Model Checkpointing**: Automatically saves best model

**Memory Usage**:
- DeBERTa-v3-base with QLoRA: ~3GB VRAM (fits easily in your 24GB GPU!)
- Full fine-tuning: ~12GB VRAM
- **Recommendation**: Use QLoRA for maximum efficiency

**Training Configuration**:
```python
from src.mlm_trainer import setup_model_for_mlm_training, train_mlm_model

# Setup with QLoRA (memory efficient)
model, is_quantized = setup_model_for_mlm_training(
    model_name="models/deberta-v3-base",
    use_qlora=True,
    use_lora=True,
    lora_r=16,
    lora_alpha=32,
    load_in_4bit=True
)

# Train
trainer = train_mlm_model(
    model=model,
    tokenizer=tokenizer,
    datasets=datasets,
    output_dir="results/training-output",
    epochs=10,
    batch_size=8,
    learning_rate=2e-4,
    mlm_probability=0.15
)
```

---

### 4. ✅ Save/Load Functions
**File**: [`src/mlm_trainer.py`](src/mlm_trainer.py) - `save_trained_model()` and `load_trained_model()`

**Features**:
- Saves LoRA adapters (tiny size, ~10-50MB vs multi-GB full model)
- Saves tokenizer alongside model
- Includes training metadata
- Easy loading with base model reference

**Usage**:
```python
from src.mlm_trainer import save_trained_model, load_trained_model

# Save
save_trained_model(
    model=model,
    tokenizer=tokenizer,
    save_path="models/deberta-v3-base-mlm-trained",
    is_peft_model=True
)

# Load
model, tokenizer = load_trained_model(
    model_path="models/deberta-v3-base-mlm-trained",
    base_model_name="models/deberta-v3-base",
    is_peft_model=True
)
```

---

### 5. ✅ Embedding Generation Function
**File**: [`src/mlm_trainer.py`](src/mlm_trainer.py) - `generate_embeddings()` and `get_embedding_info()`

**Features**:
- Multiple pooling strategies: mean, CLS token, max pooling
- L2 normalization option
- Batch processing support
- Returns numpy arrays for easy integration with scikit-learn, numpy, etc.

**Embedding Dimensions**:
- DeBERTa-v3-base: **768 dimensions**
- ELECTRA-base: **768 dimensions**
- RoBERTa-base: **768 dimensions**
- DeBERTa-v3-large: **1024 dimensions**

**Usage**:
```python
from src.mlm_trainer import generate_embeddings, get_embedding_info

# Get model info
info = get_embedding_info(
    model_path="models/deberta-v3-base-mlm-trained",
    base_model_name="models/deberta-v3-base"
)
print(f"Embedding size: {info['embedding_size']}")  # 768

# Generate embeddings
texts = [
    "Machine learning is a subset of AI.",
    "Deep learning uses neural networks."
]

embeddings = generate_embeddings(
    text=texts,
    model_path="models/deberta-v3-base-mlm-trained",
    base_model_name="models/deberta-v3-base",
    is_peft_model=True,
    pooling_strategy="mean",  # or "cls" or "max"
    normalize=True
)

print(embeddings.shape)  # (2, 768)
```

---

### 6. ✅ Dependencies Added
**File**: [`pyproject.toml`](pyproject.toml)

**New Dependencies**:
- `peft>=0.13.0` - Parameter-efficient fine-tuning (LoRA/QLoRA)
- `bitsandbytes>=0.45.0` - Quantization for QLoRA
- `accelerate>=1.2.1` - Distributed training support
- `sentencepiece>=0.2.0` - Tokenization for some models
- `tensorboard>=2.18.0` - Training visualization
- `scipy>=1.15.1` - Scientific computing utilities

**Install all dependencies**:
```bash
cd c:\projects\mlm-training
uv sync
```

---

## 🚀 Quick Start Guide

### Step 1: Install Dependencies
```bash
cd c:\projects\mlm-training
uv sync
```

### Step 2: Download a Model
```bash
# Download all recommended models
uv run python download_models.py --all

# Or just one model
uv run python download_models.py --model deberta-v3-base
```

### Step 3: Prepare Your Data
Place your PDF books in the `data/books` directory:
```
data/
  books/
    ai/
      agents/
        book1.pdf
        book2.pdf
      genai/
        book3.pdf
    language_models/
      book4.pdf
```

### Step 4: Run Training
```bash
# Run the complete example
uv run python example_mlm_training.py
```

**Or customize** by editing [`example_mlm_training.py`](example_mlm_training.py):
- Change `MODEL_NAME` to your preferred model
- Adjust `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`
- Set `MAX_CHUNKS` for quick testing
- Modify LoRA parameters for experimentation

### Step 5: Monitor Training
```bash
# In a separate terminal, launch TensorBoard
tensorboard --logdir=results
```

Then open http://localhost:6006 in your browser to view:
- Training loss over time
- Validation loss
- Learning rate schedule
- GPU utilization

---

## 📁 Project Structure

```
mlm-training/
├── src/
│   ├── mlm_trainer.py          # ⭐ Main MLM training module (NEW)
│   ├── ingest.py                # Existing PDF ingestion
│   ├── train.py                 # Your original training code
│   ├── hyperparams.py          # Hyperparameter utilities
│   └── storage.py              # Cloud storage utilities
│
├── models/                      # Downloaded & trained models
│   ├── deberta-v3-base/        # Base model (downloaded)
│   └── deberta-v3-base-mlm-trained-20260202/  # Your trained model
│
├── data/
│   └── books/                  # Your PDF books here
│
├── results/                    # Training outputs
│   └── training-*/             # Checkpoints, logs, TensorBoard
│
├── logs/                       # Training log files
│
├── example_mlm_training.py    # ⭐ Complete example (NEW)
├── download_models.py         # ⭐ Model download script (NEW)
├── MODEL_RECOMMENDATIONS.md   # ⭐ Model guide (NEW)
├── SUMMARY_AND_INSTRUCTIONS.md # ⭐ This file (NEW)
└── pyproject.toml             # Updated with new dependencies
```

---

## 💡 Usage Examples

### Example 1: Basic Training
```python
from src.mlm_trainer import (
    prepare_mlm_dataset,
    setup_model_for_mlm_training,
    train_mlm_model,
    save_trained_model
)

# 1. Prepare data
datasets, tokenizer = prepare_mlm_dataset(
    data_dir="data/books",
    model_name="models/deberta-v3-base"
)

# 2. Setup model with QLoRA
model, _ = setup_model_for_mlm_training(
    model_name="models/deberta-v3-base",
    use_qlora=True,
    use_lora=True
)

# 3. Train
trainer = train_mlm_model(
    model=model,
    tokenizer=tokenizer,
    datasets=datasets,
    epochs=10
)

# 4. Save
save_trained_model(
    model=model,
    tokenizer=tokenizer,
    save_path="models/my-trained-model",
    is_peft_model=True
)
```

### Example 2: Generate Embeddings
```python
from src.mlm_trainer import generate_embeddings
import numpy as np

texts = [
    "Natural language processing enables AI to understand text.",
    "Machine learning models learn patterns from data.",
    "Deep learning architectures use multiple neural network layers."
]

# Generate embeddings
embeddings = generate_embeddings(
    text=texts,
    model_path="models/deberta-v3-base-mlm-trained",
    base_model_name="models/deberta-v3-base",
    is_peft_model=True,
    normalize=True
)

# Use for similarity search
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity(embeddings)
print(similarities)
```

### Example 3: Custom Training Configuration
```python
# For longer sequences (if you have memory)
datasets, tokenizer = prepare_mlm_dataset(
    data_dir="data/books",
    model_name="models/deberta-v3-base",
    max_length=768,  # Longer sequences
    chunk_size=3072
)

# More aggressive LoRA settings
model, _ = setup_model_for_mlm_training(
    model_name="models/deberta-v3-base",
    use_qlora=True,
    lora_r=32,      # Higher rank = more capacity
    lora_alpha=64,  # Higher alpha = stronger adaptation
    lora_dropout=0.05
)

# Longer training with smaller batches
trainer = train_mlm_model(
    model=model,
    tokenizer=tokenizer,
    datasets=datasets,
    epochs=20,
    batch_size=4,
    gradient_accumulation_steps=8,  # Effective batch = 32
    learning_rate=1e-4
)
```

---

## ⚙️ Configuration & Hyperparameters

### Key Hyperparameters

| Parameter | Default | Description | Recommendations |
|-----------|---------|-------------|-----------------|
| `max_length` | 512 | Max sequence length | 512 for most models, 768 for long docs |
| `chunk_size` | 2048 | PDF chunk size | 2-4x max_length |
| `epochs` | 10 | Training epochs | Start with 5-10, increase if underfitting |
| `batch_size` | 8 | Batch size per GPU | 4-16 depending on memory |
| `learning_rate` | 2e-4 | Learning rate | 1e-4 to 5e-4 for LoRA |
| `mlm_probability` | 0.15 | % tokens masked | 0.15 is standard for MLM |
| `lora_r` | 16 | LoRA rank | 8-64, higher = more parameters |
| `lora_alpha` | 32 | LoRA scaling | Typically 2x lora_r |
| `lora_dropout` | 0.1 | LoRA dropout | 0.05-0.1 for regularization |

### LoRA Parameters Explained

- **`lora_r` (rank)**: Number of low-rank dimensions. Higher = more trainable parameters = better quality but more memory.
  - Small tasks: 8-16
  - Medium tasks: 16-32
  - Large tasks: 32-64

- **`lora_alpha`**: Scaling factor. Higher = stronger adaptation.
  - Rule of thumb: `lora_alpha = 2 * lora_r`

- **`lora_dropout`**: Regularization to prevent overfitting.
  - Small datasets: 0.1-0.2
  - Large datasets: 0.05-0.1

### Embedding Pooling Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `mean` | Average all token embeddings | General-purpose, most common |
| `cls` | Use [CLS] token only | Classification tasks |
| `max` | Max pool across tokens | Highlighting key features |

---

## 🔧 Troubleshooting

### Out of Memory (OOM) Errors

**Solutions**:
1. Enable QLoRA: `use_qlora=True, load_in_4bit=True`
2. Reduce batch size: `batch_size=4`
3. Increase gradient accumulation: `gradient_accumulation_steps=8`
4. Reduce max_length: `max_length=256`
5. Use smaller model (ELECTRA-base instead of DeBERTa)

### Slow Training

**Solutions**:
1. Ensure GPU is being used: Check logs for "cuda" device
2. Enable FP16: `fp16=True` (default)
3. Reduce `eval_steps` and `logging_steps`
4. Use smaller dataset for testing: `max_chunks=1000`

### Poor Performance

**Solutions**:
1. Train longer: Increase `epochs`
2. Increase LoRA rank: `lora_r=32`
3. Lower learning rate: `learning_rate=1e-4`
4. Check data quality: Ensure PDFs are being read correctly
5. Increase dataset size: Add more books

### Import Errors

```bash
# Reinstall dependencies
uv sync --force

# Or install missing package
uv add <package-name>
```

---

## 📊 Expected Training Time

On your hardware (24GB GPU, 200GB RAM):

| Model | Dataset Size | Epochs | Batch Size | Estimated Time |
|-------|--------------|--------|-----------|----------------|
| ELECTRA-base | 10k samples | 10 | 8 | ~1.5-2 hours |
| DeBERTa-v3-base | 10k samples | 10 | 8 | ~2-3 hours |
| DeBERTa-v3-base | 50k samples | 10 | 8 | ~8-12 hours |
| RoBERTa-base | 10k samples | 10 | 8 | ~1.5-2.5 hours |

With QLoRA, you can comfortably run overnight training sessions.

---

## 🎓 Key Concepts

### Masked Language Modeling (MLM)
- Randomly mask 15% of tokens in input
- Model learns to predict masked tokens
- Teaches model to understand context and semantics
- Pre-training technique used by BERT, RoBERTa, DeBERTa

### LoRA (Low-Rank Adaptation)
- Freezes base model weights
- Adds small trainable adapter layers
- Trains only ~1% of parameters
- Much faster and more memory-efficient
- Final model = base model + tiny adapter weights

### QLoRA (Quantized LoRA)
- Loads base model in 4-bit precision
- Reduces memory by ~75%
- Maintains quality with proper configuration
- Enables training larger models on smaller GPUs

### Embedding Pooling
- Converts variable-length sequences to fixed vectors
- Required for similarity search, clustering, classification
- Different strategies capture different information

---

## 🔬 Advanced Usage

### Continue Training from Checkpoint
```python
# Load existing model
model, tokenizer = load_trained_model(
    model_path="models/deberta-v3-base-mlm-trained",
    base_model_name="models/deberta-v3-base",
    is_peft_model=True
)

# Train more with lower learning rate
trainer = train_mlm_model(
    model=model,
    tokenizer=tokenizer,
    datasets=datasets,
    epochs=5,
    learning_rate=1e-4  # Lower for fine-tuning
)
```

### Merge LoRA Adapters to Base Model
```python
from peft import PeftModel

# Load PEFT model
base_model = AutoModelForMaskedLM.from_pretrained("models/deberta-v3-base")
peft_model = PeftModel.from_pretrained(base_model, "models/deberta-v3-base-mlm-trained")

# Merge and save as full model
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained("models/deberta-v3-base-merged")
```

### Custom Data Collator
```python
from transformers import DataCollatorForLanguageModeling

# Custom masking probability
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.20  # Mask 20% instead of 15%
)
```

---

## 📈 Next Steps

After MLM training, you can:

1. **Use Embeddings for RAG**: Generate embeddings for your books and use for retrieval
2. **Task-Specific Fine-Tuning**: Further fine-tune for classification, NER, etc.
3. **Create Synthetic SFT Dataset**: Your original plan - generate Q&A pairs from paragraphs
4. **Evaluate on Benchmarks**: Test on GLUE, SuperGLUE, or domain-specific tasks
5. **Deploy as API**: Serve embeddings via FastAPI or similar

---

## 📚 Additional Resources

- **Transformers Documentation**: https://huggingface.co/docs/transformers
- **PEFT Documentation**: https://huggingface.co/docs/peft
- **LoRA Paper**: https://arxiv.org/abs/2106.09685
- **QLoRA Paper**: https://arxiv.org/abs/2305.14314
- **MLM Tutorial**: https://huggingface.co/course/chapter7/3

---

## 🐛 Support & Issues

If you encounter issues:

1. Check logs in `logs/` directory
2. Verify GPU is available: `torch.cuda.is_available()`
3. Check CUDA version: `torch.version.cuda`
4. Ensure sufficient disk space (models can be large)
5. Review error messages carefully

---

## ✅ Summary Checklist

- [x] Dataset preparation function created
- [x] 3 encoder models recommended with HF links
- [x] Download script provided
- [x] MLM training function with LoRA/QLoRA
- [x] Model save/load functions
- [x] Embedding generation function
- [x] Dependencies added to pyproject.toml
- [x] Complete example script
- [x] Comprehensive documentation

---

## 🎉 Ready to Train!

You now have a complete MLM training pipeline with:
- Memory-efficient QLoRA/LoRA optimization
- Professional logging and checkpointing
- Flexible configuration options
- Production-ready code

**Start training**:
```bash
uv run python download_models.py --model deberta-v3-base
uv run python example_mlm_training.py
```

Happy training! 🚀

---

**Project**: MLM Training for Encoder Models
**Date**: February 2, 2026
**Hardware**: Windows 11, 24GB GPU, 200GB RAM
**Status**: ✅ Production Ready
