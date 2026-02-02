# MLM Training - Quick Reference

## 🚀 Quick Start (3 Commands)

```bash
# 1. Install dependencies
uv sync

# 2. Download a model
uv run python download_models.py --model deberta-v3-base

# 3. Start training
uv run python example_mlm_training.py
```

---

## 📝 Common Commands

### Setup & Installation
```bash
# Install all dependencies
uv sync

# Test setup
uv run python test_setup.py

# Activate virtual environment
.\venv\Scripts\activate  # Windows
```

### Download Models
```bash
# All recommended models
uv run python download_models.py --all

# Specific model
uv run python download_models.py --model deberta-v3-base
uv run python download_models.py --model electra-base
uv run python download_models.py --model roberta-base

# List available models
uv run python download_models.py --list
```

### Training
```bash
# Run example training
uv run python example_mlm_training.py

# Monitor with TensorBoard
tensorboard --logdir=results
# Then open: http://localhost:6006
```

---

## 🎯 Code Snippets

### Prepare Dataset
```python
from src.mlm_trainer import prepare_mlm_dataset

datasets, tokenizer = prepare_mlm_dataset(
    data_dir="data/books",
    model_name="models/deberta-v3-base",
    max_length=512,
    test_split=0.1
)
```

### Setup Model with QLoRA
```python
from src.mlm_trainer import setup_model_for_mlm_training

model, is_quantized = setup_model_for_mlm_training(
    model_name="models/deberta-v3-base",
    use_qlora=True,
    use_lora=True,
    lora_r=16,
    lora_alpha=32,
    load_in_4bit=True
)
```

### Train Model
```python
from src.mlm_trainer import train_mlm_model

trainer = train_mlm_model(
    model=model,
    tokenizer=tokenizer,
    datasets=datasets,
    output_dir="results/training",
    epochs=10,
    batch_size=8,
    learning_rate=2e-4
)
```

### Save Model
```python
from src.mlm_trainer import save_trained_model

save_trained_model(
    model=model,
    tokenizer=tokenizer,
    save_path="models/my-trained-model",
    is_peft_model=True
)
```

### Load Model
```python
from src.mlm_trainer import load_trained_model

model, tokenizer = load_trained_model(
    model_path="models/my-trained-model",
    base_model_name="models/deberta-v3-base",
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
    pooling_strategy="mean",
    normalize=True
)

# embeddings.shape = (2, 768) for base models
```

### Get Embedding Info
```python
from src.mlm_trainer import get_embedding_info

info = get_embedding_info(
    model_path="models/my-trained-model",
    base_model_name="models/deberta-v3-base"
)

print(f"Embedding size: {info['embedding_size']}")  # 768
```

---

## ⚙️ Key Configuration Options

### Dataset Preparation
- `data_dir`: PDF books location
- `max_length`: 256, 512, 768 (model dependent)
- `chunk_size`: 2048-4096 recommended
- `test_split`: 0.1 (10% for validation)

### Model Setup (LoRA/QLoRA)
- `use_qlora`: True (saves 75% memory)
- `use_lora`: True (parameter-efficient)
- `lora_r`: 8-64 (higher = more capacity)
- `lora_alpha`: 2 × lora_r (typically)
- `load_in_4bit`: True for QLoRA

### Training
- `epochs`: 5-20 depending on dataset
- `batch_size`: 4-16 (adjust for memory)
- `learning_rate`: 1e-4 to 5e-4 for LoRA
- `mlm_probability`: 0.15 (standard)
- `gradient_accumulation_steps`: 4-8

### Embeddings
- `pooling_strategy`: "mean", "cls", "max"
- `normalize`: True (for cosine similarity)
- `max_length`: 512 (or model's max)

---

## 📊 Model Comparison

| Model | Size | Embedding Dim | Quality | Speed |
|-------|------|---------------|---------|-------|
| ELECTRA-base | 110M | 768 | ⭐⭐⭐⭐ | ⚡⚡⚡ |
| RoBERTa-base | 125M | 768 | ⭐⭐⭐⭐ | ⚡⚡⚡ |
| **DeBERTa-v3-base** ⭐ | 184M | 768 | ⭐⭐⭐⭐⭐ | ⚡⚡ |
| DeBERTa-v3-large | 434M | 1024 | ⭐⭐⭐⭐⭐⭐ | ⚡ |

**Recommended**: Start with DeBERTa-v3-base

---

## 🔧 Troubleshooting Quick Fixes

### Out of Memory
```python
# Use these settings
use_qlora=True
load_in_4bit=True
batch_size=4
gradient_accumulation_steps=8
max_length=256
```

### Slow Training
```python
# Enable these
fp16=True
num_proc=4  # in dataset.map()
dataloader_num_workers=4
```

### Import Errors
```bash
uv sync --force
```

### Model Not Found
```bash
uv run python download_models.py --model <model-name>
```

---

## 📁 File Locations

| Item | Location |
|------|----------|
| Training code | `src/mlm_trainer.py` |
| Example script | `example_mlm_training.py` |
| Downloaded models | `models/` |
| PDF books | `data/books/` |
| Training outputs | `results/` |
| Logs | `logs/` |
| Checkpoints | `results/training-*/checkpoint-*` |

---

## 🎯 Typical Workflow

1. **Setup**
   ```bash
   uv sync
   uv run python test_setup.py
   ```

2. **Download Model**
   ```bash
   uv run python download_models.py --model deberta-v3-base
   ```

3. **Add Data**
   - Place PDFs in `data/books/`

4. **Train**
   ```bash
   uv run python example_mlm_training.py
   ```

5. **Monitor**
   ```bash
   tensorboard --logdir=results
   ```

6. **Use Embeddings**
   ```python
   from src.mlm_trainer import generate_embeddings
   embeddings = generate_embeddings(texts, "models/my-model", ...)
   ```

---

## 💡 Pro Tips

1. **Start Small**: Use `max_chunks=1000` for quick testing
2. **Monitor GPU**: Use `nvidia-smi` to check memory usage
3. **Save Checkpoints**: Default saves every 1000 steps
4. **Use QLoRA**: Saves memory, minimal quality loss
5. **Normalize Embeddings**: For cosine similarity
6. **Check Logs**: All logs saved to `logs/` directory

---

## 📚 Documentation Files

- [`SUMMARY_AND_INSTRUCTIONS.md`](SUMMARY_AND_INSTRUCTIONS.md) - Complete guide
- [`MODEL_RECOMMENDATIONS.md`](MODEL_RECOMMENDATIONS.md) - Model details
- [`example_mlm_training.py`](example_mlm_training.py) - Full example
- [`src/mlm_trainer.py`](src/mlm_trainer.py) - Core functions

---

## 🆘 Quick Help

```bash
# Check environment
uv run python test_setup.py

# List available models  
uv run python download_models.py --list

# View this guide
cat QUICK_REFERENCE.md
```

---

**Ready to train?** Run: `uv run python example_mlm_training.py` 🚀
