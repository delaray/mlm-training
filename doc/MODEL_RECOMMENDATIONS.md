# Encoder Model Recommendations for MLM Training

## Hardware Context
- **GPU**: 24GB VRAM
- **RAM**: 200GB
- **OS**: Windows 11

With this hardware, you can train medium-sized encoder models efficiently, especially with QLoRA optimization.

---

## Recommended Models

### 1. **DeBERTa-v3-base** (Recommended for Best Quality)
- **Model Card**: https://huggingface.co/microsoft/deberta-v3-base
- **Size**: ~184M parameters
- **Embedding Dimension**: 768
- **Max Sequence Length**: 512
- **Memory Requirement**: ~700MB base model, ~2-3GB during training with QLoRA

**Why DeBERTa-v3?**
- State-of-the-art performance among encoder models
- Better than BERT on most NLU benchmarks
- Disentangled attention mechanism improves context understanding
- Excellent for domain adaptation (your use case)

**Download Command**:
```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

model_name = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Save locally
model.save_pretrained("./models/deberta-v3-base")
tokenizer.save_pretrained("./models/deberta-v3-base")
```

**Training Time Estimate**: ~2-4 hours for 10 epochs on your GPU with QLoRA

---

### 2. **ELECTRA-base** (Recommended for Efficiency)
- **Model Card**: https://huggingface.co/google/electra-base-discriminator
- **Size**: ~110M parameters
- **Embedding Dimension**: 768
- **Max Sequence Length**: 512
- **Memory Requirement**: ~450MB base model, ~1.5-2GB during training with QLoRA

**Why ELECTRA?**
- More sample-efficient than BERT (learns faster)
- Uses "replaced token detection" instead of MLM (but can still be fine-tuned with MLM)
- Smaller and faster than DeBERTa
- Great quality-to-size ratio

**Download Command**:
```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

model_name = "google/electra-base-discriminator"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Save locally
model.save_pretrained("./models/electra-base")
tokenizer.save_pretrained("./models/electra-base")
```

**Training Time Estimate**: ~1.5-3 hours for 10 epochs on your GPU with QLoRA

---

### 3. **RoBERTa-base** (Classic, Reliable Choice)
- **Model Card**: https://huggingface.co/FacebookAI/roberta-base
- **Size**: ~125M parameters
- **Embedding Dimension**: 768
- **Max Sequence Length**: 512
- **Memory Requirement**: ~500MB base model, ~1.5-2.5GB during training with QLoRA

**Why RoBERTa?**
- Improved version of BERT with better training approach
- No next-sentence prediction (NSP) - just MLM
- Extensively tested and widely used
- Large community support

**Download Command**:
```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

model_name = "FacebookAI/roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Save locally
model.save_pretrained("./models/roberta-base")
tokenizer.save_pretrained("./models/roberta-base")
```

**Training Time Estimate**: ~1.5-3 hours for 10 epochs on your GPU with QLoRA

---

## Alternative: Larger Models (If You Want Maximum Quality)

### 4. **DeBERTa-v3-large**
- **Model Card**: https://huggingface.co/microsoft/deberta-v3-large
- **Size**: ~434M parameters
- **Embedding Dimension**: 1024
- **Memory Requirement**: ~1.7GB base model, ~4-6GB during training with QLoRA

This will work well with your 24GB GPU using QLoRA + gradient checkpointing.

---

## Comparison Table

| Model | Parameters | Embedding Size | Speed | Quality | Recommended For |
|-------|-----------|----------------|-------|---------|-----------------|
| ELECTRA-base | 110M | 768 | ⚡⚡⚡ | ⭐⭐⭐⭐ | Fast iteration |
| RoBERTa-base | 125M | 768 | ⚡⚡⚡ | ⭐⭐⭐⭐ | Reliable baseline |
| DeBERTa-v3-base | 184M | 768 | ⚡⚡ | ⭐⭐⭐⭐⭐ | **Best quality** |
| DeBERTa-v3-large | 434M | 1024 | ⚡ | ⭐⭐⭐⭐⭐⭐ | Maximum quality |

---

## Download Instructions

### Method 1: Direct Download via Python (Recommended)

Create a script `download_models.py`:

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import os

models_to_download = [
    "microsoft/deberta-v3-base",
    "google/electra-base-discriminator",
    "FacebookAI/roberta-base"
]

models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

for model_name in models_to_download:
    print(f"Downloading {model_name}...")
    
    # Extract model short name
    short_name = model_name.split("/")[-1]
    save_path = os.path.join(models_dir, short_name)
    
    # Download and save
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    
    print(f"✓ Saved to {save_path}\n")

print("All models downloaded successfully!")
```

Run it:
```bash
uv run python download_models.py
```

### Method 2: Using Hugging Face CLI

```bash
# Install huggingface-hub
uv add huggingface-hub

# Download models
huggingface-cli download microsoft/deberta-v3-base --local-dir models/deberta-v3-base
huggingface-cli download google/electra-base-discriminator --local-dir models/electra-base
huggingface-cli download FacebookAI/roberta-base --local-dir models/roberta-base
```

---

## My Recommendation

**Start with DeBERTa-v3-base** because:
1. ✅ Best quality-to-size ratio
2. ✅ Fits comfortably in your GPU with QLoRA
3. ✅ State-of-the-art performance
4. ✅ Well-maintained by Microsoft
5. ✅ Excellent documentation

**Fallback to ELECTRA-base** if:
- You need faster training iterations
- You want to experiment quickly
- Training time is a concern

---

## Memory Estimates with QLoRA

| Model | Base Size | Training (QLoRA) | Training (Full) |
|-------|-----------|------------------|-----------------|
| ELECTRA-base | 450MB | ~2GB | ~8GB |
| RoBERTa-base | 500MB | ~2.5GB | ~9GB |
| DeBERTa-v3-base | 700MB | ~3GB | ~12GB |
| DeBERTa-v3-large | 1.7GB | ~6GB | ~20GB |

With 24GB GPU, you can comfortably train any of these with QLoRA!

---

## Next Steps

1. Download your chosen model(s) to `models/` directory
2. Use the functions in `src/mlm_trainer.py` to train
3. See `example_mlm_training.py` for a complete working example
