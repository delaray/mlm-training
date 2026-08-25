# Setup and Run Guide

This guide covers a clean local workflow from installation to a trained adapter. Commands assume the repository root as the working directory.

## Requirements

| Resource | Minimum for tutorial | Recommended for larger runs |
|---|---:|---:|
| Python | 3.12 | 3.12 |
| RAM | 8 GB | 32 GB+ |
| GPU | Not required | CUDA GPU with 12–24 GB VRAM |
| Disk | 3 GB plus model cache | 20 GB+ |

## Install

```bash
uv sync
uv run python tests/test_setup.py
```

PyTorch GPU wheels are platform- and CUDA-specific. Install the correct build from the official PyTorch selector after `uv sync`. CPU runs automatically disable FP16 and QLoRA quantization.

## Prepare data

Place authorized PDFs in any directory. Discovery is recursive.

```text
data/books/
├── handbook.pdf
└── references/
    └── field-guide.pdf
```

The loader ignores unsupported extensions, records unreadable files, extracts text, and creates overlapping chunks. For formal validation, keep evaluation documents outside the training directory.

## Configure

Copy `configs/mlm_training.yaml` and change the copy. Start conservatively on a laptop:

```yaml
dataset:
  max_length: 256
  max_chunks: 500
model:
  device: cpu
  use_lora: true
  use_qlora: false
  load_in_4bit: false
training:
  epochs: 1
  batch_size: 2
  gradient_accumulation_steps: 4
  fp16: false
```

## Select hyperparameters

```bash
uv run python run_optuna.py data/books configs/best.yaml \
  --model microsoft/deberta-v3-xsmall \
  --trials 8 \
  --config configs/mlm_training.yaml
```

Each trial trains a fresh model and minimizes held-out MLM loss. Trial checkpoints are placed below `results/optuna`; the output YAML contains the winning training values and study metadata.

## Train the selected model

```bash
uv run python run_mlm.py microsoft/deberta-v3-xsmall data/books \
  --config configs/best.yaml
```

Artifacts are written below the configured `models_dir`; checkpoints and TensorBoard events go to `results_dir`; logs go to `logs_dir`.

## Inspect training

```bash
uv run tensorboard --logdir results
```

Track training and validation loss together. A falling training loss with rising validation loss indicates overfitting. Compare embeddings on a frozen task-specific benchmark before adopting the adapted model.

## Run the notebooks

```bash
uv add --dev jupyter
uv run jupyter lab notebooks
```

Run the notebooks top-to-bottom. Their first execution downloads the model. Set `FAST_MODE = True` for the shortest CPU demonstration.

## Common failures

| Symptom | Likely cause | Action |
|---|---|---|
| CUDA out of memory | Batch/sequence too large | Reduce both, enable LoRA/QLoRA, increase accumulation |
| No text chunks | Scanned or protected PDF | Add OCR or validate extraction manually |
| Missing `sentencepiece` | DeBERTa tokenizer dependency | Re-run `uv sync` and verify the lockfile |
| 4-bit load fails | CPU run or incompatible bitsandbytes/CUDA | Disable QLoRA or install a compatible CUDA stack |
| Validation is suspiciously strong | Overlapping chunks crossed the split | Split by document before chunking |
| Notebook metric does not improve | Small stochastic demonstration | Increase corpus diversity/steps and test multiple seeds |

## Reproducible run checklist

1. Record the git commit, base model revision, config, random seed, and corpus manifest.
2. Keep a document-level held-out set untouched by Optuna.
3. Save the Optuna study or database, not only its winning trial.
4. Report base and adapted results using identical preprocessing.
5. Repeat training across seeds and publish variance.
