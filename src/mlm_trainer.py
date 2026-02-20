"""
MLM (Masked Language Modeling) Training Module

This module provides functions to:
1. Prepare datasets from PDF books for MLM training
2. Train encoder models using MLM with PEFT (LoRA/QLoRA) optimization
3. Save and load trained models
4. Generate text embeddings

Author: MLM Training Project
Date: February 2026
"""

import os
import logging
from datetime import datetime
from typing import List, Dict, Union, Optional, Tuple
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training
)
import numpy as np

from src.ingest import read_files_directory


# ------------------------------------------------------------------------------
# Configuration & Defaults
# ------------------------------------------------------------------------------

DEFAULT_DATA_DIR = "data/books"
DEFAULT_MODELS_DIR = "models"
DEFAULT_RESULTS_DIR = "results"
DEFAULT_MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 10
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_MLM_PROBABILITY = 0.15

# LoRA Configuration
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.1


# ------------------------------------------------------------------------------
# 1. Dataset Preparation Functions
# ------------------------------------------------------------------------------

def prepare_mlm_dataset(
    data_dir: str = DEFAULT_DATA_DIR,
    model_name: str = "google/electra-small-discriminator",
    max_length: int = DEFAULT_MAX_LENGTH,
    chunk_size: int = 2048,
    chunk_overlap: int = 200,
    test_split: float = 0.1,
    max_chunks: Optional[int] = None,
    use_fast_tokenizer: bool = True
) -> Tuple[DatasetDict, AutoTokenizer]:
    """
    Prepare a dataset from a directory of PDF books for MLM training.

    Args:
        data_dir: Directory containing PDF books
        model_name: HuggingFace model name/path for tokenizer
        max_length: Maximum sequence length for tokenization
        chunk_size: Size of text chunks from PDF extraction
        chunk_overlap: Overlap between chunks
        test_split: Fraction of data to use for validation
        max_chunks: Maximum number of chunks to use (None = all)
        use_fast_tokenizer: Whether to use fast tokenizer

    Returns:
        Tuple of (DatasetDict with train/test splits, tokenizer)
    """
    logging.info("="*80)
    logging.info("Starting MLM Dataset Preparation")
    logging.info("="*80)

    start_time = datetime.now()

    # Load tokenizer
    logging.info(f"Loading tokenizer from: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=use_fast_tokenizer
    )

    # Read and chunk documents
    logging.info(f"Reading documents from: {data_dir}")
    chunks, files_count, problem_files = read_files_directory(
        data_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    logging.info(f"Successfully read {files_count} files")
    if problem_files:
        logging.warning(f"Failed to read {len(problem_files)} "
                        f"files: {problem_files}")

    # Limit chunks if specified
    if max_chunks is not None:
        chunks = chunks[:max_chunks]
        logging.info(f"Limited to {max_chunks} chunks")

    logging.info(f"Total chunks available: {len(chunks)}")

    if len(chunks) == 0:
        raise ValueError("No text chunks extracted. Check data directory.")

    # Create dataset
    data_dict = {"text": chunks}
    dataset = Dataset.from_dict(data_dict)

    # Split into train/test
    logging.info(f"Splitting dataset (test_split={test_split})")
    dataset = dataset.train_test_split(test_size=test_split, seed=42)

    # Tokenization function
    def tokenize_function(examples):
        """Tokenize text examples for MLM training"""
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True
        )

    # Apply tokenization
    logging.info("Tokenizing dataset...")
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=["text"],
        desc="Tokenizing"
    )

    # Log statistics
    train_size = len(tokenized_datasets['train'])
    test_size = len(tokenized_datasets['test'])
    logging.info(f"Training samples: {train_size}")
    logging.info(f"Test samples: {test_size}")
    logging.info(f"Max sequence length: {max_length}")

    elapsed = datetime.now() - start_time
    logging.info(f"Dataset preparation completed in {elapsed}")
    logging.info("="*80)

    return tokenized_datasets, tokenizer


# ------------------------------------------------------------------------------
# 2. Model Training Functions with PEFT (LoRA/QLoRA)
# ------------------------------------------------------------------------------

def setup_model_for_mlm_training(
    model_name: str,
    use_qlora: bool = True,
    use_lora: bool = True,
    lora_r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    lora_dropout: float = DEFAULT_LORA_DROPOUT,
    target_modules: Optional[List[str]] = None,
    load_in_4bit: bool = True,
    load_in_8bit: bool = False
) -> Tuple[AutoModelForMaskedLM, bool]:
    """
    Load and configure an encoder model for MLM training with
    optional LoRA/QLoRA.

    Args:
        model_name: HuggingFace model name or local path
        use_qlora: Use QLoRA (quantized LoRA) - saves memory
        use_lora: Use LoRA for parameter-efficient fine-tuning
        lora_r: LoRA rank dimension
        lora_alpha: LoRA alpha parameter (scaling factor)
        lora_dropout: Dropout for LoRA layers
        target_modules: Specific modules to apply LoRA (None = auto-detect)
        load_in_4bit: Load model in 4-bit quantization (QLoRA)
        load_in_8bit: Load model in 8-bit quantization

    Returns:
        Tuple of (model, is_quantized)
    """
    logging.info("="*80)
    logging.info("Setting up model for MLM training")
    logging.info("="*80)

    is_quantized = False

    # Configure quantization if using QLoRA
    if use_qlora and (load_in_4bit or load_in_8bit):
        logging.info(f"Configuring quantization: 4-bit={load_in_4bit}, "
                     f"8-bit={load_in_8bit}")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            bnb_4bit_quant_type="nf4" if load_in_4bit else None,
            bnb_4bit_compute_dtype=torch.float16 if load_in_4bit else None,
            bnb_4bit_use_double_quant=True if load_in_4bit else False,
        )

        model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        is_quantized = True

        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
        logging.info("Model prepared for quantized training")

    else:
        # Load model normally (no quantization, no device_map for RTX 5090
        # compatibility) Keep on CPU initially to avoid sm_120 kernel issues
        # during PEFT setup
        logging.info("Loading model without quantization (on CPU first)")
        model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=torch.float32  # Load in FP32 on CPU
        )
        logging.info("Model loaded on CPU")

    # Apply LoRA if requested (do this on CPU to avoid sm_120 kernel issues)
    if use_lora:
        logging.info("Applying LoRA configuration on CPU")

        # Auto-detect target modules if not specified
        if target_modules is None:
            # Common patterns for encoder models
            target_modules = ["query", "key", "value", "dense"]
            logging.info(f"Auto-detected target modules: {target_modules}")

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION  # For encoder models
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        logging.info("LoRA applied successfully on CPU")

    # RTX 5090 (sm_120) lacks kernels even for basic operations like embeddings
    # Keep model on CPU for now - will be faster than hitting kernel errors
    logging.info("Keeping model on CPU (RTX 5090 sm_120 lacks CUDA kernels)")
    logging.info("Note: Training will use CPU.")
    logging.info("For GPU wait for PyTorch with full sm_120 support.")

    logging.info("="*80)
    return model, is_quantized


def train_mlm_model(
    model: AutoModelForMaskedLM,
    tokenizer: AutoTokenizer,
    datasets: DatasetDict,
    output_dir: str = DEFAULT_RESULTS_DIR,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    mlm_probability: float = DEFAULT_MLM_PROBABILITY,
    weight_decay: float = 0.01,
    warmup_steps: int = 500,
    logging_steps: int = 100,
    save_steps: int = 1000,
    eval_steps: int = 500,
    gradient_accumulation_steps: int = 4,
    fp16: bool = True,
    save_total_limit: int = 2
) -> Trainer:
    """
    Train an encoder model using Masked Language Modeling (MLM).

    Args:
        model: The model to train (can be LoRA/QLoRA wrapped)
        tokenizer: Tokenizer for the model
        datasets: DatasetDict with 'train' and 'test' splits
        output_dir: Directory to save checkpoints and results
        epochs: Number of training epochs
        batch_size: Training batch size per device
        learning_rate: Learning rate
        mlm_probability: Probability of masking tokens (typically 0.15)
        weight_decay: Weight decay for optimization
        warmup_steps: Number of warmup steps
        logging_steps: Log every N steps
        save_steps: Save checkpoint every N steps
        eval_steps: Evaluate every N steps
        gradient_accumulation_steps: Accumulate gradients for larger eff. batch
        fp16: Use mixed precision training (fp16)
        save_total_limit: Keep only N most recent checkpoints

    Returns:
        Trained Trainer object
    """
    logging.info("="*80)
    logging.info("Starting MLM Training")
    logging.info("="*80)

    start_time = datetime.now()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set TensorBoard logging directory (new method, logging_dir parameter is deprecated)
    os.environ["TENSORBOARD_LOGGING_DIR"] = f"{output_dir}/logs"

    # Data collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        # Renamed from evaluation_strategy in newer transformers
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Disabled - RTX 5090 sm_120 requires CPU training
        fp16=False,
        # Force CPU training for RTX 5090 compatibility
        use_cpu=True,
        report_to=["tensorboard"],
        push_to_hub=False,
        dataloader_num_workers=4,
        remove_unused_columns=True,
    )

    # Log training configuration
    logging.info("Training configuration:")
    logging.info(f"  Epochs: {epochs}")
    logging.info(f"  Batch size: {batch_size}")
    logging.info(f"  Learning rate: {learning_rate}")
    logging.info(f"  MLM probability: {mlm_probability}")
    logging.info(f"  Gradient accumulation: {gradient_accumulation_steps}")
    effective_batch_size = batch_size * gradient_accumulation_steps
    logging.info(f"  Effective batch size: {effective_batch_size}")
    logging.info(f"  FP16: {training_args.fp16}")

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        data_collator=data_collator,
    )

    # Train
    logging.info("Starting training...")
    train_result = trainer.train()

    # Log training results
    logging.info("Training completed!")
    logging.info(f"Training loss: {train_result.training_loss:.4f}")

    # Evaluate
    logging.info("Evaluating model...")
    eval_results = trainer.evaluate()
    logging.info("Evaluation results:")
    for key, value in eval_results.items():
        logging.info(f"  {key}: {value:.4f}")

    elapsed = datetime.now() - start_time
    logging.info(f"Total training time: {elapsed}")
    logging.info("="*80)

    return trainer


# ------------------------------------------------------------------------------
# 3. Model Save/Load Functions
# ------------------------------------------------------------------------------

def save_trained_model(
    model: AutoModelForMaskedLM,
    tokenizer: AutoTokenizer,
    save_path: str,
    is_peft_model: bool = True
) -> None:
    """
    Save the trained model and tokenizer.

    Args:
        model: Trained model (can be PEFT model)
        tokenizer: Tokenizer
        save_path: Directory to save the model
        is_peft_model: Whether this is a PEFT (LoRA) model
    """
    logging.info("="*80)
    logging.info(f"Saving model to: {save_path}")
    logging.info("="*80)

    os.makedirs(save_path, exist_ok=True)

    if is_peft_model:
        # Save LoRA adapters
        logging.info("Saving PEFT (LoRA) adapters...")
        model.save_pretrained(save_path)
        logging.info(f"PEFT adapters saved to: {save_path}")
    else:
        # Save full model
        logging.info("Saving full model...")
        model.save_pretrained(save_path)
        logging.info(f"Full model saved to: {save_path}")

    # Save tokenizer
    logging.info("Saving tokenizer...")
    tokenizer.save_pretrained(save_path)

    # Save training info
    info_path = os.path.join(save_path, "training_info.txt")
    with open(info_path, "w") as f:
        f.write(f"Model saved: {datetime.now()}\n")
        f.write(f"Is PEFT model: {is_peft_model}\n")
        f.write(f"Device: {next(model.parameters()).device}\n")

    logging.info(f"Model and tokenizer saved successfully to: {save_path}")
    logging.info("="*80)


def load_trained_model(
    model_path: str,
    base_model_name: Optional[str] = None,
    is_peft_model: bool = True,
    device: str = "cuda"  # Default to CUDA if available
) -> Tuple[AutoModelForMaskedLM, AutoTokenizer]:
    """
    Load a trained model and tokenizer.

    Args:
        model_path: Path to saved model
        base_model_name: Base model name (required if loading PEFT adapters)
        is_peft_model: Whether this is a PEFT model
        device: Device to load model on ("auto", "cuda", "cpu")

    Returns:
        Tuple of (model, tokenizer)
    """
    logging.info("="*80)
    logging.info(f"Loading model from: {model_path}")
    logging.info("="*80)

    # Load tokenizer
    logging.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if is_peft_model:
        if base_model_name is None:
            raise ValueError("base_model_name required for loading PEFT model")

        # Load base model
        logging.info(f"Loading base model: {base_model_name}")
        base_model = AutoModelForMaskedLM.from_pretrained(
            base_model_name,
            device_map=device,
            trust_remote_code=True
        )

        # Load PEFT adapters
        logging.info("Loading PEFT adapters...")
        model = PeftModel.from_pretrained(base_model, model_path)
        logging.info("PEFT model loaded successfully")
    else:
        # Load full model
        logging.info("Loading full model...")
        model = AutoModelForMaskedLM.from_pretrained(
            model_path,
            device_map=device,
            trust_remote_code=True
        )

    logging.info("Model loaded successfully")
    logging.info("="*80)

    return model, tokenizer


# ------------------------------------------------------------------------------
# 4. Embedding Generation Functions
# ------------------------------------------------------------------------------

def generate_embeddings(
    text: Union[str, List[str]],
    model_path: str,
    base_model_name: Optional[str] = None,
    is_peft_model: bool = True,
    pooling_strategy: str = "mean",
    normalize: bool = True,
    max_length: int = DEFAULT_MAX_LENGTH,
    device: Optional[str] = None
) -> np.ndarray:
    """
    Generate embeddings for text using the trained encoder model.

    Args:
        text: Single text or list of texts to embed
        model_path: Path to saved model
        base_model_name: Base model name (required for PEFT models)
        is_peft_model: Whether the model is a PEFT model
        pooling_strategy: How to pool token embeddings ("mean", "cls", "max")
        normalize: Whether to L2-normalize embeddings
        max_length: Maximum sequence length
        device: Device to use (None = auto-detect)

    Returns:
        numpy array of embeddings (shape: [num_texts, embedding_dim])
    """
    # Detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n------------------------------------\nUsing Device: {device}\n------------------------------------\n")

    logging.info(f"Generating embeddings on device: {device}")

    # Load model for embeddings (we need the encoder, not the MLM head)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if is_peft_model and base_model_name:
        base_model = AutoModel.from_pretrained(
            base_model_name,
            device_map=device,
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        # For non-PEFT or if loading MLM model, get base encoder
        try:
            model = AutoModel.from_pretrained(
                model_path,
                device_map=device,
                trust_remote_code=True
            )
        except Exception as e:
            e = e.message if hasattr(e, "message") else str(e)
            # If saved as MLM model, load and extract encoder
            mlm_model = AutoModelForMaskedLM.from_pretrained(
                model_path,
                device_map=device,
                trust_remote_code=True
            )
            # Most models have the encoder as .base_model or similar
            model = getattr(mlm_model, 'base_model', mlm_model)

    model.eval()

    # Ensure text is a list
    if isinstance(text, str):
        texts = [text]
    else:
        texts = text

    # Tokenize
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        # Get hidden states (last layer)
        # [batch_size, seq_len, hidden_dim]
        hidden_states = outputs.last_hidden_state

        # Apply pooling strategy
        if pooling_strategy == "mean":
            # Mean pooling (accounting for padding)
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            mask_expanded = attention_mask.expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
        elif pooling_strategy == "cls":
            # Use [CLS] token embedding (first token)
            embeddings = hidden_states[:, 0, :]
        elif pooling_strategy == "max":
            # Max pooling
            embeddings = torch.max(hidden_states, dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")

        # Normalize if requested
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Convert to numpy
        embeddings_np = embeddings.cpu().numpy()

    logging.info(f"Generated embeddings shape: {embeddings_np.shape}")

    return embeddings_np


def get_embedding_info(
    model_path: str,
    base_model_name: Optional[str] = None
) -> Dict[str, any]:
    """
    Get information about embedding dimensions and model configuration.

    Args:
        model_path: Path to saved model
        base_model_name: Base model name (for PEFT models)

    Returns:
        Dictionary with embedding info
    """
    # tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Try to load config
    try:
        if base_model_name:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(base_model_name)
        else:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_path)

        info = {
            "embedding_size": config.hidden_size,
            "max_position_embeddings": config.max_position_embeddings,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "vocab_size": config.vocab_size,
            "model_type": config.model_type
        }

        return info

    except Exception as err:
        logging.error(f"Could not load config: {err}")
        return {"error": str(err)}


# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------

def setup_logging(log_file: Optional[str] = None, level=logging.INFO):
    """Setup logging configuration"""
    handlers = [logging.StreamHandler()]

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


# ------------------------------------------------------------------------------
# End of Module
# ------------------------------------------------------------------------------
