"""
Complete Example: MLM Training Pipeline

This script demonstrates the complete workflow for MLM training:
1. Prepare dataset from PDF books
2. Setup and train encoder model with LoRA/QLoRA
3. Save the trained model
4. Generate embeddings from text

Author: MLM Training Project
Date: February 2026
"""

import os
import logging
from datetime import datetime

from src.mlm_trainer import (
    prepare_mlm_dataset,
    setup_model_for_mlm_training,
    train_mlm_model,
    save_trained_model,
    load_trained_model,
    generate_embeddings,
    get_embedding_info,
    setup_logging
)


# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

# Directories
DATA_DIR = "data/books"  # Your PDF books directory
MODELS_DIR = "models"
RESULTS_DIR = "results"
LOGS_DIR = "logs"

# Model selection - Choose one:
# "microsoft/deberta-v3-base"     # Recommended - best quality
# "google/electra-base-discriminator"  # Fast and efficient
# "FacebookAI/roberta-base"       # Classic, reliable

MODEL_NAME = "microsoft/deberta-v3-base"
MODEL_SHORT_NAME = MODEL_NAME.split("/")[-1]

# Training parameters
MAX_LENGTH = 512
CHUNK_SIZE = 2048
CHUNK_OVERLAP = 200
TEST_SPLIT = 0.1
MAX_CHUNKS = None  # None = use all chunks, or set a number for testing

EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 2e-4
MLM_PROBABILITY = 0.15

# PEFT parameters
# NOTE: Disabling QLoRA for RTX 5090 compatibility (sm_120 not fully supported yet)
# You can enable these once PyTorch has full sm_120 support
USE_QLORA = False  # Disabled due to RTX 5090 compute capability
USE_LORA = True    # LoRA without quantization still works
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1

# Paths
BASE_MODEL_PATH = os.path.join(MODELS_DIR, MODEL_SHORT_NAME)
TRAINED_MODEL_PATH = os.path.join(MODELS_DIR, f"{MODEL_SHORT_NAME}-mlm-trained-{datetime.now().strftime('%Y%m%d')}")
OUTPUT_DIR = os.path.join(RESULTS_DIR, f"training-{MODEL_SHORT_NAME}-{datetime.now().strftime('%Y%m%d-%H%M')}")


# ------------------------------------------------------------------------------
# Main Training Pipeline
# ------------------------------------------------------------------------------

def main():
    """Complete MLM training pipeline"""
    
    # Setup logging
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_file = os.path.join(LOGS_DIR, f"mlm_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    setup_logging(log_file=log_file)
    
    logging.info("="*80)
    logging.info("MLM TRAINING PIPELINE")
    logging.info("="*80)
    logging.info(f"Model: {MODEL_NAME}")
    logging.info(f"Data Directory: {DATA_DIR}")
    logging.info(f"Output Directory: {OUTPUT_DIR}")
    logging.info("="*80)
    
    # ------------------------------------------------------------------------
    # Step 1: Prepare Dataset
    # ------------------------------------------------------------------------
    logging.info("\n" + "="*80)
    logging.info("STEP 1: PREPARING DATASET")
    logging.info("="*80 + "\n")
    
    datasets, tokenizer = prepare_mlm_dataset(
        data_dir=DATA_DIR,
        model_name=BASE_MODEL_PATH,  # Use local model if already downloaded
        max_length=MAX_LENGTH,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        test_split=TEST_SPLIT,
        max_chunks=MAX_CHUNKS
    )
    
    logging.info(f"✓ Dataset prepared successfully")
    logging.info(f"  Training samples: {len(datasets['train'])}")
    logging.info(f"  Test samples: {len(datasets['test'])}")
    
    # ------------------------------------------------------------------------
    # Step 2: Setup Model
    # ------------------------------------------------------------------------
    logging.info("\n" + "="*80)
    logging.info("STEP 2: SETTING UP MODEL")
    logging.info("="*80 + "\n")
    
    model, is_quantized = setup_model_for_mlm_training(
        model_name=BASE_MODEL_PATH,
        use_qlora=USE_QLORA,
        use_lora=USE_LORA,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        load_in_4bit=True,  # 4-bit quantization for QLoRA
        load_in_8bit=False
    )
    
    logging.info(f"✓ Model setup complete")
    logging.info(f"  Quantized: {is_quantized}")
    logging.info(f"  Using LoRA: {USE_LORA}")
    
    # ------------------------------------------------------------------------
    # Step 3: Train Model
    # ------------------------------------------------------------------------
    logging.info("\n" + "="*80)
    logging.info("STEP 3: TRAINING MODEL")
    logging.info("="*80 + "\n")
    
    trainer = train_mlm_model(
        model=model,
        tokenizer=tokenizer,
        datasets=datasets,
        output_dir=OUTPUT_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        mlm_probability=MLM_PROBABILITY,
        gradient_accumulation_steps=4,
        fp16=False  # Disabled for RTX 5090 sm_120 compatibility
    )
    
    logging.info(f"✓ Training complete")
    
    # ------------------------------------------------------------------------
    # Step 4: Save Model
    # ------------------------------------------------------------------------
    logging.info("\n" + "="*80)
    logging.info("STEP 4: SAVING MODEL")
    logging.info("="*80 + "\n")
    
    save_trained_model(
        model=model,
        tokenizer=tokenizer,
        save_path=TRAINED_MODEL_PATH,
        is_peft_model=USE_LORA
    )
    
    logging.info(f"✓ Model saved to: {TRAINED_MODEL_PATH}")
    
    # ------------------------------------------------------------------------
    # Step 5: Test Embeddings Generation
    # ------------------------------------------------------------------------
    logging.info("\n" + "="*80)
    logging.info("STEP 5: TESTING EMBEDDINGS")
    logging.info("="*80 + "\n")
    
    # Get embedding info
    info = get_embedding_info(
        model_path=TRAINED_MODEL_PATH,
        base_model_name=BASE_MODEL_PATH
    )
    
    logging.info("Embedding Information:")
    for key, value in info.items():
        logging.info(f"  {key}: {value}")
    
    # Generate test embeddings
    test_texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing enables computers to understand text."
    ]
    
    embeddings = generate_embeddings(
        text=test_texts,
        model_path=TRAINED_MODEL_PATH,
        base_model_name=BASE_MODEL_PATH,
        is_peft_model=USE_LORA,
        pooling_strategy="mean",
        normalize=True
    )
    
    logging.info(f"✓ Generated embeddings for {len(test_texts)} texts")
    logging.info(f"  Embedding shape: {embeddings.shape}")
    logging.info(f"  Embedding dimension: {embeddings.shape[1]}")
    
    # ------------------------------------------------------------------------
    # Completion
    # ------------------------------------------------------------------------
    logging.info("\n" + "="*80)
    logging.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    logging.info("="*80)
    logging.info(f"\nTrained model location: {TRAINED_MODEL_PATH}")
    logging.info(f"Training outputs location: {OUTPUT_DIR}")
    logging.info(f"Log file: {log_file}")
    logging.info("\n" + "="*80 + "\n")


# ------------------------------------------------------------------------------
# Example: Just Generate Embeddings from Existing Model
# ------------------------------------------------------------------------------

def example_generate_embeddings():
    """Example of generating embeddings from a trained model"""
    
    # Your trained model path
    trained_model_path = TRAINED_MODEL_PATH
    base_model_path = BASE_MODEL_PATH
    
    # Texts to embed
    texts = [
        "Artificial intelligence is transforming technology.",
        "Natural language processing helps computers understand human language.",
        "Machine learning algorithms learn from data."
    ]
    
    # Generate embeddings
    embeddings = generate_embeddings(
        text=texts,
        model_path=trained_model_path,
        base_model_name=base_model_path,
        is_peft_model=True,
        pooling_strategy="mean",  # "mean", "cls", or "max"
        normalize=True
    )
    
    print(f"Generated embeddings shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Use embeddings for similarity, clustering, etc.
    from sklearn.metrics.pairwise import cosine_similarity
    
    similarities = cosine_similarity(embeddings)
    print("\nCosine Similarities:")
    for i, text1 in enumerate(texts):
        for j, text2 in enumerate(texts):
            if i < j:
                print(f"  '{text1[:40]}...' <-> '{text2[:40]}...': {similarities[i][j]:.4f}")


# ------------------------------------------------------------------------------
# Example: Load and Continue Training
# ------------------------------------------------------------------------------

def example_continue_training():
    """Example of loading a trained model and continuing training"""
    
    # Load the model
    model, tokenizer = load_trained_model(
        model_path=TRAINED_MODEL_PATH,
        base_model_name=BASE_MODEL_PATH,
        is_peft_model=True
    )
    
    # Prepare new dataset (or use existing)
    datasets, _ = prepare_mlm_dataset(
        data_dir=DATA_DIR,
        model_name=BASE_MODEL_PATH,
        max_length=MAX_LENGTH
    )
    
    # Continue training
    trainer = train_mlm_model(
        model=model,
        tokenizer=tokenizer,
        datasets=datasets,
        output_dir=OUTPUT_DIR + "-continued",
        epochs=5,  # Additional epochs
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE * 0.5  # Lower learning rate for fine-tuning
    )
    
    # Save again
    save_trained_model(
        model=model,
        tokenizer=tokenizer,
        save_path=TRAINED_MODEL_PATH + "-continued",
        is_peft_model=True
    )


# ------------------------------------------------------------------------------
# Run Main Pipeline
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # Run the complete training pipeline
    main()
    
    # Uncomment to run other examples:
    # example_generate_embeddings()
    # example_continue_training()
