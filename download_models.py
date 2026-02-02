"""
Model Download Script

This script downloads recommended encoder models from HuggingFace
and saves them to the local models directory.

Usage:
    uv run python download_models.py

Or download specific model:
    uv run python download_models.py --model deberta-v3-base

Author: MLM Training Project
Date: February 2026
"""

import os
import argparse
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig


# Model configurations
MODELS = {
    "deberta-v3-base": {
        "hf_name": "microsoft/deberta-v3-base",
        "description": "DeBERTa-v3 Base - Best quality (184M params, 768 dim)",
        "recommended": True
    },
    "deberta-v3-large": {
        "hf_name": "microsoft/deberta-v3-large",
        "description": "DeBERTa-v3 Large - Maximum quality (434M params, 1024 dim)",
        "recommended": False
    },
    "electra-base": {
        "hf_name": "google/electra-base-discriminator",
        "description": "ELECTRA Base - Fast and efficient (110M params, 768 dim)",
        "recommended": True
    },
    "roberta-base": {
        "hf_name": "FacebookAI/roberta-base",
        "description": "RoBERTa Base - Classic, reliable (125M params, 768 dim)",
        "recommended": True
    },
    "bert-base": {
        "hf_name": "google-bert/bert-base-uncased",
        "description": "BERT Base - Original (110M params, 768 dim)",
        "recommended": False
    }
}

DEFAULT_MODELS_DIR = "models"


def download_model(model_key: str, models_dir: str = DEFAULT_MODELS_DIR, force: bool = False):
    """
    Download a model from HuggingFace and save locally.
    
    Args:
        model_key: Key from MODELS dict
        models_dir: Directory to save models
        force: Force re-download even if exists
    """
    if model_key not in MODELS:
        print(f"❌ Unknown model: {model_key}")
        print(f"Available models: {', '.join(MODELS.keys())}")
        return False
    
    model_config = MODELS[model_key]
    hf_name = model_config["hf_name"]
    description = model_config["description"]
    
    save_path = os.path.join(models_dir, model_key)
    
    # Check if already exists
    if os.path.exists(save_path) and not force:
        print(f"✓ Model '{model_key}' already exists at {save_path}")
        print(f"  Use --force to re-download")
        return True
    
    print(f"\n{'='*80}")
    print(f"Downloading: {model_key}")
    print(f"HuggingFace: {hf_name}")
    print(f"Description: {description}")
    print(f"Save path: {save_path}")
    print(f"{'='*80}\n")
    
    try:
        # Create directory
        os.makedirs(save_path, exist_ok=True)
        
        # Download config first to get model info
        print(f"📥 Downloading config...")
        config = AutoConfig.from_pretrained(hf_name)
        
        # Display model info
        print(f"\nModel Information:")
        print(f"  Model type: {config.model_type}")
        print(f"  Hidden size (embedding dim): {config.hidden_size}")
        print(f"  Max position embeddings: {config.max_position_embeddings}")
        print(f"  Number of layers: {config.num_hidden_layers}")
        print(f"  Number of attention heads: {config.num_attention_heads}")
        print(f"  Vocabulary size: {config.vocab_size}")
        
        # Download tokenizer
        print(f"\n📥 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(hf_name)
        tokenizer.save_pretrained(save_path)
        print(f"✓ Tokenizer saved")
        
        # Download model
        print(f"\n📥 Downloading model (this may take a few minutes)...")
        model = AutoModelForMaskedLM.from_pretrained(hf_name)
        model.save_pretrained(save_path)
        print(f"✓ Model saved")
        
        # Calculate model size
        param_count = sum(p.numel() for p in model.parameters())
        param_count_millions = param_count / 1_000_000
        
        print(f"\n{'='*80}")
        print(f"✅ SUCCESS!")
        print(f"{'='*80}")
        print(f"Model: {model_key}")
        print(f"Location: {save_path}")
        print(f"Parameters: {param_count_millions:.1f}M")
        print(f"Embedding dimension: {config.hidden_size}")
        print(f"{'='*80}\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR downloading {model_key}:")
        print(f"  {str(e)}\n")
        return False


def download_all_recommended(models_dir: str = DEFAULT_MODELS_DIR, force: bool = False):
    """Download all recommended models"""
    
    recommended = [key for key, config in MODELS.items() if config.get("recommended", False)]
    
    print(f"\n{'='*80}")
    print(f"Downloading {len(recommended)} recommended models")
    print(f"{'='*80}\n")
    
    results = {}
    for model_key in recommended:
        success = download_model(model_key, models_dir, force)
        results[model_key] = success
    
    # Summary
    print(f"\n{'='*80}")
    print(f"DOWNLOAD SUMMARY")
    print(f"{'='*80}")
    
    successful = [k for k, v in results.items() if v]
    failed = [k for k, v in results.items() if not v]
    
    if successful:
        print(f"\n✅ Successfully downloaded ({len(successful)}):")
        for model_key in successful:
            print(f"  - {model_key}")
    
    if failed:
        print(f"\n❌ Failed to download ({len(failed)}):")
        for model_key in failed:
            print(f"  - {model_key}")
    
    print(f"\n{'='*80}\n")
    
    return results


def list_available_models():
    """List all available models"""
    
    print(f"\n{'='*80}")
    print(f"AVAILABLE MODELS")
    print(f"{'='*80}\n")
    
    for key, config in MODELS.items():
        recommended = "⭐ RECOMMENDED" if config.get("recommended", False) else ""
        print(f"{key}")
        print(f"  HuggingFace: {config['hf_name']}")
        print(f"  Description: {config['description']}")
        if recommended:
            print(f"  {recommended}")
        print()
    
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download encoder models from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all recommended models
  python download_models.py --all
  
  # Download specific model
  python download_models.py --model deberta-v3-base
  
  # List available models
  python download_models.py --list
  
  # Download to custom directory
  python download_models.py --model electra-base --dir my_models
  
  # Force re-download
  python download_models.py --model roberta-base --force
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        help="Model key to download (e.g., deberta-v3-base)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all recommended models"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=DEFAULT_MODELS_DIR,
        help=f"Directory to save models (default: {DEFAULT_MODELS_DIR})"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if model exists"
    )
    
    args = parser.parse_args()
    
    # List models
    if args.list:
        list_available_models()
        return
    
    # Download all recommended
    if args.all:
        download_all_recommended(args.dir, args.force)
        return
    
    # Download specific model
    if args.model:
        download_model(args.model, args.dir, args.force)
        return
    
    # No arguments - show help and download recommended
    print("No arguments provided. Downloading recommended models...\n")
    download_all_recommended(args.dir, args.force)


if __name__ == "__main__":
    main()
