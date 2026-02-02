"""
Quick Test Script for MLM Training Setup

This script performs a quick test to verify everything is set up correctly
without running a full training session.

Usage:
    uv run python test_setup.py

Author: MLM Training Project
Date: February 2026
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM


def check_environment():
    """Check Python environment and dependencies"""
    print("\n" + "="*80)
    print("ENVIRONMENT CHECK")
    print("="*80 + "\n")
    
    # Python version
    print(f"✓ Python version: {sys.version.split()[0]}")
    
    # PyTorch
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    
    # CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠ CUDA not available - training will be VERY slow on CPU")
    
    # Check other dependencies
    try:
        import transformers
        print(f"✓ Transformers version: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not installed")
        return False
    
    try:
        import peft
        print(f"✓ PEFT version: {peft.__version__}")
    except ImportError:
        print("❌ PEFT not installed - LoRA training will not work")
        return False
    
    try:
        import bitsandbytes
        print(f"✓ BitsAndBytes available")
    except ImportError:
        print("⚠ BitsAndBytes not installed - QLoRA will not work")
    
    try:
        import datasets
        print(f"✓ Datasets version: {datasets.__version__}")
    except ImportError:
        print("❌ Datasets not installed")
        return False
    
    print("\n✅ All core dependencies installed!")
    return True


def check_directories():
    """Check required directories exist"""
    print("\n" + "="*80)
    print("DIRECTORY CHECK")
    print("="*80 + "\n")
    
    dirs = {
        "data/books": "PDF books location",
        "models": "Downloaded models",
        "results": "Training outputs",
        "logs": "Log files"
    }
    
    all_exist = True
    for dir_path, description in dirs.items():
        exists = os.path.exists(dir_path)
        status = "✓" if exists else "⚠"
        print(f"{status} {dir_path:20s} - {description}")
        if not exists:
            all_exist = False
            print(f"    Creating directory: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
    
    # Check if any PDF files exist
    books_dir = "data/books"
    if os.path.exists(books_dir):
        pdf_files = []
        for root, dirs, files in os.walk(books_dir):
            pdf_files.extend([f for f in files if f.endswith('.pdf')])
        
        if pdf_files:
            print(f"\n✓ Found {len(pdf_files)} PDF file(s) in {books_dir}")
        else:
            print(f"\n⚠ No PDF files found in {books_dir}")
            print(f"  Add your PDF books to this directory before training")
    
    return True


def check_models():
    """Check if any models are downloaded"""
    print("\n" + "="*80)
    print("MODEL CHECK")
    print("="*80 + "\n")
    
    models_dir = "models"
    recommended_models = [
        "deberta-v3-base",
        "electra-base",
        "roberta-base",
        "bert-base-uncased"
    ]
    
    downloaded_models = []
    if os.path.exists(models_dir):
        for model_name in recommended_models:
            model_path = os.path.join(models_dir, model_name)
            if os.path.exists(model_path):
                # Check if it has required files
                has_config = os.path.exists(os.path.join(model_path, "config.json"))
                has_model = any(
                    os.path.exists(os.path.join(model_path, f))
                    for f in ["pytorch_model.bin", "model.safetensors"]
                )
                
                if has_config and has_model:
                    print(f"✓ {model_name}")
                    downloaded_models.append(model_name)
                else:
                    print(f"⚠ {model_name} - incomplete")
            else:
                print(f"  {model_name} - not downloaded")
    
    if downloaded_models:
        print(f"\n✓ {len(downloaded_models)} model(s) ready to use")
    else:
        print("\n⚠ No models downloaded yet")
        print("  Run: uv run python download_models.py --model deberta-v3-base")
    
    return len(downloaded_models) > 0


def test_model_loading():
    """Test loading a model if available"""
    print("\n" + "="*80)
    print("MODEL LOADING TEST")
    print("="*80 + "\n")
    
    # Try to find a model
    models_dir = "models"
    test_models = [
        "deberta-v3-base",
        "electra-base",
        "roberta-base",
        "bert-base-uncased"
    ]
    
    model_path = None
    for model in test_models:
        path = os.path.join(models_dir, model)
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("⚠ No model available to test")
        print("  Download a model first: uv run python download_models.py --all")
        return False
    
    try:
        print(f"Testing model: {os.path.basename(model_path)}")
        print(f"Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"✓ Tokenizer loaded")
        
        print(f"Loading model...")
        model = AutoModelForMaskedLM.from_pretrained(model_path)
        print(f"✓ Model loaded")
        
        # Test tokenization
        test_text = "Machine learning is transforming artificial intelligence."
        print(f"\nTest tokenization: '{test_text}'")
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"✓ Tokenized to {tokens['input_ids'].shape[1]} tokens")
        
        # Test forward pass
        print(f"Testing forward pass...")
        with torch.no_grad():
            outputs = model(**tokens)
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {outputs.logits.shape}")
        
        print(f"\n✅ Model loading and inference working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing model: {e}")
        return False


def test_dataset_loading():
    """Test dataset loading from PDF files"""
    print("\n" + "="*80)
    print("DATASET LOADING TEST")
    print("="*80 + "\n")
    
    books_dir = "data/books"
    
    # Check if PDFs exist
    if not os.path.exists(books_dir):
        print(f"⚠ Directory {books_dir} does not exist")
        return False
    
    pdf_count = 0
    for root, dirs, files in os.walk(books_dir):
        pdf_count += len([f for f in files if f.endswith('.pdf')])
    
    if pdf_count == 0:
        print(f"⚠ No PDF files found in {books_dir}")
        print(f"  Add PDF books to this directory to test dataset preparation")
        return False
    
    print(f"Found {pdf_count} PDF file(s)")
    
    try:
        from src.ingest import read_files_directory
        
        print(f"Testing PDF reading (limiting to first 2 chunks)...")
        chunks, files_count, problem_files = read_files_directory(
            books_dir,
            chunk_size=512,
            chunk_overlap=50
        )
        
        if chunks:
            print(f"✓ Successfully read {files_count} file(s)")
            print(f"✓ Generated {len(chunks)} text chunks")
            if problem_files:
                print(f"⚠ Failed to read {len(problem_files)} file(s)")
            
            # Show sample
            if len(chunks) > 0:
                print(f"\nSample chunk (first 200 chars):")
                print(f"  '{chunks[0][:200]}...'")
            
            print(f"\n✅ Dataset loading working correctly!")
            return True
        else:
            print(f"❌ No chunks extracted from PDFs")
            return False
            
    except Exception as e:
        print(f"❌ Error testing dataset loading: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_next_steps(checks_passed):
    """Print next steps based on what passed"""
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80 + "\n")
    
    if all(checks_passed.values()):
        print("✅ All checks passed! You're ready to start training.")
        print("\nTo begin training:")
        print("  1. uv run python example_mlm_training.py")
        print("\nTo monitor training:")
        print("  2. tensorboard --logdir=results")
        print("  3. Open http://localhost:6006 in your browser")
        
    else:
        if not checks_passed['environment']:
            print("❌ Fix environment issues:")
            print("  uv sync")
        
        if not checks_passed['models']:
            print("\n📥 Download a model:")
            print("  uv run python download_models.py --model deberta-v3-base")
        
        if not checks_passed['dataset']:
            print("\n📚 Add PDF books:")
            print(f"  Place your PDF files in: data/books/")
    
    print("\n" + "="*80 + "\n")


def main():
    """Run all checks"""
    print("\n" + "="*80)
    print("MLM TRAINING SETUP TEST")
    print("="*80)
    
    checks = {
        'environment': check_environment(),
        'directories': check_directories(),
        'models': check_models(),
        'model_loading': False,
        'dataset': False
    }
    
    # Only test model loading if models exist
    if checks['models']:
        checks['model_loading'] = test_model_loading()
    
    # Only test dataset if PDFs exist
    if os.path.exists("data/books"):
        checks['dataset'] = test_dataset_loading()
    
    print_next_steps(checks)
    
    # Summary
    passed = sum(1 for v in checks.values() if v)
    total = len(checks)
    
    print(f"Summary: {passed}/{total} checks passed")
    
    if all(checks.values()):
        print("\n🎉 Everything is ready for MLM training!")
        return 0
    else:
        print("\n⚠ Some checks need attention - see above for details")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
