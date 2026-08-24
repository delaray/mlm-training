# Install dependencies
# !pip install transformers
# !pip install datasets
# !pip install accelerate
# !pip install langchain_text_splitters
# !pip install google-cloud-storage
# !pip install optuna
# !pip install pymupdf
# !pip install python-pptx
# !pip install ipywidgets

# Standard Python
from src.train import (
    DEFAULT_DATA_DIR,
    DEFAULT_MODELS_DIR,
    DEFAULT_MODEL_NAME,
    DEFAULT_RESULTS_DIR,
    get_trainer,
    initialize_logging,
    prepare_datasets,
    print_and_log,
)
from src.ingest import group_texts, read_files_directory
from transformers import logging as transformers_logging
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForMaskedLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datasets import Dataset
import torch
import os
import optuna
import itertools
import pickle
import datetime
import argparse
import logging
from datetime import date, datetime
from itertools import chain
from functools import partial
from typing import List

print(f'Current directory: {os.getcwd()}')

# General imports

# Project Imports


# Default directories & training parameters
DEFAULT_DATA_DIR = 'data'
DEFAULT_RESULTS_DIR = 'results'
DEFAULT_LOGS_DIRECTORY = 'logs'
DEFAULT_MODELS_DIR = 'models'

DEFAULT_TRIALS = 12
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 16

DEFAULT_LEARNING_RATE = 0.0002
DEFAULT_WEIGHT_DECAY = 0.004


# Loading Chunks

max_token_length = 256
max_sequence_length = 512

# Don't exceed model's max sequence length
max_chunk_size = min(max_token_length * 4, max_sequence_length)


def load_chunks(data_dir=DEFAULT_DATA_DIR):

    print("\nReading books...")
    books_path = os.path.join(data_dir, 'books')
    print(f'Books folder path: {books_path}')
    book_chunks = read_files_directory(books_path, chunk_size=max_chunk_size,
                                       chunk_overlap=0)
    book_chunks, _, _ = book_chunks

    print(f'\nType of a single book chunk: {type(book_chunks[0])}')
    print(f'Total books chunks: {len(book_chunks)}')

    print("\nReading intouch consolidated documents...")
    intouch_path = os.path.join(data_dir, 'intouch_documents_consolidated')
    print(f'Intouch folder path: {intouch_path}')
    intouch_chunks = read_files_directory(intouch_path, chunk_size=max_chunk_size,
                                          chunk_overlap=0)
    intouch_chunks, _, _ = intouch_chunks

    print(f'\nType of a single intouch chunk: {type(intouch_chunks[0])}')
    print(f'Total intouch chunks: {len(intouch_chunks)}')

    return book_chunks, intouch_chunks


# Optuna Objective function

def mlm_objective(trial: optuna.Trial, datasets=None, data_collator=None,
                  epochs=DEFAULT_EPOCHS, tokenizer=None,
                  model_name=DEFAULT_MODEL_NAME,
                  models_dir=DEFAULT_MODELS_DIR):

    model_path = os.path.join(models_dir, model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_path)

    if tokenizer:
        # Apparently this solves the error:
        # "RuntimeError: CUDA error: device-side assert triggered"
        model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir=f"{model_name}-further-trained-{datetime.now()}",
        eval_strategy="epoch",
        weight_decay=trial.suggest_float(
            "weight_decay", log=True, low=0.001, high=0.1),
        learning_rate=trial.suggest_float(
            "learning_rate", log=True, low=1e-5, high=2e-4),
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        disable_tqdm=True,
    )

    trainer = get_trainer(model, training_args, datasets, data_collator)
    result = trainer.train()

    return result.training_loss


def optimize_hyperparameters(model_name, epochs: int = DEFAULT_EPOCHS, batch_size: int = 16,
                             max_chunks=None, trials=DEFAULT_TRIALS,
                             models_dir=DEFAULT_MODELS_DIR,
                             results_dir=DEFAULT_RESULTS_DIR):

    # device = "cpu"
    model_path = os.path.join(models_dir, model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print_and_log("\nPreparing data sets...")
    tokenized_datasets, chunks, _, _ = prepare_datasets(
        data_path='data/books', max_chunks=max_chunks
    )
    if tokenized_datasets is None:
        raise ValueError("No datasets were prepared")

    print(f"\nTotal number of chunks loaded: {len(chunks)}\n")

    lm_datasets = tokenized_datasets.map(group_texts,
                                         batched=True,
                                         batch_size=1000,
                                         num_proc=4)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=0.15)

    # Define a partial fn to supply tokenizer, model name, datasets, etc., arguments to the obj function
    objective = partial(mlm_objective, tokenizer=tokenizer, datasets=lm_datasets, data_collator=data_collator,
                        model_name=model_name, models_dir=models_dir)

    print_and_log("\nRunning hyperparam search...")
    # Create Optuna study and then run trials
    study = optuna.create_study(
        study_name="hyper-parameter-search", direction="minimize")

    study_start_time = datetime.now()
    try:
        study.optimize(objective, n_trials=trials)

    except Exception as e:
        print(f"\nError running Optuna study for {model_name}:\n{e}\n")
        print(
            f"\nWARNING: The Optuna study for {model_name} ended prematurely.")
        print(
            f"\nOnly completed {len(study.trials)} trials out of {trials} trials.\n")

    # Study duration
    study_end_time = datetime.now()
    print_and_log(f"\nThe study took: {study_end_time - study_start_time}")

    # Log the best trial results
    print_and_log(f"\nBest accuracy: {study.best_trial}")

    # Save the study to be loaded later if desired
    save_optuna_study(study, model_name, results_dir=results_dir)

    return study


def save_optuna_study(
    study: optuna.Study,
    model_name: str,
    results_dir: str = DEFAULT_RESULTS_DIR,
) -> None:
    """Persist an Optuna study for later analysis."""
    os.makedirs(results_dir, exist_ok=True)
    study_path = os.path.join(results_dir, f"{model_name}-optuna-study.pkl")
    with open(study_path, "wb") as study_file:
        pickle.dump(study, study_file)


# ------------------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Optimize MLM hyperparameters")
    parser.add_argument("name")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--chunks", type=int, default=None)
    parser.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    model_name, trials, epochs, chunks = args.name, args.trials, args.epochs, args.chunks
    models_dir, results_dir = args.models_dir, args.results_dir

    # Set up Logging configuration and filename
    logger = initialize_logging(model_name, epochs)

    # Load base model and tokenizer
    print_and_log("\nLoading model & tokenizer...")
    model_path = os.path.join(DEFAULT_MODELS_DIR, model_name)

    print_and_log(
        f"\nStarting Optuna study for mode (model_name) with the following parameters:")
    print_and_log(f"\nNumber of Optuna trials: {trials}")
    print_and_log(f"\nNumber of epochs for each trial: {epochs}")
    print_and_log(
        f"\nNumber of data chunks for each trial: {chunks or 'All Chunks'}")

    optimize_hyperparameters(
        model_name=model_name, trials=trials, epochs=epochs, batch_size=16, max_chunks=chunks,
        models_dir=models_dir, results_dir=results_dir
    )
