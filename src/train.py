import os
import pickle
import argparse
import logging
from datetime import date, datetime
from typing import Union
import matplotlib.pyplot as plt
from datasets import Dataset

from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from transformers import logging as transformers_logging

from src.ingest import read_files_directory


# ----------------------------------------------------------------------------
# Global Defaults & Settings
# -----------------------------------------------------------------------------

DEFAULT_DATA_DIR =  "C:/education/books/Computer Science/Artificial Intelligence/Generative AI"
DEFAULT_RESULTS_DIR = 'results'
DEFAULT_LOGS_DIRECTORY = 'logs'
DEFAULT_MODELS_DIR = 'models'
DEFAULT_MODEL_NAME = 'bert-base-uncased'

DEFAULT_MAX_TOKEN_LENGTH = 512
DEFAULT_MAX_CHUNK_SIZE = 4 * DEFAULT_MAX_TOKEN_LENGTH

DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.0002
DEFAULT_WEIGHT_DECAY = 0.004

folder_intouch_docs = 'intouch_documents_consolidated'
folder_intouch_docs_path = os.path.join(DEFAULT_DATA_DIR, folder_intouch_docs)

max_token_length = 512
max_chunk_size = max_token_length * 4
chunk_overlap = 0

today = str(date.today())

model_name = DEFAULT_MODEL_NAME
model_path = os.path.join(DEFAULT_MODELS_DIR, model_name)
model_save_path = f"{model_path}-trained-{today}"

DEFAULT_TOKENIZER = AutoTokenizer.from_pretrained(model_path)
DEFAULT_MODEL = AutoModelForMaskedLM.from_pretrained(model_path)

MODEL = DEFAULT_MODEL
TOKENIZER = DEFAULT_TOKENIZER


# ------------------------------------------------------------------------------
# Logging Setup
# ------------------------------------------------------------------------------

def initialize_logging(model_name, epochs, logs_path=DEFAULT_LOGS_DIRECTORY):

    logger = logging.getLogger(__name__)

    train_date = datetime.now().strftime('%Y-%m-%d')
    log_filename = datetime.now().strftime(
        f"train-log-{model_name}-{train_date}-epochs-{epochs}.log")

    log_filepath = os.path.join(logs_path, log_filename)

    logging.basicConfig(
        encoding='utf-8',
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_filepath, mode='w'),
            logging.StreamHandler()
        ]
    )

    transformers_logging.set_verbosity_info()
    transformers_logging.enable_default_handler()
    transformers_logging.enable_explicit_format()

    return logger


def print_and_log(msg: str):
    """Print <msg> to console and log <msg> to log file"""
    print(msg)
    logging.info(msg)


# ------------------------------------------------------------------------------
# Data Preparation
# ------------------------------------------------------------------------------


def tokenize_function(examples, tokenizer=DEFAULT_TOKENIZER):
    return tokenizer(examples["text"],
                     truncation=True,
                     max_length=max_token_length,
                     return_overflowing_tokens=True)


# ------------------------------------------------------------------------------

def prepare_datasets(data_path=DEFAULT_DATA_DIR, 
                     max_chunks: Union[int | None] = None) -> dict:

    prep_start_time = datetime.now()
    logging.info("Preparing data sets")

    # Load and chunk data
    chunks, files_count, problem_files =\
        read_files_directory(data_path, chunk_size=max_chunk_size,
                             chunk_overlap=0)
    chunks = chunks[:max_chunks] if max_chunks is not None else chunks

    if len(chunks) > 0:
        data_dict = {"text": chunks}

        dataset = Dataset.from_dict(data_dict)
        dataset = dataset.train_test_split(test_size=0.1, seed=2024)

        data_sets = dataset.map(tokenize_function,
                                batched=True,
                                num_proc=4,
                                remove_columns=["text"])

        # Total number of chunks
        n_chunks = len(data_sets['train']) + len(data_sets['test'])

        # Log dataset dimensions
        logging.info(f"A total of {n_chunks} have been prepared")
        logging.info(f"The max length tokens is {max_token_length}")
        logging.info(f"{len(data_sets['train'])} will be used for training")

        # Log total data preparation time
        prep_end_time = datetime.now()
        logging.info(
            f"Data preparation took {prep_end_time - prep_start_time}")

        return data_sets, chunks, files_count, problem_files

    else:
        logging.info(
            'The list of chunks is empty, there is no dataset to prepare.')
        return None, [], 0, []


# ------------------------------------------------------------------------------

def get_training_args(model_name=DEFAULT_MODEL_NAME,
                      epochs: float = DEFAULT_EPOCHS,
                      batch_size: int = DEFAULT_BATCH_SIZE,
                      learning_rate: float = DEFAULT_LEARNING_RATE,
                      weight_decay: float = DEFAULT_WEIGHT_DECAY):

    # Divide batch size by number of GPU’s
    per_device_batch_size = int(batch_size / 4)

    training_args = TrainingArguments(
        model_name,
        eval_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        push_to_hub=False,
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=8,
    )

    return training_args


class CustomTrainer(Trainer):

    def compute_loss(self, *args, **kwargs):
        # Perform evaluation
        result = super().compute_loss(*args, **kwargs)
        # Log Training Loss
        # print_and_log(f"Epoch {self.state.epoch}: Train Loss: {result[0]}")
        return result

    def evaluate(self, *args, **kwargs):
        # Perform evaluation
        result = super().evaluate(*args, **kwargs)
        # Log evaluation Loss
        print_and_log(
            f"Epoch {self.state.epoch}: Eval loss: {result['eval_loss']}")
        return result


def get_trainer(model, training_args, lm_datasets, data_collator):
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["test"],
        data_collator=data_collator
    )

    return trainer

# ------------------------------------------------------------------------------
# Model Training
# ------------------------------------------------------------------------------


def train_model(model_name, model, data_collator, datasets, epochs=25,
                batch_size: int = DEFAULT_BATCH_SIZE,
                learning_rate: float = DEFAULT_LEARNING_RATE,
                weight_decay: float = DEFAULT_WEIGHT_DECAY):

    training_start_time = datetime.now()
    print_and_log("\nPreparing model training with the following parameters:")
    print_and_log(f"Epochs: {epochs}\nBatch size: {batch_size}")
    print_and_log(
        f"Learning rate: {learning_rate}\nWeight decay: {weight_decay}\n")

    print_and_log("\nSetting up training args and trainer.")
    training_args = get_training_args(model_name,
                                      epochs=epochs,
                                      batch_size=batch_size,
                                      learning_rate=learning_rate,
                                      weight_decay=weight_decay)

    trainer = get_trainer(model, training_args, datasets, data_collator)

    try:
        print_and_log("\nRunning model training...\n")
        trainer.train()

        training_end_time = datetime.now()
        logging.info("\nModel training ended.\n")
        print_and_log(
            f"Training took {training_end_time - training_start_time}")

        return trainer

    except Exception as e:
        print_and_log(f"\nError in train_model {model_name}\n{e}\n")
        return trainer

# ------------------------------------------------------------------------------
# Load and Save Model
# ------------------------------------------------------------------------------


def save_model(model_name, trainer, tokenizer, models_dir=DEFAULT_MODELS_DIR):

    train_date = datetime.now().strftime('%Y-%m-%d')
    dir_name = f"{model_name}-trained-{train_date}"
    dir_pathname = os.path.join(models_dir, dir_name)

    trainer.save_model(dir_pathname)
    tokenizer.save_pretrained(dir_pathname)

    model_saved_time = datetime.now()
    print_and_log(
        f"Model & tokenizer saved to directory: {dir_pathname} at {model_saved_time}")


def load_model(model_name, model_save_path=DEFAULT_MODELS_DIR):
    model_path = os.path.join(model_save_path, model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForMaskedLM.from_pretrained(model_path)

    return model, tokenizer


# ------------------------------------------------------------------------------
# Save & Load Training History
# ------------------------------------------------------------------------------

# NB: History is in trainer.state.log_history as a list of dictionaries


def training_history_file(model_name, epochs):
    today = str(date.today())
    return f"training-history-{model_name}-trained-{today}-{epochs}.pkl"


def save_training_history(trainer, model_name, epochs, path='results'):
    history = trainer.state.log_history
    pathname = os.path.join(path, training_history_file(model_name, epochs))
    with open(pathname, 'wb') as f:
        pickle.dump(history, f)
    print_and_log(f"Training history saved to {pathname}")


def load_training_history(model_name, epochs, path='results'):
    pathname = os.path.join(path, training_history_file(model_name, epochs))
    with open(pathname, 'rb') as f:
        history = pickle.load(f)
    return history

# ------------------------------------------------------------------------------


def plot_training_loss(log_history: list[dict]):
    steps = []
    train_loss = []
    eval_loss = []

    for entry in log_history:
        if 'loss' in entry and 'step' in entry:
            steps.append(entry['step'])
            train_loss.append(entry['loss'])
        if 'eval_loss' in entry:
            eval_loss.append(entry['eval_loss'])

    plt.figure(figsize=(10, 6))
    if train_loss:
        plt.plot(steps, train_loss, label="Training Loss", marker='o')
    if eval_loss:
        plt.plot(steps, eval_loss, label="Evaluation Loss", marker='x')

    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training & Evaluation Loss")
    plt.legend()
    plt.grid(True)

    plt.show()

# ------------------------------------------------------------------------------
# CLI Arguments
# ------------------------------------------------------------------------------


parser = argparse.ArgumentParser(description="Training module for MLM")

parser.add_argument('name', metavar='name', type=str,
                    help='The name of the model to train.')

parser.add_argument('--chunks', metavar='chunks', type=int, default=None,
                    help='The number of chunks or all chunks if unspecified.')

parser.add_argument('--epochs', metavar='epochs', type=int, default=20,
                    help='The number of epochs for training. Default is 20.')

parser.add_argument('--rate', metavar='rate', type=float, default=4e-5,
                    help='Training rate for training, default is 4e-5.')

parser.add_argument('--decay', metavar='decay', type=float, default=0.002,
                    help='Weight decay for training, default is 0.002.')

parser.add_argument('--batch-size', metavar='batch_size', type=int, default=16,
                    help='The batch size for training. Default is 16.')


# ------------------------------------------------------------------------------
# Entry function for Model Training
# ------------------------------------------------------------------------------

def run_model_training(model_name: str, tokenizer, model,
                       data_path=DEFAULT_DATA_DIR,
                       epochs: Union[int | None] = 30,
                       batch_size: int = DEFAULT_BATCH_SIZE,
                       learning_rate: float = DEFAULT_LEARNING_RATE,
                       weight_decay: float = DEFAULT_WEIGHT_DECAY,
                       max_chunks: Union[int | None] = None,
                       models_dir=DEFAULT_MODELS_DIR):

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        start_time = datetime.now()
        print_and_log(f"Training script started at {start_time}")
        logging.info(f"Training model {model_name}")

        # Prepare Training data & data collator
        datasets, _, count, problems = prepare_datasets(
            data_path=data_path, max_chunks=max_chunks)
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm_probability=0.15)

        # Train the model with specified parameters
        trainer = train_model(model_name, model, data_collator, datasets,
                              epochs=epochs, batch_size=batch_size,
                              learning_rate=learning_rate, 
                              weight_decay=weight_decay)

        # Save model, tokenizer & training history
        save_model(model_name, trainer, tokenizer, models_dir=models_dir)
        save_training_history(trainer, model_name, epochs, path="results")

        # Return the trainer object
        return trainer, count, problems

    except Exception as e:
        print_and_log(f"\nError in run_training_model {model_name}\n{e}\n")
        return trainer, count, problems 


# ------------------------------------------------------------------------------
# End of File
# ------------------------------------------------------------------------------
