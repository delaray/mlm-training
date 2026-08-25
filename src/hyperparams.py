"""Optuna hyperparameter search for masked-language-model training."""

from __future__ import annotations

import gc
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, cast

import optuna
import torch
import yaml
from datasets import DatasetDict
from transformers import PreTrainedTokenizerBase

from src.mlm_trainer import (
    prepare_mlm_dataset,
    setup_model_for_mlm_training,
    train_mlm_model,
)

DEFAULT_TRIALS = 12
DEFAULT_EPOCHS = 3
DEFAULT_MODEL_NAME = "google/electra-small-discriminator"

DEFAULT_CONFIG: dict[str, Any] = {
    "paths": {"models_dir": "models", "results_dir": "results", "logs_dir": "logs"},
    "dataset": {
        "max_length": 512, "chunk_size": 2048, "chunk_overlap": 200,
        "test_split": 0.1, "max_chunks": None, "use_fast_tokenizer": True,
    },
    "model": {
        "device": "auto", "use_lora": True, "use_qlora": True,
        "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.1,
        "target_modules": None, "load_in_4bit": True, "load_in_8bit": False,
    },
    "training": {
        "epochs": DEFAULT_EPOCHS, "batch_size": 8, "learning_rate": 2e-4,
        "mlm_probability": 0.15, "weight_decay": 0.01, "warmup_steps": 500,
        "logging_steps": 100, "save_steps": 1000, "eval_steps": 500,
        "gradient_accumulation_steps": 4, "fp16": True, "save_total_limit": 1,
    },
}


def load_search_config(config_path: Path | None = None) -> dict[str, Any]:
    """Return defaults merged with an optional MLM YAML configuration."""
    config = deepcopy(DEFAULT_CONFIG)
    if config_path is None:
        return config
    with config_path.open(encoding="utf-8") as config_file:
        overrides = yaml.safe_load(config_file)
    if not isinstance(overrides, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")
    for section, values in overrides.items():
        if section in config and isinstance(config[section], dict):
            if not isinstance(values, dict):
                raise ValueError(f"Configuration section '{section}' must be a mapping")
            config[section].update(values)
        else:
            config[section] = values
    return config


def resolve_model_source(model_name: str, models_dir: Path) -> str:
    """Resolve a local model name/path, falling back to a Hub model ID."""
    direct_path = Path(model_name).expanduser()
    if direct_path.exists():
        return str(direct_path.resolve())
    models_path = (models_dir / model_name).expanduser()
    if models_path.exists():
        return str(models_path.resolve())
    if direct_path.is_absolute():
        raise FileNotFoundError(f"Local model directory not found: {direct_path}")
    return model_name


def prepare_search_dataset(
    pdf_directory: Path,
    model_source: str,
    dataset_config: Mapping[str, Any],
) -> tuple[DatasetDict, PreTrainedTokenizerBase]:
    """Read and tokenize PDFs once for reuse by every Optuna trial."""
    return prepare_mlm_dataset(
        data_dir=str(pdf_directory), model_name=model_source,
        max_length=int(dataset_config["max_length"]),
        chunk_size=int(dataset_config["chunk_size"]),
        chunk_overlap=int(dataset_config["chunk_overlap"]),
        test_split=float(dataset_config["test_split"]),
        max_chunks=dataset_config.get("max_chunks"),
        use_fast_tokenizer=bool(dataset_config["use_fast_tokenizer"]),
    )


def create_objective(
    *, datasets: DatasetDict, tokenizer: PreTrainedTokenizerBase,
    model_source: str, model_config: Mapping[str, Any],
    training_config: Mapping[str, Any], results_dir: Path,
):
    """Build an Optuna objective that minimizes validation loss."""
    def objective(trial: optuna.Trial) -> float:
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 2e-4, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-3, 0.1, log=True)
        model, _ = setup_model_for_mlm_training(
            model_name=model_source, device=str(model_config["device"]),
            use_qlora=bool(model_config["use_qlora"]),
            use_lora=bool(model_config["use_lora"]),
            lora_r=int(model_config["lora_r"]),
            lora_alpha=int(model_config["lora_alpha"]),
            lora_dropout=float(model_config["lora_dropout"]),
            target_modules=model_config.get("target_modules"),
            load_in_4bit=bool(model_config["load_in_4bit"]),
            load_in_8bit=bool(model_config["load_in_8bit"]),
        )
        trainer = None
        try:
            trainer = train_mlm_model(
                model=model, tokenizer=tokenizer, datasets=datasets,
                output_dir=str(results_dir / f"trial-{trial.number}"),
                epochs=int(training_config["epochs"]),
                batch_size=int(training_config["batch_size"]),
                learning_rate=learning_rate,
                mlm_probability=float(training_config["mlm_probability"]),
                weight_decay=weight_decay,
                warmup_steps=int(training_config["warmup_steps"]),
                logging_steps=int(training_config["logging_steps"]),
                save_steps=int(training_config["save_steps"]),
                eval_steps=int(training_config["eval_steps"]),
                gradient_accumulation_steps=int(training_config["gradient_accumulation_steps"]),
                fp16=bool(training_config["fp16"]),
                save_total_limit=int(training_config["save_total_limit"]),
                device=str(model_config["device"]),
            )
            metrics = trainer.evaluate()
            if "eval_loss" not in metrics:
                raise RuntimeError("Trainer evaluation did not return eval_loss")
            return float(metrics["eval_loss"])
        finally:
            del trainer
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return objective


def optimize_hyperparameters(
    pdf_directory: Path, model_name: str = DEFAULT_MODEL_NAME, *,
    trials: int = DEFAULT_TRIALS, config: Mapping[str, Any] | None = None,
) -> tuple[optuna.Study, dict[str, Any]]:
    """Run the search and return both the study and best runnable config."""
    if trials < 1:
        raise ValueError("trials must be at least 1")
    pdf_directory = pdf_directory.expanduser().resolve()
    if not pdf_directory.is_dir():
        raise NotADirectoryError(f"PDF directory not found: {pdf_directory}")
    if not any(path.is_file() and path.suffix.lower() == ".pdf" for path in pdf_directory.rglob("*")):
        raise ValueError(f"No PDF files found under: {pdf_directory}")

    resolved_config = deepcopy(dict(config)) if config is not None else load_search_config()
    paths = cast(dict[str, Any], resolved_config["paths"])
    dataset_config = cast(dict[str, Any], resolved_config["dataset"])
    model_config = cast(dict[str, Any], resolved_config["model"])
    training_config = cast(dict[str, Any], resolved_config["training"])
    model_source = resolve_model_source(model_name, Path(str(paths["models_dir"])))
    results_dir = Path(str(paths["results_dir"])) / "optuna"
    results_dir.mkdir(parents=True, exist_ok=True)
    datasets, tokenizer = prepare_search_dataset(pdf_directory, model_source, dataset_config)

    study = optuna.create_study(direction="minimize", study_name="mlm-hyperparameters")
    study.optimize(create_objective(
        datasets=datasets, tokenizer=tokenizer, model_source=model_source,
        model_config=model_config, training_config=training_config,
        results_dir=results_dir,
    ), n_trials=trials)

    resolved_config["training"].update(study.best_params)
    resolved_config["optuna"] = {
        "best_validation_loss": float(study.best_value),
        "best_trial": study.best_trial.number,
        "completed_trials": len(study.trials),
        "model_name": model_name,
        "pdf_directory": str(pdf_directory),
    }
    return study, resolved_config


def save_best_config(config: Mapping[str, Any], output_path: Path) -> None:
    """Write the winning configuration as portable, safe YAML."""
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(dict(config), config_file, sort_keys=False)
