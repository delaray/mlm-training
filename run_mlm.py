"""Run the end-to-end masked-language-model training pipeline."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import yaml
from dotenv import load_dotenv

from src.mlm_trainer import (
    prepare_mlm_dataset,
    save_trained_model,
    setup_logging,
    setup_model_for_mlm_training,
    train_mlm_model,
)

# Load environment variables from .env file
load_dotenv(override=True)
logger: logging.Logger = logging.getLogger("hyperparams")


DEFAULT_CONFIG_PATH = Path("configs/mlm_training.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an MLM dataset from PDFs and train an encoder model."
        )
    )
    parser.add_argument(
        "model_name",
        help=(
            "Local model directory, name under paths.models_dir, or a "
            "Hugging Face model ID."
        ),
    )
    parser.add_argument(
        "pdf_directory",
        type=Path,
        help="Directory containing PDF files (searched recursively).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML configuration file (default: {DEFAULT_CONFIG_PATH}).",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")

    required_sections = {"paths", "dataset", "model", "training"}
    missing_sections = required_sections.difference(config)
    if missing_sections:
        missing = ", ".join(sorted(missing_sections))
        raise ValueError(f"Configuration is missing sections: {missing}")

    return cast(dict[str, Any], config)


def resolve_model_source(model_name: str, models_dir: Path) -> str:
    direct_path = Path(model_name).expanduser()
    if direct_path.exists():
        return str(direct_path.resolve())

    models_path = (models_dir / model_name).expanduser()
    if models_path.exists():
        return str(models_path.resolve())

    if direct_path.is_absolute():
        raise FileNotFoundError(f"Local model directory not found: {direct_path}")

    # A non-local value may be a Hugging Face repository ID.
    return model_name


def validate_pdf_directory(pdf_directory: Path) -> Path:
    pdf_directory = pdf_directory.expanduser().resolve()
    if not pdf_directory.is_dir():
        raise NotADirectoryError(f"PDF directory not found: {pdf_directory}")

    has_pdf = any(
        path.is_file() and path.suffix.lower() == ".pdf"
        for path in pdf_directory.rglob("*")
    )
    if not has_pdf:
        raise ValueError(f"No PDF files found under: {pdf_directory}")

    return pdf_directory


def main() -> None:
    args = parse_args()
    config = load_config(args.config.expanduser().resolve())

    paths = cast(dict[str, Any], config["paths"])
    dataset_config = cast(dict[str, Any], config["dataset"])
    model_config = cast(dict[str, Any], config["model"])
    training_config = cast(dict[str, Any], config["training"])

    models_dir = Path(str(paths["models_dir"]))
    results_dir = Path(str(paths["results_dir"]))
    logs_dir = Path(str(paths["logs_dir"]))
    pdf_directory = validate_pdf_directory(args.pdf_directory)
    model_source = resolve_model_source(args.model_name, models_dir)

    model_short_name = args.model_name.rstrip("/").split("/")[-1]
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # noqa: DTZ005
    run_name = f"{model_short_name}-mlm-{run_timestamp}"
    output_dir = results_dir / run_name
    save_dir = models_dir / run_name
    log_file = logs_dir / f"{run_name}.log"

    output_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(str(log_file), level=logger.INFO)

    logger.info("MLM run: %s", run_name)
    logger.info("Model source: %s", model_source)
    logger.info("PDF directory: %s", pdf_directory)

    datasets, tokenizer = prepare_mlm_dataset(
        data_dir=str(pdf_directory),
        model_name=model_source,
        max_length=int(dataset_config["max_length"]),
        chunk_size=int(dataset_config["chunk_size"]),
        chunk_overlap=int(dataset_config["chunk_overlap"]),
        test_split=float(dataset_config["test_split"]),
        max_chunks=dataset_config.get("max_chunks"),
        use_fast_tokenizer=bool(dataset_config["use_fast_tokenizer"]),
    )

    model, is_quantized = setup_model_for_mlm_training(
        model_name=model_source,
        device=str(model_config["device"]),
        use_qlora=bool(model_config["use_qlora"]),
        use_lora=bool(model_config["use_lora"]),
        lora_r=int(model_config["lora_r"]),
        lora_alpha=int(model_config["lora_alpha"]),
        lora_dropout=float(model_config["lora_dropout"]),
        target_modules=model_config.get("target_modules"),
        load_in_4bit=bool(model_config["load_in_4bit"]),
        load_in_8bit=bool(model_config["load_in_8bit"]),
    )

    trainer = train_mlm_model(
        model=model,
        tokenizer=tokenizer,
        datasets=datasets,
        output_dir=str(output_dir),
        epochs=int(training_config["epochs"]),
        batch_size=int(training_config["batch_size"]),
        learning_rate=float(training_config["learning_rate"]),
        mlm_probability=float(training_config["mlm_probability"]),
        weight_decay=float(training_config["weight_decay"]),
        warmup_steps=int(training_config["warmup_steps"]),
        logging_steps=int(training_config["logging_steps"]),
        save_steps=int(training_config["save_steps"]),
        eval_steps=int(training_config["eval_steps"]),
        gradient_accumulation_steps=int(
            training_config["gradient_accumulation_steps"]
        ),
        fp16=bool(training_config["fp16"]),
        save_total_limit=int(training_config["save_total_limit"]),
        device=str(model_config["device"]),
    )

    save_trained_model(
        model=model,
        tokenizer=tokenizer,
        save_path=str(save_dir),
        is_peft_model=bool(model_config["use_lora"]),
    )
    logger.info("Training artifacts saved to: %s", save_dir)


if __name__ == "__main__":
    main()
