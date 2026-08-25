"""Run an Optuna MLM search over a directory of PDF documents."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from src.hyperparams import (
    DEFAULT_MODEL_NAME,
    DEFAULT_TRIALS,
    load_search_config,
    optimize_hyperparameters,
    save_best_config,
)

# Load environment variables from .env file
load_dotenv(override=True)
logger: logging.Logger = logging.getLogger("hyperparams")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune MLM hyperparameters on PDFs and save the best YAML config."
    )
    parser.add_argument("pdf_directory", type=Path, help="Directory of PDFs (recursive).")
    parser.add_argument("output_config", type=Path, help="Destination YAML config file.")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_NAME,
        help=f"Local model path/name or Hugging Face model ID (default: {DEFAULT_MODEL_NAME}).",
    )
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--config", type=Path, help="Optional base MLM YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    study, best_config = optimize_hyperparameters(
        args.pdf_directory, args.model, trials=args.trials,
        config=load_search_config(args.config),
    )
    save_best_config(best_config, args.output_config)
    logger.info("Best validation loss: %.6f", study.best_value)
    logger.info("Best configuration saved to: %s", args.output_config)


if __name__ == "__main__":
    main()
