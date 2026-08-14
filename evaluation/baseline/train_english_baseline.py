#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# evaluation/baseline/train_english_baseline.py

"""
Train one English baseline language model.

Usage:
python -m evaluation.baseline.train_english_baseline \
  --seed 42 \
  --model-size small
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
import sys

from training.train_lm import MODEL_SIZE_TO_KEY, main as train_lm_main
from utils import DATA_DIR, MODELS_DIR, TRAINING_CONFIG


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-size", choices=sorted(MODEL_SIZE_TO_KEY), required=True)
    parser.add_argument("--max-steps", type=int, default=450000)
    parser.add_argument("--resume-from-checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_key = MODEL_SIZE_TO_KEY[args.model_size]
    seed_name = f"seed_{args.seed}"

    TRAINING_CONFIG["eval_steps"] = 50000
    TRAINING_CONFIG["save_steps"] = 50000
    TRAINING_CONFIG["save_total_limit"] = 20

    sys.argv = [
        "training.train_lm",
        "--train-input",
        str(Path(DATA_DIR) / "english_baseline" / "train.jsonl"),
        "--dev-input",
        str(Path(DATA_DIR) / "english_baseline" / "dev.jsonl"),
        "--output-dir",
        str(Path(MODELS_DIR) / model_key / "english_baseline" / seed_name),
        "--seed",
        str(args.seed),
        "--model-size",
        args.model_size,
        "--max-steps",
        str(args.max_steps),
    ]

    if args.resume_from_checkpoint:
        sys.argv.extend(["--resume-from-checkpoint", args.resume_from_checkpoint])

    train_lm_main()


if __name__ == "__main__":
    main()
