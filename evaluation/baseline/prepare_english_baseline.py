#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# evaluation/baseline/prepare_english_baseline.py

"""
Convert English train/dev text files to JSONL for baseline training.

Usage:
python -m evaluation.baseline.prepare_english_baseline
"""

from pathlib import Path
import json

from utils import DATA_DIR


def read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def write_jsonl(path: Path, sentences: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sentence in sentences:
            f.write(json.dumps({"sent": sentence}, ensure_ascii=False) + "\n")


def main() -> None:
    data_dir = Path(DATA_DIR)
    output_dir = data_dir / "english_baseline"

    train_sentences = read_lines(data_dir / "train.txt")
    dev_sentences = read_lines(data_dir / "dev.txt")

    write_jsonl(output_dir / "train.jsonl", train_sentences)
    write_jsonl(output_dir / "dev.jsonl", dev_sentences)

    stats = {
        "train_input": str(data_dir / "train.txt"),
        "dev_input": str(data_dir / "dev.txt"),
        "train_output": str(output_dir / "train.jsonl"),
        "dev_output": str(output_dir / "dev.jsonl"),
        "train_sentences": len(train_sentences),
        "dev_sentences": len(dev_sentences),
    }

    with (output_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
