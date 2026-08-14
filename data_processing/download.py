# data_processing/download.py

"""
Download BabyLM data to the artifact directory.

Usage:
python -m data_processing.download
"""

from pathlib import Path
from datasets import load_dataset
from utils import DATA_DIR

data_dir = Path(DATA_DIR)
output_file = data_dir / "babylm.txt"

ds = load_dataset("BabyLM-community/BabyLM-2026-Strict")

print(ds)

def save_split_to_txt(dataset_split, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for example in dataset_split:
            text = example.get("text", "")
            if text:
                f.write(text.strip() + "\n")


save_split_to_txt(ds["train"], output_file)
