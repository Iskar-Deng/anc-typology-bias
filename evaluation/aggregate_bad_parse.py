#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# evaluation/aggregate_bad_parse.py

"""
Aggregate per-language BAD parse summaries into one table.

Usage:
python -m evaluation.aggregate_bad_parse
"""

from pathlib import Path
import csv
import json

from utils import RESULTS_DIR


OUTPUT_DIR = Path("results")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_tsv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> None:
    phenomena = read_tsv(Path("evaluation") / "phenomena_manifest.tsv")
    languages = read_tsv(Path("choices") / "manifest.tsv")
    bad_parse_root = Path(RESULTS_DIR) / "bad_parse"

    fieldnames = [
        "phenomenon",
        "figure_stem",
        "label",
        "title",
        "language",
        "clause_wo",
        "np_wo",
        "alignment",
        "comp_system",
        "strategy",
        "anc_choice_word_order",
        "anc_iv_order",
        "anc_tv_order",
        "n_pairs",
        "bad_parse",
        "bad_parse_rate",
    ]

    rows = []
    for phenomenon in phenomena:
        phenomenon_id = phenomenon["phenomenon"]
        for language in languages:
            language_id = language["language"]
            summary_path = bad_parse_root / phenomenon_id / f"{language_id}.bad_parse.summary.json"
            summary = read_json(summary_path)
            rows.append(
                {
                    **phenomenon,
                    **language,
                    "n_pairs": summary.get("n_pairs"),
                    "bad_parse": summary.get("bad_parse"),
                    "bad_parse_rate": summary.get("bad_parse_rate"),
                }
            )

    output_path = OUTPUT_DIR / "bad_parse.tsv"
    write_tsv(output_path, rows, fieldnames)
    print(f"{output_path}\t{len(rows)}")


if __name__ == "__main__":
    main()
