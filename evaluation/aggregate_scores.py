#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# evaluation/aggregate_scores.py

"""
Aggregate per-language score summaries into seed-level and three-seed tables.

Usage:
python -m evaluation.aggregate_scores \
  --tau 5
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
import csv
import json
import math
import statistics

from utils import RESULTS_DIR


MODEL_SIZE = "gpt2-small"
SEEDS = ["seed_42", "seed_43", "seed_44"]
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


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--tau", type=float, default=5.0)
    return parser.parse_args()


def softplus(x: float) -> float:
    if x > 0:
        return x + math.log1p(math.exp(-x))

    return math.log1p(math.exp(x))


def learning_score(accuracy_pct: float, tau: float) -> float:
    low = softplus(-50.0 / tau)
    high = softplus(50.0 / tau)
    return (softplus((accuracy_pct - 50.0) / tau) - low) / (high - low)


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def read_summary(score_root: Path, seed: str, phenomenon_id: str, language_id: str) -> dict:
    summary_path = score_root / seed / phenomenon_id / f"{language_id}.scores.summary.json"
    return read_json(summary_path)


def build_seed_rows(
    score_root: Path,
    seed: str,
    phenomena: list[dict[str, str]],
    languages: list[dict[str, str]],
) -> list[dict]:
    rows = []

    for phenomenon in phenomena:
        phenomenon_id = phenomenon["phenomenon"]
        for language in languages:
            language_id = language["language"]
            summary = read_summary(score_root, seed, phenomenon_id, language_id)
            rows.append(
                {
                    "seed": seed,
                    **phenomenon,
                    **language,
                    "n_pairs": summary.get("n_pairs"),
                    "accuracy": summary.get("accuracy"),
                    "ties": summary.get("ties"),
                    "tie_rate": summary.get("tie_rate"),
                    "mean_delta_good_minus_bad": summary.get("mean_delta_good_minus_bad"),
                    "score_mode": summary.get("score_mode"),
                }
            )

    return rows


def build_disagreement_rows(
    score_root: Path,
    tau: float,
    phenomena: list[dict[str, str]],
    languages: list[dict[str, str]],
) -> list[dict]:
    rows = []

    for phenomenon in phenomena:
        phenomenon_id = phenomenon["phenomenon"]
        for language in languages:
            language_id = language["language"]
            summaries = {
                seed: read_summary(score_root, seed, phenomenon_id, language_id)
                for seed in SEEDS
            }
            accuracies = {
                seed: float(summary["accuracy"]) * 100.0
                for seed, summary in summaries.items()
            }
            scores = {
                seed: learning_score(accuracy, tau)
                for seed, accuracy in accuracies.items()
            }
            disagreement = max(scores.values()) - min(scores.values())

            rows.append(
                {
                    **phenomenon,
                    **language,
                    "tau": tau,
                    "accuracy_pct_seed_42": accuracies["seed_42"],
                    "accuracy_pct_seed_43": accuracies["seed_43"],
                    "accuracy_pct_seed_44": accuracies["seed_44"],
                    "learning_score_seed_42": scores["seed_42"],
                    "learning_score_seed_43": scores["seed_43"],
                    "learning_score_seed_44": scores["seed_44"],
                    "seed_disagreement": disagreement,
                }
            )

    return rows


def build_mean_rows(
    score_root: Path,
    phenomena: list[dict[str, str]],
    languages: list[dict[str, str]],
) -> list[dict]:
    rows = []

    for phenomenon in phenomena:
        phenomenon_id = phenomenon["phenomenon"]
        for language in languages:
            language_id = language["language"]
            summaries = [
                read_summary(score_root, seed, phenomenon_id, language_id)
                for seed in SEEDS
            ]
            accuracies = [float(summary["accuracy"]) for summary in summaries]
            deltas = [float(summary["mean_delta_good_minus_bad"]) for summary in summaries]
            tie_rates = [float(summary["tie_rate"]) for summary in summaries]
            ties = [float(summary["ties"]) for summary in summaries]
            n_pairs = int(summaries[0]["n_pairs"])
            score_modes = sorted({str(summary.get("score_mode", "")) for summary in summaries})

            rows.append(
                {
                    **phenomenon,
                    **language,
                    "n_pairs": n_pairs,
                    "accuracy": statistics.mean(accuracies),
                    "accuracy_sd": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
                    "accuracy_pct": statistics.mean(accuracies) * 100.0,
                    "accuracy_pct_sd": (
                        statistics.stdev([accuracy * 100.0 for accuracy in accuracies])
                        if len(accuracies) > 1
                        else 0.0
                    ),
                    "accuracy_min": min(accuracies),
                    "accuracy_max": max(accuracies),
                    "ties_mean": statistics.mean(ties),
                    "tie_rate": statistics.mean(tie_rates),
                    "mean_delta_good_minus_bad": statistics.mean(deltas),
                    "mean_delta_good_minus_bad_sd": (
                        statistics.stdev(deltas) if len(deltas) > 1 else 0.0
                    ),
                    "score_mode": ",".join(score_modes),
                    "seeds": ",".join(SEEDS),
                }
            )

    return rows


def main() -> None:
    args = parse_args()
    phenomena = read_tsv(Path("evaluation") / "phenomena_manifest.tsv")
    languages = read_tsv(Path("choices") / "manifest.tsv")
    score_root = Path(RESULTS_DIR) / "scoring" / MODEL_SIZE

    seed_fields = [
        "seed",
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
        "accuracy",
        "ties",
        "tie_rate",
        "mean_delta_good_minus_bad",
        "score_mode",
    ]

    mean_fields = [
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
        "accuracy",
        "accuracy_sd",
        "accuracy_pct",
        "accuracy_pct_sd",
        "accuracy_min",
        "accuracy_max",
        "ties_mean",
        "tie_rate",
        "mean_delta_good_minus_bad",
        "mean_delta_good_minus_bad_sd",
        "score_mode",
        "seeds",
    ]

    for seed in SEEDS:
        rows = build_seed_rows(score_root, seed, phenomena, languages)
        output_path = OUTPUT_DIR / f"{seed}_scores.tsv"
        write_tsv(output_path, rows, seed_fields)
        print(f"{output_path}\t{len(rows)}")

    mean_rows = build_mean_rows(score_root, phenomena, languages)
    mean_path = OUTPUT_DIR / "mean_scores.tsv"
    write_tsv(mean_path, mean_rows, mean_fields)
    print(f"{mean_path}\t{len(mean_rows)}")

    disagreement_fields = [
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
        "tau",
        "accuracy_pct_seed_42",
        "accuracy_pct_seed_43",
        "accuracy_pct_seed_44",
        "learning_score_seed_42",
        "learning_score_seed_43",
        "learning_score_seed_44",
        "seed_disagreement",
    ]
    disagreement_rows = build_disagreement_rows(score_root, args.tau, phenomena, languages)
    disagreement_path = OUTPUT_DIR / "seed_disagreement.tsv"
    write_tsv(disagreement_path, disagreement_rows, disagreement_fields)

    disagreements = [float(row["seed_disagreement"]) for row in disagreement_rows]
    print(f"{disagreement_path}\t{len(disagreement_rows)}")
    print(f"D_median\t{statistics.median(disagreements)}")
    print(f"D_90\t{quantile(disagreements, 0.90)}")


if __name__ == "__main__":
    main()
