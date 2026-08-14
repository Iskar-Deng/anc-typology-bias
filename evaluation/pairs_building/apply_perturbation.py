# evaluation/pairs_building/apply_perturbation.py

"""
Apply one phenomenon's perturbation rule to generated sentences.

Usage:
python -m evaluation.pairs_building.apply_perturbation \
  --phenomenon evaluation/pairs_building/phenomena/1_1_intran_V_form \
  --input artifact/eval_materials/1_1_intran_V_form/generated/selected/00_sov_gn_ac_b_se.jsonl \
  --output artifact/eval_materials/1_1_intran_V_form/pairs/00_sov_gn_ac_b_se.pairs.jsonl \
  --sample-size 100 \
  --seed 42
"""

from argparse import ArgumentParser, Namespace
from collections import Counter
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any
import csv
import json
import random

from utils import ANC_MARK_TABLE, FINITE_MARK_TABLE, VERB_MARK_TABLE


JsonDict = dict[str, Any]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "choices" / "manifest.tsv"


def read_jsonl(path: Path) -> list[JsonDict]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def write_jsonl(rows: list[JsonDict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def infer_language(input_path: Path) -> str:
    return input_path.stem


def load_language_config(language: str) -> JsonDict:
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["language"] != language:
                continue

            config = dict(row)
            config["anc_wo"] = config["anc_choice_word_order"]
            config["anc_wo_choice"] = config["anc_choice_word_order"]
            config["gen_mark"] = "ge"
            config.update(FINITE_MARK_TABLE[config["alignment"]])
            config.update(ANC_MARK_TABLE[(config["strategy"], config["alignment"])])
            config.update(VERB_MARK_TABLE[config["comp_system"]])
            return config

    raise ValueError(f"Language not found in {MANIFEST_PATH}: {language}")


def load_rules(phenomenon_dir: Path):
    rules_path = phenomenon_dir / "rules.py"
    spec = spec_from_file_location("phenomenon_rules", rules_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_id(row: JsonDict, fallback: int) -> int:
    for key in ("source_index", "source_id", "template_index", "template_id", "id"):
        if key in row:
            return int(row[key])
    return fallback


def good_sentence(row: JsonDict) -> str:
    return str(row.get("good", row.get("sent", ""))).strip()


def shuffled_rows(rows: list[JsonDict], seed: int) -> list[JsonDict]:
    output = rows[:]
    random.Random(seed).shuffle(output)
    return output


def apply_rule(
    rows: list[JsonDict],
    rules,
    config: JsonDict,
    language: str,
    sample_size: int,
    seed: int,
) -> tuple[list[JsonDict], Counter]:
    output_rows = []
    stats = Counter(total=len(rows))

    phenomenon_id = getattr(rules, "PHENOMENON_ID", Path(rules.__file__).parent.name)
    phenomenon_name = getattr(rules, "PHENOMENON_NAME", phenomenon_id)

    for index, row in enumerate(shuffled_rows(rows, seed), start=1):
        if len(output_rows) >= sample_size:
            break

        stats["processed"] += 1
        good = good_sentence(row)
        row_source_id = source_id(row, index)
        result = rules.perturb(
            good_sentence=good,
            language_config=config,
            source_index=row_source_id,
            row=row,
        )

        if result.get("skip"):
            stats["skipped"] += 1
            stats[f"skip:{result.get('skip_reason', 'unknown')}"] += 1
            continue

        out_row = {
            **row,
            "phenomenon_id": phenomenon_id,
            "phenomenon_name": phenomenon_name,
            "pair_index": len(output_rows) + 1,
            "language": language,
            "source_id": row_source_id,
            "source_index": row_source_id,
            "good": good,
            "bad": result["bad"].strip(),
        }
        out_row.update({key: value for key, value in result.items() if key != "bad"})
        output_rows.append(out_row)
        stats["written"] += 1

    return output_rows, stats


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--phenomenon", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    language = infer_language(input_path)
    if args.sample_size < 1:
        raise ValueError(f"--sample-size must be >= 1, got {args.sample_size}")

    rows = read_jsonl(input_path)
    rules = load_rules(Path(args.phenomenon))
    config = load_language_config(language)
    output_rows, stats = apply_rule(rows, rules, config, language, args.sample_size, args.seed)

    write_jsonl(output_rows, output_path)

    print(f"total input rows: {stats['total']}")
    print(f"processed rows: {stats['processed']}")
    print(f"written pairs: {stats['written']}")
    print(f"skipped rows: {stats['skipped']}")

    for key in sorted(k for k in stats if k.startswith("skip:")):
        print(f"{key}: {stats[key]}")


if __name__ == "__main__":
    main()
