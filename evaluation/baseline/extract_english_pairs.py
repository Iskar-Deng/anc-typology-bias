#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# evaluation/baseline/extract_english_pairs.py

"""
Extract English baseline minimal pairs for valency evaluation phenomena.

Usage:
python -m evaluation.baseline.extract_english_pairs \
  --phenomenon 1_9_intran_V_valency
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any
import csv
import json

from utils import EVAL_MATERIALS_DIR


JsonDict = dict[str, Any]
REF_LANGUAGE = "00_sov_gn_ac_b_se"
VALENCY_PHENOMENA = {
    "1_9_intran_V_valency",
    "1_10_tran_V_valency",
    "3_6_clausal_CV_valency",
}


def read_jsonl(path: Path) -> list[JsonDict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def pair_key(row: JsonDict) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("source_id", row.get("source_index", row.get("id")))),
        str(row.get("source_uid", row.get("blimp_source_uid"))),
        str(row.get("source_pair_id", row.get("blimp_pair_id"))),
        str(row.get("good_stem")),
        str(row.get("bad_stem")),
    )


def cv_key(row: JsonDict) -> str:
    return str(row.get("source_id", row.get("source_index", row.get("id"))))


def build_valency_rows(phenomenon: str, ref_rows: list[JsonDict], tsv_rows: list[dict[str, str]]) -> list[JsonDict]:
    source_rows = {
        (row["source_id"], row["source_uid"], row["pair_id"], row["good_stem"], row["bad_stem"]): row
        for row in tsv_rows
    }

    output = []
    for index, ref_row in enumerate(ref_rows, start=1):
        row = source_rows[pair_key(ref_row)]
        output.append(
            {
                "pair_index": index,
                "phenomenon": phenomenon,
                "source_id": row["source_id"],
                "source_uid": row["source_uid"],
                "pair_id": row["pair_id"],
                "good_stem": row["good_stem"],
                "bad_stem": row["bad_stem"],
                "good": row["good_source"],
                "bad": row["bad_source"],
            }
        )
    return output


def build_cv_rows(phenomenon: str, ref_rows: list[JsonDict], tsv_rows: list[dict[str, str]]) -> list[JsonDict]:
    source_rows = {row["source_id"]: row for row in tsv_rows}

    output = []
    for index, ref_row in enumerate(ref_rows, start=1):
        row = source_rows[cv_key(ref_row)]
        output.append(
            {
                "pair_index": index,
                "phenomenon": phenomenon,
                "source_id": row["source_id"],
                "source_uid": row["source_uid"],
                "pair_id": row["pair_id"],
                "good_stem": row.get("good_stem", ""),
                "bad_stem": row.get("bad_stem", ""),
                "embedded_type": row.get("embedded_type", ""),
                "cv_thatS_mean": row.get("cv_thatS_mean", ""),
                "tv_NPVNP_mean": row.get("tv_NPVNP_mean", ""),
                "tv_thatS_mean": row.get("tv_thatS_mean", ""),
                "good": row["good_source"],
                "bad": row["bad_source"],
            }
        )
    return output


def write_jsonl(path: Path, rows: list[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--phenomenon", choices=sorted(VALENCY_PHENOMENA), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    phenomenon_dir = Path(EVAL_MATERIALS_DIR) / args.phenomenon
    source_tsv = Path("evaluation") / "pairs_building" / "phenomena" / args.phenomenon / "valency_pairs.tsv"
    ref_pairs = phenomenon_dir / "pairs" / f"{REF_LANGUAGE}.pairs.jsonl"
    output = phenomenon_dir / "english_pairs" / "pairs.jsonl"

    ref_rows = read_jsonl(ref_pairs)
    tsv_rows = read_tsv(source_tsv)
    if args.phenomenon == "3_6_clausal_CV_valency":
        rows = build_cv_rows(args.phenomenon, ref_rows, tsv_rows)
    else:
        rows = build_valency_rows(args.phenomenon, ref_rows, tsv_rows)

    write_jsonl(output, rows)
    print(json.dumps({"output": str(output), "written": len(rows)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
