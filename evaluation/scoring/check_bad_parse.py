#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# evaluation/scoring/check_bad_parse.py

"""
Check whether BAD sentences in one minimal-pair file can still be parsed.

Usage:
python -m evaluation.scoring.check_bad_parse \
  --pairs artifact/eval_materials/1_1_intran_V_form/pairs/00_sov_gn_ac_b_se.pairs.jsonl \
  --grammar artifact/grammars/00_sov_gn_ac_b_se/00_sov_gn_ac_b_se.dat
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any
import json
import re
import subprocess

from utils import RESULTS_DIR, TOOLS_DIR


JsonDict = dict[str, Any]
NOTE_READINGS_RE = re.compile(r"^NOTE:\s+(\d+)\s+readings,")
BATCH_SIZE = 200
TIMEOUT_SECONDS = 900


def read_jsonl(path: Path) -> list[JsonDict]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def clean_tsv(value: Any) -> str:
    return str(value).replace("\t", " ").replace("\n", " ").replace("\r", " ")


def ace_path() -> Path:
    return Path(TOOLS_DIR) / "ace" / "ace"


def language_from_pairs(pairs: Path) -> str:
    name = pairs.name
    if name.endswith(".pairs.jsonl"):
        return name.removesuffix(".pairs.jsonl")

    return pairs.stem


def phenomenon_from_pairs(pairs: Path) -> str:
    if pairs.parent.name == "pairs":
        return pairs.parent.parent.name

    return pairs.parent.name


def output_path(pairs: Path) -> Path:
    phenomenon = phenomenon_from_pairs(pairs)
    language = language_from_pairs(pairs)
    return Path(RESULTS_DIR) / "bad_parse" / phenomenon / f"{language}.bad_parse.tsv"


def parse_batch(ace: Path, grammar: Path, sentences: list[str]) -> list[int]:
    proc = subprocess.run(
        [str(ace), "-g", str(grammar), "-n", "1", "-R"],
        input="\n".join(sentences) + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=TIMEOUT_SECONDS,
    )

    readings = []
    for line in proc.stdout.splitlines():
        match = NOTE_READINGS_RE.match(line)
        if match:
            readings.append(int(match.group(1)))

    if len(readings) == len(sentences):
        return readings

    if len(sentences) == 1:
        return [0]

    output = []
    for sentence in sentences:
        output.extend(parse_batch(ace, grammar, [sentence]))
    return output


def parse_sentences(ace: Path, grammar: Path, sentences: list[str]) -> list[int]:
    output = []
    for start in range(0, len(sentences), BATCH_SIZE):
        output.extend(parse_batch(ace, grammar, sentences[start:start + BATCH_SIZE]))
    return output


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--grammar", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pairs = Path(args.pairs)
    grammar = Path(args.grammar)
    ace = ace_path()
    output = output_path(pairs)
    summary = output.with_suffix(".summary.json")
    phenomenon = phenomenon_from_pairs(pairs)
    language = language_from_pairs(pairs)

    rows = read_jsonl(pairs)
    bad_sentences = []
    for row in rows:
        bad = row.get("bad")
        if not isinstance(bad, str) or not bad.strip():
            raise ValueError(f"Missing bad sentence: {row}")
        bad_sentences.append(bad.strip())

    readings = parse_sentences(ace, grammar, bad_sentences)

    output.parent.mkdir(parents=True, exist_ok=True)
    bad_parse_count = 0

    with output.open("w", encoding="utf-8") as f:
        f.write("pair_index\tid\tbad_parse_readings\tbad_parse\tgood\tbad\n")
        for index, (row, n_readings) in enumerate(zip(rows, readings), start=1):
            bad_parse = int(n_readings > 0)
            bad_parse_count += bad_parse
            f.write(
                f"{row.get('pair_index', index)}\t"
                f"{row.get('id', '')}\t"
                f"{n_readings}\t"
                f"{bad_parse}\t"
                f"{clean_tsv(row.get('good', ''))}\t"
                f"{clean_tsv(row.get('bad', ''))}\n"
            )

    summary_data = {
        "phenomenon": phenomenon,
        "language": language,
        "pairs": str(pairs),
        "grammar": str(grammar),
        "output": str(output),
        "n_pairs": len(rows),
        "bad_parse": bad_parse_count,
        "bad_parse_rate": bad_parse_count / len(rows),
    }

    with summary.open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary_data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
