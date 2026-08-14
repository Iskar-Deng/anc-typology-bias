# language_generation/generate_from_mrs_bank.py

"""
Generate target-language candidates from an MRS bank.

Usage:
python -m language_generation.generate_from_mrs_bank \
  --grammar GRAMMAR_DAT \
  --ace-bin ACE_BIN \
  --input MRS_JSONL \
  --output RAW_JSONL
"""

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any

from delphin import ace
from tqdm import tqdm


WORKER_GRAMMAR = ""
WORKER_ACE_BIN = ""
WORKER_MAX_GEN = 20


def iter_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def count_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def generate_surfaces(grammar: str, ace_bin: str, mrs: str, max_gen: int) -> list[str]:
    response = ace.generate(
        grammar,
        mrs,
        executable=ace_bin,
        cmdargs=["-n", str(max_gen)],
    )

    surfaces = []
    seen = set()

    for result in response.get("results", []):
        surface = result.get("surface")
        if isinstance(surface, str) and surface and surface not in seen:
            surfaces.append(surface)
            seen.add(surface)

    return surfaces


def init_worker(grammar: str, ace_bin: str, max_gen: int) -> None:
    global WORKER_GRAMMAR
    global WORKER_ACE_BIN
    global WORKER_MAX_GEN

    WORKER_GRAMMAR = grammar
    WORKER_ACE_BIN = ace_bin
    WORKER_MAX_GEN = max_gen


def process_row(row: dict[str, Any]) -> dict[str, Any] | None:
    mrs = row.get("mrs")
    if row.get("id") is None or not isinstance(mrs, str) or not mrs.strip():
        return None

    try:
        surfaces = generate_surfaces(WORKER_GRAMMAR, WORKER_ACE_BIN, mrs, WORKER_MAX_GEN)
    except (OSError, RuntimeError, ValueError):
        return None

    output = {
        "id": row["id"],
        "sent": surfaces,
    }

    for key in ("source_id", "pseudo_index", "sentence", "pseudo_english"):
        if key in row:
            output[key] = row[key]

    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grammar", required=True)
    parser.add_argument("--ace-bin", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunksize", type=int, default=50)
    parser.add_argument("--max-gen", type=int, default=20)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    skipped = 0
    overgenerated = 0
    saved = 0

    with output_path.open("w", encoding="utf-8") as f:
        with mp.get_context("spawn").Pool(
            processes=args.workers,
            initializer=init_worker,
            initargs=(args.grammar, args.ace_bin, args.max_gen),
        ) as pool:
            rows = pool.imap(process_row, iter_rows(input_path), chunksize=args.chunksize)

            for row in tqdm(rows, total=count_rows(input_path), desc="Generating"):
                if row is None:
                    skipped += 1
                    continue

                if len(row["sent"]) > 1:
                    overgenerated += 1

                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                saved += 1

    total = saved + skipped
    print(f"total input records: {total}")
    print(f"saved records: {saved}")
    print(f"skipped records: {skipped}")
    print(f"overgenerated records: {overgenerated}")


if __name__ == "__main__":
    main()
