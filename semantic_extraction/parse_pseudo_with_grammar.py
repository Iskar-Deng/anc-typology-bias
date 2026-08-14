# semantic_extraction/parse_pseudo_with_grammar.py

"""
Parse pseudo-English sentences with an ACE grammar.

Usage:
python -m semantic_extraction.parse_pseudo_with_grammar \
  --ace-bin ACE_BIN \
  --grammar GRAMMAR_DAT \
  --input PSEUDO_JSONL \
  --output MRS_JSONL
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
from pathlib import Path
from typing import Any, Iterator

from delphin import ace
from tqdm import tqdm

from utils import MRS_REWRITE_RULES


JsonDict = dict[str, Any]

_WORKER_GRAMMAR: str | None = None
_WORKER_ACE_BIN: str | None = None
_WORKER_MAX_PARSES = 20
_WORKER_RESTART_EVERY = 5000
_WORKER_FIRST_PARSE_ONLY = False
_WORKER_SKIP_FAILED = False
_WORKER_PARSER: Any = None
_WORKER_PARSED_SINCE_RESTART = 0


def iter_pseudo_jsonl(path: Path) -> Iterator[JsonDict]:
    with path.open("r", encoding="utf-8") as f:
        for line_num, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue

            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_num} in {path} is not a JSON object")

            pseudo = row.get("pseudo_english")
            if isinstance(pseudo, str) and pseudo.strip():
                yield row


def count_valid_rows(path: Path) -> int:
    return sum(1 for _ in iter_pseudo_jsonl(path))


def normalize_mrs(mrs: str) -> str:
    for source, target in MRS_REWRITE_RULES:
        mrs = mrs.replace(source, target)

    return re.sub(r"ICONS:\s*<[^>]*>", "ICONS: < >", mrs, flags=re.DOTALL)


def sentence_id(row: JsonDict, fallback: int) -> int:
    value = row.get("id")
    if isinstance(value, int):
        return value
    return fallback


def source_sentence(row: JsonDict) -> str | None:
    value = row.get("sentence")
    if isinstance(value, str):
        return value
    return None


def pseudo_sentence(row: JsonDict) -> str:
    value = row.get("pseudo_english")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Missing pseudo_english")
    return value.strip()


def metadata(row: JsonDict) -> JsonDict:
    return {key: row[key] for key in ("source_id", "pseudo_index") if key in row}


def make_parser(grammar: str, ace_bin: str, max_parses: int) -> Any:
    cmdargs = ["-n", str(max_parses)]
    return ace.ACEParser(grammar, executable=ace_bin, cmdargs=cmdargs)


def close_parser(parser: Any) -> None:
    close = getattr(parser, "close", None)
    if callable(close):
        close()


def parse_with_oneoff(grammar: str, ace_bin: str, sentence: str, max_parses: int) -> list[JsonDict]:
    cmdargs = ["-n", str(max_parses)]
    response = ace.parse(grammar, sentence, executable=ace_bin, cmdargs=cmdargs)

    results = response.get("results", [])
    if isinstance(results, list):
        return results
    return []


def init_worker(
    grammar: str,
    ace_bin: str,
    max_parses: int,
    restart_every: int,
    first_parse_only: bool,
    skip_failed: bool,
) -> None:
    global _WORKER_GRAMMAR
    global _WORKER_ACE_BIN
    global _WORKER_MAX_PARSES
    global _WORKER_RESTART_EVERY
    global _WORKER_FIRST_PARSE_ONLY
    global _WORKER_SKIP_FAILED
    global _WORKER_PARSER
    global _WORKER_PARSED_SINCE_RESTART

    _WORKER_GRAMMAR = grammar
    _WORKER_ACE_BIN = ace_bin
    _WORKER_MAX_PARSES = max_parses
    _WORKER_RESTART_EVERY = restart_every
    _WORKER_FIRST_PARSE_ONLY = first_parse_only
    _WORKER_SKIP_FAILED = skip_failed
    _WORKER_PARSER = None
    _WORKER_PARSED_SINCE_RESTART = 0


def get_worker_parser() -> Any:
    global _WORKER_PARSER
    global _WORKER_PARSED_SINCE_RESTART

    if _WORKER_GRAMMAR is None:
        raise RuntimeError("Worker grammar was not initialized")
    if _WORKER_ACE_BIN is None:
        raise RuntimeError("Worker ACE path was not initialized")

    if _WORKER_PARSER is None:
        _WORKER_PARSER = make_parser(_WORKER_GRAMMAR, _WORKER_ACE_BIN, _WORKER_MAX_PARSES)
        _WORKER_PARSED_SINCE_RESTART = 0
        return _WORKER_PARSER

    if _WORKER_RESTART_EVERY > 0 and _WORKER_PARSED_SINCE_RESTART >= _WORKER_RESTART_EVERY:
        close_parser(_WORKER_PARSER)
        _WORKER_PARSER = make_parser(_WORKER_GRAMMAR, _WORKER_ACE_BIN, _WORKER_MAX_PARSES)
        _WORKER_PARSED_SINCE_RESTART = 0

    return _WORKER_PARSER


def restart_worker_parser() -> None:
    global _WORKER_PARSER
    global _WORKER_PARSED_SINCE_RESTART

    close_parser(_WORKER_PARSER)
    _WORKER_PARSER = None
    _WORKER_PARSED_SINCE_RESTART = 0


def parse_worker_sentence(sentence: str) -> list[JsonDict]:
    global _WORKER_PARSED_SINCE_RESTART

    parser = get_worker_parser()

    try:
        response = parser.interact(sentence)
    except (OSError, RuntimeError, ValueError):
        restart_worker_parser()
        response = get_worker_parser().interact(sentence)

    _WORKER_PARSED_SINCE_RESTART += 1

    results = response.get("results", [])
    if isinstance(results, list):
        return results
    return []


def output_rows(
    row: JsonDict,
    fallback_id: int,
    results: list[JsonDict],
    first_parse_only: bool,
    skip_failed: bool,
) -> tuple[list[JsonDict], int]:
    out_rows = []
    parse_count = len(results)
    if first_parse_only:
        results = results[:1]

    base = {
        "id": sentence_id(row, fallback_id),
        "sentence": source_sentence(row),
        "pseudo_english": pseudo_sentence(row),
        **metadata(row),
    }

    for parse_index, result in enumerate(results, start=1):
        mrs = result.get("mrs")
        if not isinstance(mrs, str) or not mrs.strip():
            continue

        out_rows.append(
            {
                **base,
                "parse_found": True,
                "parse_count": parse_count,
                "parse_index": parse_index,
                "mrs": normalize_mrs(mrs),
            }
        )

    if out_rows:
        return out_rows, len(out_rows)

    if skip_failed:
        return [], 0

    return [
        {
            **base,
            "parse_found": False,
            "parse_count": parse_count,
            "parse_index": None,
            "mrs": None,
        }
    ], 0


def parse_single(args: argparse.Namespace, input_path: Path, output_path: Path, total: int) -> None:
    saved_count = 0
    success_count = 0

    with output_path.open("w", encoding="utf-8") as f:
        rows = tqdm(iter_pseudo_jsonl(input_path), total=total, desc="Parsing pseudo-English")
        for index, row in enumerate(rows, start=1):
            results = parse_with_oneoff(args.grammar, args.ace_bin, pseudo_sentence(row), args.max_parses)
            parsed_rows, parsed_count = output_rows(
                row,
                index,
                results,
                args.first_parse_only,
                args.skip_failed,
            )

            for parsed_row in parsed_rows:
                f.write(json.dumps(parsed_row, ensure_ascii=False) + "\n")

            saved_count += len(parsed_rows)
            success_count += parsed_count

    print(f"Saved rows: {saved_count}")
    print(f"Successful parses: {success_count}")
    print(f"Output: {output_path}")


def iter_chunks(path: Path, chunksize: int) -> Iterator[list[tuple[int, JsonDict]]]:
    chunk = []
    for index, row in enumerate(iter_pseudo_jsonl(path), start=1):
        chunk.append((index, row))
        if len(chunk) == chunksize:
            yield chunk
            chunk = []

    if chunk:
        yield chunk


def parse_chunk(chunk: list[tuple[int, JsonDict]]) -> tuple[list[JsonDict], int, int]:
    parsed_rows = []
    success_count = 0

    for index, row in chunk:
        results = parse_worker_sentence(pseudo_sentence(row))
        rows, parsed_count = output_rows(
            row,
            index,
            results,
            _WORKER_FIRST_PARSE_ONLY,
            _WORKER_SKIP_FAILED,
        )
        parsed_rows.extend(rows)
        success_count += parsed_count

    return parsed_rows, success_count, len(chunk)


def parse_parallel(args: argparse.Namespace, input_path: Path, output_path: Path, total: int) -> None:
    saved_count = 0
    success_count = 0
    context = mp.get_context("spawn")

    with output_path.open("w", encoding="utf-8") as f:
        with context.Pool(
            processes=args.workers,
            initializer=init_worker,
            initargs=(
                args.grammar,
                args.ace_bin,
                args.max_parses,
                args.restart_every,
                args.first_parse_only,
                args.skip_failed,
            ),
        ) as pool:
            with tqdm(total=total, desc="Parsing pseudo-English") as progress:
                for rows, parsed_count, input_count in pool.imap(
                    parse_chunk,
                    iter_chunks(input_path, args.chunksize),
                    chunksize=1,
                ):
                    for row in rows:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")

                    saved_count += len(rows)
                    success_count += parsed_count
                    progress.update(input_count)

    print(f"Saved rows: {saved_count}")
    print(f"Successful parses: {success_count}")
    print(f"Output: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ace-bin", required=True)
    parser.add_argument("--grammar", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-parses", type=int, default=20)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunksize", type=int, default=100)
    parser.add_argument("--restart-every", type=int, default=5000)
    parser.add_argument("--first-parse-only", action="store_true")
    parser.add_argument("--skip-failed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    grammar_path = Path(args.grammar)
    ace_path = Path(args.ace_bin)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not grammar_path.exists():
        raise FileNotFoundError(f"Grammar file not found: {grammar_path}")
    if not ace_path.exists():
        raise FileNotFoundError(f"ACE binary not found: {ace_path}")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.chunksize < 1:
        raise ValueError("--chunksize must be >= 1")
    if args.restart_every < 0:
        raise ValueError("--restart-every must be >= 0")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = count_valid_rows(input_path)

    if args.workers == 1:
        parse_single(args, input_path, output_path, total)
    else:
        parse_parallel(args, input_path, output_path, total)


if __name__ == "__main__":
    main()
