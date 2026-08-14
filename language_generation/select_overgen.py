# language_generation/select_overgen.py

"""
Select one generated sentence per MRS id.

Usage:
python -m language_generation.select_overgen \
  --input RAW_JSONL \
  --output SELECTED_JSONL \
  --language LANGUAGE_ID \
  --stats-output
"""

from argparse import ArgumentParser, Namespace
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import csv
import json
import random

from utils import ANC_MARK_TABLE, FINITE_MARK_TABLE


JsonDict = dict[str, Any]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "choices" / "manifest.tsv"


def load_jsonl(path: Path) -> list[JsonDict]:
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


def default_stats_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_stats.json")


def load_language_config(language: str) -> JsonDict:
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["language"] != language:
                continue

            config = dict(row)
            config.update(FINITE_MARK_TABLE[config["alignment"]])
            config.update(ANC_MARK_TABLE[(config["strategy"], config["alignment"])])
            return config

    raise ValueError(f"Language not found in {MANIFEST_PATH}: {language}")


def tokenize(sent: str) -> list[str]:
    return sent.strip().split()


def bag_of_words(sent: str) -> Counter:
    return Counter(tok.lower() for tok in tokenize(sent))


def all_same_bag(sents: list[str]) -> bool:
    if len(sents) <= 1:
        return True
    first = bag_of_words(sents[0])
    return all(bag_of_words(sent) == first for sent in sents[1:])


def sort_key(value: Any) -> tuple[int, Any]:
    if isinstance(value, int):
        return (0, value)
    if isinstance(value, str):
        try:
            return (0, int(value))
        except ValueError:
            return (1, value)
    return (2, str(value))


def group_sentences_by_id(rows: list[JsonDict]) -> dict[Any, list[str]]:
    grouped = defaultdict(list)

    for row in rows:
        if "id" not in row:
            continue

        sents = row.get("sent", [])
        if not isinstance(sents, list):
            continue

        grouped[row["id"]].extend(sent for sent in sents if isinstance(sent, str) and sent.strip())

    return {row_id: list(dict.fromkeys(sents)) for row_id, sents in grouped.items()}


def first_metadata_by_id(rows: list[JsonDict]) -> dict[Any, JsonDict]:
    metadata_by_id = {}
    carry_keys = ("source_id", "pseudo_index", "sentence", "pseudo_english")

    for row in rows:
        row_id = row.get("id")
        if row_id is None or row_id in metadata_by_id:
            continue

        metadata_by_id[row_id] = {key: row[key] for key in carry_keys if key in row}

    return metadata_by_id


def nonempty_marks(marks: list[str]) -> list[str]:
    return sorted({mark.lower() for mark in marks if mark}, key=len, reverse=True)


def strip_mark(token: str, marks: list[str]) -> tuple[str, str]:
    token = token.lower()
    for mark in nonempty_marks(marks):
        if token.endswith(mark) and len(token) > len(mark):
            return token[: -len(mark)], mark
    return token, ""


def strip_mark_with_allowed_set(token: str, allowed_marks: list[str]) -> tuple[str, str]:
    allowed = {mark.lower() for mark in allowed_marks}
    base, mark = strip_mark(token, list(allowed))

    if mark in allowed:
        return base, mark
    if "" in allowed:
        return token.lower(), ""
    return token.lower(), "__NO_ALLOWED_MARK__"


def base_bag(sent: str, allowed_marks: list[str]) -> Counter | None:
    bag = Counter()

    for token in tokenize(sent):
        base, mark = strip_mark_with_allowed_set(token, allowed_marks)
        if mark == "__NO_ALLOWED_MARK__":
            return None
        bag[base] += 1

    return bag


def marks_by_base(sent: str, allowed_marks: list[str]) -> dict[str, Counter] | None:
    result = defaultdict(Counter)

    for token in tokenize(sent):
        base, mark = strip_mark_with_allowed_set(token, allowed_marks)
        if mark == "__NO_ALLOWED_MARK__":
            return None
        result[base][mark] += 1

    return result


def detect_single_bag_suffix_variant(sents: list[str], allowed_marks: list[str]) -> str | None:
    if len(sents) <= 1:
        return None

    base_bags = [base_bag(sent, allowed_marks) for sent in sents]
    if any(bag is None for bag in base_bags):
        return None

    first_bag = base_bags[0]
    if any(bag != first_bag for bag in base_bags[1:]):
        return None

    marks_per_sent = [marks_by_base(sent, allowed_marks) for sent in sents]
    if any(marks is None for marks in marks_per_sent):
        return None

    changed_bases = []
    for base in first_bag:
        first_marks = marks_per_sent[0].get(base, Counter())
        if any(marks.get(base, Counter()) != first_marks for marks in marks_per_sent[1:]):
            changed_bases.append(base)

    if len(changed_bases) != 1:
        return None

    changed_base = changed_bases[0]
    observed_marks = set()
    for marks in marks_per_sent:
        observed_marks.update(marks[changed_base].keys())

    allowed = {mark.lower() for mark in allowed_marks}
    if observed_marks.issubset(allowed) and len(observed_marks) > 1:
        return changed_base

    return None


def choose_by_s_mark_variant(
    sents: list[str],
    anc_s_mark: str,
    allowed_marks: list[str],
) -> tuple[str, str] | None:
    changed_base = detect_single_bag_suffix_variant(sents, allowed_marks)
    if changed_base is None:
        return None

    for sent in sents:
        marks = marks_by_base(sent, allowed_marks)
        if marks is not None and marks.get(changed_base, Counter()).get(anc_s_mark.lower(), 0) > 0:
            return sent, "single_bag_suffix_variant_s_mark"

    return None


def differing_positions_for_same_length_sents(sents: list[str]) -> list[int] | None:
    tokenized = [tokenize(sent) for sent in sents]
    sent_len = len(tokenized[0])

    if any(len(tokens) != sent_len for tokens in tokenized[1:]):
        return None

    return [
        i
        for i in range(sent_len)
        if len({tokens[i].lower() for tokens in tokenized}) > 1
    ]


def is_two_token_swap_at_positions(sents: list[str], positions: list[int]) -> bool:
    if len(positions) != 2:
        return False

    tokenized = [tokenize(sent) for sent in sents]
    i, j = positions
    first_pair = sorted([tokenized[0][i].lower(), tokenized[0][j].lower()])

    return all(sorted([tokens[i].lower(), tokens[j].lower()]) == first_pair for tokens in tokenized[1:])


def role_for_token(token: str, anc_a_mark: str, anc_p_mark: str) -> str | None:
    anc_a_mark = anc_a_mark.lower()
    anc_p_mark = anc_p_mark.lower()
    token = token.lower()

    if anc_a_mark == anc_p_mark:
        return None
    if anc_a_mark and token.endswith(anc_a_mark):
        return "A"
    if anc_p_mark and token.endswith(anc_p_mark):
        return "P"
    if anc_a_mark == "" and anc_p_mark:
        return "A"
    if anc_p_mark == "" and anc_a_mark:
        return "P"
    return None


def expected_ap_order(anc_tv_order: str) -> tuple[str, str]:
    if anc_tv_order.index("A") < anc_tv_order.index("P"):
        return "A", "P"
    return "P", "A"


def choose_by_ap_order_swap(
    sents: list[str],
    anc_tv_order: str,
    anc_a_mark: str,
    anc_p_mark: str,
    rng: random.Random,
) -> tuple[str, str] | None:
    if len(sents) <= 1 or not all_same_bag(sents):
        return None

    positions = differing_positions_for_same_length_sents(sents)
    if positions is None or not is_two_token_swap_at_positions(sents, positions):
        return None

    expected = expected_ap_order(anc_tv_order)
    i, j = positions
    good = []

    for sent in sents:
        tokens = tokenize(sent)
        roles = (
            role_for_token(tokens[i], anc_a_mark, anc_p_mark),
            role_for_token(tokens[j], anc_a_mark, anc_p_mark),
        )
        if roles == expected:
            good.append(sent)

    if len(good) == 1:
        return good[0], "same_bag_two_token_swap_order_resolved"
    if len(good) > 1:
        return rng.choice(good), "same_bag_two_token_swap_multiple_match_random"
    return None


def choose_sentence(sents: list[str], config: JsonDict, rng: random.Random) -> tuple[str, str]:
    if not sents:
        return "", "empty"
    if len(sents) == 1:
        return sents[0], "single"

    allowed_marks = [
        config["ANC_S_MARK"],
        config["ANC_A_MARK"],
        config["ANC_P_MARK"],
        config["FIN_S_MARK"],
        config["FIN_A_MARK"],
        config["FIN_P_MARK"],
    ]

    chosen = choose_by_s_mark_variant(sents, config["ANC_S_MARK"], allowed_marks)
    if chosen is not None:
        return chosen

    chosen = choose_by_ap_order_swap(
        sents=sents,
        anc_tv_order=config["anc_tv_order"],
        anc_a_mark=config["ANC_A_MARK"],
        anc_p_mark=config["ANC_P_MARK"],
        rng=rng,
    )
    if chosen is not None:
        return chosen

    if all_same_bag(sents):
        return rng.choice(sents), "same_bag_unresolved_random"

    return rng.choice(sents), "different_bag_random"


def update_stats(stats: Counter, sents: list[str], reason: str) -> None:
    stats["total_unique_ids"] += 1
    stats[reason] += 1

    if len(sents) == 0:
        stats["empty_ids"] += 1
    elif len(sents) == 1:
        stats["single_candidate_ids"] += 1
    else:
        stats["overgenerated_ids"] += 1
        if all_same_bag(sents):
            stats["same_bag_overgenerated_ids"] += 1
        else:
            stats["different_bag_overgenerated_ids"] += 1


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--stats-output", nargs="?", const=True, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats_path = None
    if args.stats_output is True:
        stats_path = default_stats_path(output_path)
    elif args.stats_output:
        stats_path = Path(args.stats_output)

    config = load_language_config(args.language)
    rows = load_jsonl(input_path)
    grouped = group_sentences_by_id(rows)
    metadata_by_id = first_metadata_by_id(rows)
    rng = random.Random(args.seed)

    output_rows = []
    stats = Counter(total_input_rows=len(rows))

    for row_id in sorted(grouped.keys(), key=sort_key):
        sents = grouped[row_id]
        chosen, reason = choose_sentence(sents, config, rng)
        update_stats(stats, sents, reason)

        output = {
            "id": row_id,
            "sent": chosen,
        }
        output.update(metadata_by_id.get(row_id, {}))
        output_rows.append(output)

    write_jsonl(output_rows, output_path)

    stats_dict = {
        "language": args.language,
        "seed": args.seed,
        "clause_wo": config["clause_wo"],
        "np_wo": config["np_wo"],
        "alignment": config["alignment"],
        "comp_system": config["comp_system"],
        "strategy": config["strategy"],
        "anc_choice_word_order": config["anc_choice_word_order"],
        "anc_iv_order": config["anc_iv_order"],
        "anc_tv_order": config["anc_tv_order"],
        "expected_ap_order": "-".join(expected_ap_order(config["anc_tv_order"])),
        **dict(stats),
    }

    if stats_path is not None:
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(stats_dict, f, ensure_ascii=False, indent=2)

    for key, value in stats_dict.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
