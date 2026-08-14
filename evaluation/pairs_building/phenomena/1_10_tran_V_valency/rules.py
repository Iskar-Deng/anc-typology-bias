#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# evaluation/pairs_building/phenomena/1_10_tran_V_valency/rules.py

"""
Define perturbation rules for 1.10 Transitive V valency.

Usage:
python -m evaluation.pairs_building.apply_perturbation \
  --phenomenon evaluation/pairs_building/phenomena/1_10_tran_V_valency \
  --input GENERATED_SELECTED_JSONL \
  --output PAIRS_JSONL \
  --sample-size 100 \
  --seed 42
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import csv

from evaluation.pairs_building.rule_utils import (
    find_transitive_template_match,
    load_json_templates,
    marker_value,
    transitive_shape_from_pseudo,
)


PHENOMENON_ID = "1.10"
PHENOMENON_NAME = "tran_V_valency"
TEMPLATE_PATH = Path(__file__).with_name("templates.json")
VALENCY_PATH = Path(__file__).with_name("valency_pairs.tsv")


TEMPLATES = load_json_templates(TEMPLATE_PATH)


def normalize_stem(stem: str) -> str:
    return stem.lower().replace(" ", "")


def load_valency_rows() -> Dict[int, Dict[str, str]]:
    rows_by_source_id: Dict[int, Dict[str, str]] = {}
    with VALENCY_PATH.open(encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile, delimiter="	")
        for row in reader:
            try:
                source_id = int(row["source_id"])
            except (KeyError, ValueError):
                continue
            rows_by_source_id[source_id] = row
    return rows_by_source_id


VALENCY_ROWS = load_valency_rows()


def stem_from_finite(token: str) -> str:
    token = token.lower()
    if token.endswith("s"):
        token = token[:-1]
    return normalize_stem(token)


def finite_form(stem: str, sentence_initial: bool) -> str:
    token = normalize_stem(stem) + "s"
    if sentence_initial:
        return token[:1].upper() + token[1:]
    return token


def perturb(
    good_sentence: str,
    language_config: Dict[str, Any],
    source_index: int,
    row: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    tokens = good_sentence.strip().split()

    pseudo_english = row.get("pseudo_english") if row is not None else None
    if isinstance(pseudo_english, str) and "nmz" in pseudo_english:
        return {
            "skip": True,
            "skip_reason": "pseudo_english_contains_nominalization_artifact",
            "good": good_sentence,
            "tokens": tokens,
            "pseudo_english": pseudo_english,
        }

    clause_wo = language_config["clause_wo"]
    np_wo = language_config["np_wo"]
    good_a_mark = language_config["FIN_A_MARK"]
    good_p_mark = language_config["FIN_P_MARK"]

    parsed = find_transitive_template_match(
        templates=TEMPLATES,
        tokens=tokens,
        clause_wo=clause_wo,
        np_wo=np_wo,
        a_marker=good_a_mark,
        p_marker=good_p_mark,
        expected_shape=transitive_shape_from_pseudo(row),
    )
    if parsed is None:
        return {
            "skip": True,
            "skip_reason": "template_match_count_not_one",
            "good": good_sentence,
            "tokens": tokens,
            "clause_wo": clause_wo,
            "np_wo": np_wo,
            "good_a_mark": marker_value(good_a_mark),
            "good_p_mark": marker_value(good_p_mark),
        }

    verb_index = parsed.verb_index
    verb_token = parsed.verb_token
    good_stem = stem_from_finite(verb_token)
    valency_row = VALENCY_ROWS.get(source_index)
    if valency_row is None:
        return {
            "skip": True,
            "skip_reason": "source_id_not_in_valency_pair_table",
            "good": good_sentence,
            "tokens": tokens,
            "source_index": source_index,
            "good_stem": good_stem,
        }

    expected_good_stem = normalize_stem(valency_row["good_stem"])
    if good_stem != expected_good_stem:
        return {
            "skip": True,
            "skip_reason": "source_valency_good_stem_mismatch",
            "good": good_sentence,
            "tokens": tokens,
            "source_index": source_index,
            "expected_good_stem": expected_good_stem,
            "actual_good_stem": good_stem,
        }

    bad_stem = normalize_stem(valency_row["bad_stem"])
    bad_tokens = tokens[:]
    bad_tokens[verb_index] = finite_form(bad_stem, sentence_initial=(verb_index == 0))

    return {
        "bad": " ".join(bad_tokens),
        "target_role": "V_valency",
        "target_index": verb_index,
        "target_token": verb_token,
        "a_span": parsed.a.text,
        "p_span": parsed.p.text,
        "good_value": "transitive_verb",
        "bad_value": "intransitive_verb",
        "good_stem": good_stem,
        "bad_stem": bad_stem,
        "source_uid": valency_row["source_uid"],
        "source_pair_id": valency_row["pair_id"],
        "blimp_source_uid": valency_row["source_uid"],
        "blimp_pair_id": valency_row["pair_id"],
        "template": parsed.template_name,
        "perturbation": "replace_transitive_verb_with_intransitive_valency_foil",
    }
