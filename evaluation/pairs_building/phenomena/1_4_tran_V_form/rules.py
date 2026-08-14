#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# evaluation/pairs_building/phenomena/1_4_tran_V_form/rules.py

"""
Define perturbation rules for 1.4 Transitive finite verb form.

Usage:
python -m evaluation.pairs_building.apply_perturbation \
  --phenomenon evaluation/pairs_building/phenomena/1_4_tran_V_form \
  --input GENERATED_SELECTED_JSONL \
  --output PAIRS_JSONL \
  --sample-size 100 \
  --seed 42
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from evaluation.pairs_building.rule_utils import (
    find_transitive_template_match,
    load_json_templates,
    marker_value,
    transitive_shape_from_pseudo,
)


PHENOMENON_ID = "1.4"
PHENOMENON_NAME = "tran_V_form"
TEMPLATE_PATH = Path(__file__).with_name("templates.json")


TEMPLATES = load_json_templates(TEMPLATE_PATH)


def finite_to_nonfinite(token: str) -> str:
    if not token.endswith("s"):
        raise ValueError(f"Expected finite verb token ending in -s, got: {token}")

    stem = token[:-1]
    if not stem:
        raise ValueError(f"Could not recover stem from finite verb token: {token}")

    return stem + "ing"


def perturb(
    good_sentence: str,
    language_config: Dict[str, Any],
    source_index: int,
    row: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    tokens = good_sentence.strip().split()

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

    target_index = parsed.verb_index
    target_token = tokens[target_index]
    bad_token = finite_to_nonfinite(target_token)

    bad_tokens = tokens[:]
    bad_tokens[target_index] = bad_token

    return {
        "bad": " ".join(bad_tokens),
        "target_role": "V",
        "target_index": target_index,
        "target_token": target_token,
        "a_span": parsed.a.text,
        "p_span": parsed.p.text,
        "verb_token": parsed.verb_token,
        "good_value": "finite_s",
        "bad_value": "nonfinite_ing",
        "template": parsed.template_name,
        "perturbation": "replace_transitive_finite_s_with_nonfinite_ing",
    }
