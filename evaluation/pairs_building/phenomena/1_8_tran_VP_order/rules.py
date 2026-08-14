#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# evaluation/pairs_building/phenomena/1_8_tran_VP_order/rules.py

"""
Define perturbation rules for 1.8 Transitive V--P order.

Usage:
python -m evaluation.pairs_building.apply_perturbation \
  --phenomenon evaluation/pairs_building/phenomena/1_8_tran_VP_order \
  --input GENERATED_SELECTED_JSONL \
  --output PAIRS_JSONL \
  --sample-size 100 \
  --seed 42
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from evaluation.pairs_building.rule_utils import (
    find_transitive_template_match,
    load_json_templates,
    marker_value,
    transitive_shape_from_pseudo,
)


PHENOMENON_ID = "1.8"
PHENOMENON_NAME = "tran_VP_order"
TEMPLATE_PATH = Path(__file__).with_name("templates.json")


TEMPLATES = load_json_templates(TEMPLATE_PATH)


def sentence_case_tokens(tokens: List[str]) -> List[str]:
    if not tokens:
        return tokens

    cased = [token.lower() for token in tokens]
    cased[0] = cased[0][:1].upper() + cased[0][1:]
    return cased


def vp_value(clause_wo: str) -> str:
    if clause_wo == "sov":
        return "PV"
    if clause_wo in {"svo", "vos"}:
        return "VP"
    raise ValueError(f"Unsupported clause_wo: {clause_wo}")


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

    a_tokens = parsed.a.tokens
    p_tokens = parsed.p.tokens
    verb_tokens = [parsed.verb_token]

    if clause_wo == "sov":
        bad_tokens = a_tokens + verb_tokens + p_tokens
    elif clause_wo == "svo":
        bad_tokens = a_tokens + p_tokens + verb_tokens
    elif clause_wo == "vos":
        bad_tokens = p_tokens + verb_tokens + a_tokens
    else:
        raise ValueError(f"Unsupported clause_wo: {clause_wo}")

    good_value = vp_value(clause_wo)
    bad_value = "PV" if good_value == "VP" else "VP"

    return {
        "bad": " ".join(sentence_case_tokens(bad_tokens)),
        "target_role": "V_P_order",
        "target_index": parsed.verb_index,
        "target_token": parsed.verb_token,
        "a_span": parsed.a.text,
        "p_span": parsed.p.text,
        "verb_token": parsed.verb_token,
        "good_value": good_value,
        "bad_value": bad_value,
        "template": parsed.template_name,
        "perturbation": "swap_transitive_v_with_p_span",
    }
