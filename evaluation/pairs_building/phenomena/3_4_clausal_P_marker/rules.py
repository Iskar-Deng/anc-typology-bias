#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# evaluation/pairs_building/phenomena/3_4_clausal_P_marker/rules.py

"""
Define perturbation rules for 3.4 Clausal complement P marker.

Usage:
python -m evaluation.pairs_building.apply_perturbation \
  --phenomenon evaluation/pairs_building/phenomena/3_4_clausal_P_marker \
  --input GENERATED_SELECTED_JSONL \
  --output PAIRS_JSONL \
  --sample-size 100 \
  --seed 42
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from evaluation.pairs_building.rule_utils import (
    expand_clausal_templates,
    find_clausal_match,
    load_json_templates,
    marker_value,
    strip_nonempty_suffix,
)


PHENOMENON_ID = "3.4"
PHENOMENON_NAME = "clausal_P_marker"
TEMPLATE_PATH = Path(__file__).with_name("templates.json")


TEMPLATES = load_json_templates(TEMPLATE_PATH)
SHAPES = expand_clausal_templates(TEMPLATES, allowed_constructions={"tv"})


def replace_head_marker(token: str, good_mark: str | None, bad_mark: str) -> str:
    good = good_mark or ""
    if good == "":
        stem = token
    else:
        stem = strip_nonempty_suffix(token, good)

    if bad_mark == "0":
        return stem
    if bad_mark in {"ca", "ge", "ob"}:
        return stem + bad_mark
    raise ValueError(f"Unsupported bad marker: {bad_mark!r}")


def anc_p_foil_marker(language_config: Dict[str, Any]) -> str:
    strategy = language_config["strategy"]
    if strategy == "nomn":
        return "ob"
    if strategy in {"sent", "poss-acc", "erg-poss"}:
        return "ge"
    raise ValueError(f"Unsupported strategy: {strategy}")


def perturb(
    good_sentence: str,
    language_config: Dict[str, Any],
    source_index: int,
    row: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    tokens = good_sentence.strip().split()

    parsed = find_clausal_match(
        shapes=SHAPES,
        tokens=tokens,
        language_config=language_config,
        row=row,
    )
    if parsed is None or parsed.embedded_construction != "tv":
        return {
            "skip": True,
            "skip_reason": "template_match_count_not_one",
            "good": good_sentence,
            "tokens": tokens,
            "clause_wo": language_config["clause_wo"],
            "np_wo": language_config["np_wo"],
            "comp_system": language_config["comp_system"],
            "good_value": marker_value(language_config["FIN_P_MARK"]),
        }

    assert parsed.embedded_a is not None
    assert parsed.embedded_p is not None

    good_mark = language_config["FIN_P_MARK"]
    bad_mark = anc_p_foil_marker(language_config)
    target_np = parsed.embedded_p
    assert target_np is not None
    target_index = target_np.head_index
    target_token = tokens[target_index]
    bad_tokens = tokens[:]
    bad_tokens[target_index] = replace_head_marker(
        token=target_token,
        good_mark=good_mark,
        bad_mark=bad_mark,
    )

    return {
        "bad": " ".join(bad_tokens),
        "target_role": "COMP_P",
        "target_index": target_index,
        "target_token": target_token,
        "matrix_a_span": parsed.matrix_a.text,
        "matrix_verb_token": parsed.matrix_verb_token,
        "matrix_verb_index": parsed.matrix_verb_index,
        "embedded_construction": parsed.embedded_construction,
        "embedded_verb_token": parsed.embedded_verb_token,
        "embedded_verb_index": parsed.embedded_verb_index,
        "embedded_p_span": target_np.text,
        "embedded_p_head_index": target_index,
        "embedded_a_span": parsed.embedded_a.text,
        "good_value": marker_value(good_mark),
        "bad_value": bad_mark,
        "strategy": language_config["strategy"],
        "template": parsed.template_name,
        "perturbation": "replace_clausal_complement_p_marker_with_anc_p_marker",
    }
