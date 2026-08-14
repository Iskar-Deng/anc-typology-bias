#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# evaluation/pairs_building/phenomena/3_2_clausal_S_marker/rules.py

"""
Define perturbation rules for 3.2 Clausal complement S marker.

Usage:
python -m evaluation.pairs_building.apply_perturbation \
  --phenomenon evaluation/pairs_building/phenomena/3_2_clausal_S_marker \
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
)


PHENOMENON_ID = "3.2"
PHENOMENON_NAME = "clausal_S_marker"
TEMPLATE_PATH = Path(__file__).with_name("templates.json")


TEMPLATES = load_json_templates(TEMPLATE_PATH)
SHAPES = expand_clausal_templates(TEMPLATES, allowed_constructions={"iv"})


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
        allowed_constructions={"iv"},
    )

    if parsed is None:
        return {
            "skip": True,
            "skip_reason": "template_match_count_not_one",
            "good": good_sentence,
            "tokens": tokens,
            "clause_wo": language_config["clause_wo"],
            "np_wo": language_config["np_wo"],
            "comp_system": language_config["comp_system"],
        }

    assert parsed.embedded_s is not None
    target_index = parsed.embedded_s.head_index
    target_token = tokens[target_index]
    bad_tokens = tokens[:]
    bad_tokens[target_index] = target_token + "ge"

    return {
        "bad": " ".join(bad_tokens),
        "target_role": "COMP_S",
        "target_index": target_index,
        "target_token": target_token,
        "matrix_a_span": parsed.matrix_a.text,
        "matrix_verb_token": parsed.matrix_verb_token,
        "matrix_verb_index": parsed.matrix_verb_index,
        "embedded_s_span": parsed.embedded_s.text,
        "embedded_s_head_index": target_index,
        "embedded_verb_token": parsed.embedded_verb_token,
        "embedded_verb_index": parsed.embedded_verb_index,
        "good_value": "0",
        "bad_value": "ge",
        "template": parsed.template_name,
        "perturbation": "add_ge_to_clausal_complement_s_head",
    }
