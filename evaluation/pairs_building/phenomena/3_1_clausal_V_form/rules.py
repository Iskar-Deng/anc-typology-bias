#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# evaluation/pairs_building/phenomena/3_1_clausal_V_form/rules.py

"""
Define perturbation rules for 3.1 Clausal complement verb form.

Usage:
python -m evaluation.pairs_building.apply_perturbation \
  --phenomenon evaluation/pairs_building/phenomena/3_1_clausal_V_form \
  --input GENERATED_SELECTED_JSONL \
  --output PAIRS_JSONL \
  --sample-size 100 \
  --seed 42
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from evaluation.pairs_building.rule_utils import (
    complement_verb_form,
    expand_clausal_templates,
    find_clausal_match,
    load_json_templates,
)


PHENOMENON_ID = "3.1"
PHENOMENON_NAME = "clausal_V_form"
TEMPLATE_PATH = Path(__file__).with_name("templates.json")


TEMPLATES = load_json_templates(TEMPLATE_PATH)
SHAPES = expand_clausal_templates(TEMPLATES)


def flip_verb_form(token: str, form: str) -> tuple[str, str]:
    if form == "finite_s":
        if not token.endswith("s"):
            raise ValueError(f"Expected finite -s token, got: {token}")
        stem = token[:-1]
        return stem + "ing", "nonfinite_ing"

    if form == "nonfinite_ing":
        if not token.endswith("ing"):
            raise ValueError(f"Expected nonfinite -ing token, got: {token}")
        stem = token[:-3]
        return stem + "s", "finite_s"

    raise ValueError(f"Unsupported verb form: {form}")


def perturb(
    good_sentence: str,
    language_config: Dict[str, Any],
    source_index: int,
    row: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    tokens = good_sentence.strip().split()
    good_form = complement_verb_form(language_config["comp_system"])

    parsed = find_clausal_match(
        shapes=SHAPES,
        tokens=tokens,
        language_config=language_config,
        row=row,
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
            "good_value": good_form,
        }

    bad_token, bad_form = flip_verb_form(parsed.embedded_verb_token, good_form)
    bad_tokens = tokens[:]
    bad_tokens[parsed.embedded_verb_index] = bad_token

    metadata: Dict[str, Any] = {
        "bad": " ".join(bad_tokens),
        "target_role": "COMP_V",
        "target_index": parsed.embedded_verb_index,
        "target_token": parsed.embedded_verb_token,
        "matrix_a_span": parsed.matrix_a.text,
        "matrix_verb_token": parsed.matrix_verb_token,
        "matrix_verb_index": parsed.matrix_verb_index,
        "embedded_construction": parsed.embedded_construction,
        "embedded_verb_token": parsed.embedded_verb_token,
        "embedded_verb_index": parsed.embedded_verb_index,
        "good_value": good_form,
        "bad_value": bad_form,
        "template": parsed.template_name,
        "perturbation": "flip_clausal_complement_verb_form",
    }

    if parsed.embedded_construction == "iv":
        assert parsed.embedded_s is not None
        metadata["embedded_s_span"] = parsed.embedded_s.text
    else:
        assert parsed.embedded_a is not None
        assert parsed.embedded_p is not None
        metadata["embedded_a_span"] = parsed.embedded_a.text
        metadata["embedded_p_span"] = parsed.embedded_p.text

    return metadata
