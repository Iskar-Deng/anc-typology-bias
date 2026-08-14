#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# evaluation/pairs_building/phenomena/1_1_intran_V_form/rules.py

"""
Define perturbation rules for 1.1 Intransitive finite verb form.

Usage:
python -m evaluation.pairs_building.apply_perturbation \
  --phenomenon evaluation/pairs_building/phenomena/1_1_intran_V_form \
  --input GENERATED_SELECTED_JSONL \
  --output PAIRS_JSONL \
  --sample-size 100 \
  --seed 42
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from evaluation.pairs_building.rule_utils import (
    find_intransitive_template_match,
    load_json_templates,
)


PHENOMENON_ID = "1.1"
PHENOMENON_NAME = "intran_V_form"
TEMPLATE_PATH = Path(__file__).with_name("templates.json")


TEMPLATES = load_json_templates(TEMPLATE_PATH)


def finite_to_nonfinite(token: str) -> str:
    """
    Convert finite V-s to nonfinite V-ing.

    In these grammars, finite verb form is always suffix -s, and nonfinite
    form is suffix -ing.
    """
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

    template = find_intransitive_template_match(TEMPLATES, tokens, clause_wo, np_wo)
    if template is None:
        return {
            "skip": True,
            "skip_reason": "template_match_count_not_one",
            "good": good_sentence,
            "tokens": tokens,
            "clause_wo": clause_wo,
            "np_wo": np_wo,
        }

    target_index = template["verb_index"]
    good_token = tokens[target_index]
    bad_token = finite_to_nonfinite(good_token)

    bad_tokens = tokens[:]
    bad_tokens[target_index] = bad_token

    return {
        "bad": " ".join(bad_tokens),
        "target_role": "V",
        "target_index": target_index,
        "target_token": good_token,
        "subject_span": " ".join(
            tokens[template["subject_start"] : template["subject_start"] + template["subject_len"]]
        ),
        "good_value": "finite_s",
        "bad_value": "nonfinite_ing",
        "template": template["name"],
        "perturbation": "replace_intransitive_finite_s_with_nonfinite_ing",
    }
