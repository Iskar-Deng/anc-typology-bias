#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# evaluation/pairs_building/phenomena/1_3_intran_SV_order/rules.py

"""
Define perturbation rules for 1.3 Intransitive S--V order.

Usage:
python -m evaluation.pairs_building.apply_perturbation \
  --phenomenon evaluation/pairs_building/phenomena/1_3_intran_SV_order \
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


PHENOMENON_ID = "1.3"
PHENOMENON_NAME = "intran_SV_order"
TEMPLATE_PATH = Path(__file__).with_name("templates.json")


TEMPLATES = load_json_templates(TEMPLATE_PATH)


def sentence_case_tokens(tokens: List[str]) -> List[str]:
    if not tokens:
        return tokens

    cased = [token[:1].lower() + token[1:] for token in tokens]
    cased[0] = cased[0][:1].upper() + cased[0][1:]
    return cased


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

    verb_index = template["verb_index"]
    subject_start = template["subject_start"]
    subject_len = template["subject_len"]
    subject_tokens = tokens[subject_start : subject_start + subject_len]
    verb_token = tokens[verb_index]

    if clause_wo in {"sov", "svo"}:
        bad_tokens = [verb_token] + subject_tokens
        good_order = "SV"
        bad_order = "VS"
        target_index = verb_index
    elif clause_wo == "vos":
        bad_tokens = subject_tokens + [verb_token]
        good_order = "VS"
        bad_order = "SV"
        target_index = verb_index
    else:
        raise ValueError(f"Unsupported clause_wo: {clause_wo}")

    bad_tokens = sentence_case_tokens(bad_tokens)

    return {
        "bad": " ".join(bad_tokens),
        "target_role": "S_V_order",
        "target_index": target_index,
        "target_token": verb_token,
        "subject_span": " ".join(subject_tokens),
        "good_value": good_order,
        "bad_value": bad_order,
        "template": template["name"],
        "perturbation": "swap_intransitive_subject_and_finite_verb_order",
    }
