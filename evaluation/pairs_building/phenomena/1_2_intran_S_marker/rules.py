#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# evaluation/pairs_building/phenomena/1_2_intran_S_marker/rules.py

"""
Define perturbation rules for 1.2 Intransitive S marker.

Usage:
python -m evaluation.pairs_building.apply_perturbation \
  --phenomenon evaluation/pairs_building/phenomena/1_2_intran_S_marker \
  --input GENERATED_SELECTED_JSONL \
  --output PAIRS_JSONL \
  --sample-size 100 \
  --seed 42
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from evaluation.pairs_building.rule_utils import (
    find_intransitive_template_match,
    load_json_templates,
)

PHENOMENON_ID = "1.2"
PHENOMENON_NAME = "intran_S_marker"
TEMPLATE_PATH = Path(__file__).with_name("templates.json")

TEMPLATES = load_json_templates(TEMPLATE_PATH)

def foil_marker_for_row(row: Dict[str, Any] | None, fallback_index: int) -> str:
    """
    Alternate between real marker foils without relying on source block layout.
    """
    value: Any = fallback_index
    if row is not None:
        value = row.get("id", row.get("source_id", fallback_index))

    try:
        index = int(value)
    except (TypeError, ValueError):
        index = fallback_index

    return "ca" if index % 2 else "ge"

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

    subject_start = template["subject_start"]
    subject_len = template["subject_len"]
    subject_tokens = tokens[subject_start : subject_start + subject_len]
    head_offset = template["subject_head_offset"]
    target_index = subject_start + head_offset
    verb_index = template["verb_index"]

    foil_marker = foil_marker_for_row(row, source_index)

    bad_tokens = tokens[:]
    bad_tokens[target_index] = bad_tokens[target_index] + foil_marker

    return {
        "bad": " ".join(bad_tokens),
        "target_role": "S",
        "target_index": target_index,
        "target_token": tokens[target_index],
        "subject_span": " ".join(subject_tokens),
        "verb_token": tokens[verb_index],
        "good_value": "0",
        "bad_value": foil_marker,
        "template": template["name"],
        "perturbation": f"add_{foil_marker}_to_intransitive_s_head",
    }
