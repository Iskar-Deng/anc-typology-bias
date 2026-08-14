#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# evaluation/pairs_building/phenomena/4_5_anc_SV_order/rules.py

"""
Define perturbation rules for 4.5 ANC S--V order.

Usage:
python -m evaluation.pairs_building.apply_perturbation \
  --phenomenon evaluation/pairs_building/phenomena/4_5_anc_SV_order \
  --input GENERATED_SELECTED_JSONL \
  --output PAIRS_JSONL \
  --sample-size 100 \
  --seed 42
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from evaluation.pairs_building.rule_utils import (
    anc_template_metadata,
    bad_internal_anc_marker,
    load_anc_templates,
    marker_value,
    prepare_anc_iv_context,
    replace_expected_head_marker,
)


PHENOMENON_ID = "4.5"
PHENOMENON_NAME = "anc_SV_order"
TEMPLATE_PATH = Path(__file__).with_name("templates.json")


TEMPLATES = load_anc_templates(TEMPLATE_PATH)


def perturb(
    good_sentence: str,
    language_config: Dict[str, Any],
    source_index: int,
    row: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    context, skip = prepare_anc_iv_context(
        TEMPLATES,
        PHENOMENON_ID,
        good_sentence,
        source_index,
        row,
        language_config,
    )
    if skip is not None or context is None:
        return skip

    base = context.base
    bad_tokens = base.tokens[:]
    bad_tokens[context.s_index], bad_tokens[base.anc_verb.index] = (
        bad_tokens[base.anc_verb.index],
        bad_tokens[context.s_index],
    )

    good_value = "SV" if context.s_index < base.anc_verb.index else "VS"
    bad_value = "VS" if good_value == "SV" else "SV"

    return {
        "bad": " ".join(bad_tokens),
        "target_role": "ANC_SV_order",
        "target_index": context.s_index,
        "target_token": context.s_token,
        "anc_verb_token": base.anc_verb.token,
        "anc_verb_index": base.anc_verb.index,
        "expected_s_head": context.s_head,
        "subject_span": context.s_token,
        "good_value": good_value,
        "bad_value": bad_value,
        **anc_template_metadata(base.template, language_config, empty_overt_arguments=""),
        "perturbation": "swap_anc_s_and_verb_order",
    }

