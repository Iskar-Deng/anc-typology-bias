#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# evaluation/pairs_building/phenomena/4_2_anc_S_marker/rules.py

"""
Define perturbation rules for 4.2 ANC S marker.

Usage:
python -m evaluation.pairs_building.apply_perturbation \
  --phenomenon evaluation/pairs_building/phenomena/4_2_anc_S_marker \
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


PHENOMENON_ID = "4.2"
PHENOMENON_NAME = "anc_S_marker"
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
    bad_marker = bad_internal_anc_marker("S", language_config, context.good_marker)
    bad_tokens = base.tokens[:]
    bad_tokens[context.s_index] = replace_expected_head_marker(
        context.s_token,
        context.s_head,
        context.good_marker,
        bad_marker,
    )

    return {
        "bad": " ".join(bad_tokens),
        "target_role": "ANC_S",
        "target_index": context.s_index,
        "target_token": context.s_token,
        "anc_verb_token": base.anc_verb.token,
        "anc_verb_index": base.anc_verb.index,
        "expected_s_head": context.s_head,
        "good_value": marker_value(context.good_marker),
        "bad_value": bad_marker,
        **anc_template_metadata(base.template, language_config, empty_overt_arguments=""),
        "perturbation": "replace_anc_s_marker_with_ungrammatical_value",
    }

