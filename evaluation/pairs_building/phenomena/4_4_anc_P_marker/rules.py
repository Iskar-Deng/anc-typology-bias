#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# evaluation/pairs_building/phenomena/4_4_anc_P_marker/rules.py

"""
Define perturbation rules for 4.4 ANC P marker.

Usage:
python -m evaluation.pairs_building.apply_perturbation \
  --phenomenon evaluation/pairs_building/phenomena/4_4_anc_P_marker \
  --input GENERATED_SELECTED_JSONL \
  --output PAIRS_JSONL \
  --sample-size 100 \
  --seed 42
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from evaluation.pairs_building.rule_utils import (
    anc_template_metadata,
    anc_tv_marker_mismatch_skip,
    bad_internal_anc_marker,
    expected_internal_anc_marker,
    load_anc_templates,
    marker_value,
    prepare_anc_tv_context,
    replace_expected_head_marker,
)


PHENOMENON_ID = "4.4"
PHENOMENON_NAME = "anc_P_marker"
TEMPLATE_PATH = Path(__file__).with_name("templates.json")


TEMPLATES = load_anc_templates(TEMPLATE_PATH)


def perturb(
    good_sentence: str,
    language_config: Dict[str, Any],
    source_index: int,
    row: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    context, skip = prepare_anc_tv_context(
        TEMPLATES,
        PHENOMENON_ID,
        good_sentence,
        source_index,
        row,
        a_missing_reason="anc_a_validation_head_match_count_not_one",
        p_missing_reason="anc_p_target_head_match_count_not_one",
    )
    if skip is not None or context is None:
        return skip

    good_marker = expected_internal_anc_marker("P", language_config)
    skip = anc_tv_marker_mismatch_skip(
        context,
        "P",
        good_marker,
        language_config,
        source_index,
        good_sentence,
        use_target_fields=True,
    )
    if skip is not None:
        return skip

    bad_marker = bad_internal_anc_marker("P", language_config, good_marker)
    bad_tokens = context.base.tokens[:]
    bad_tokens[context.p_index] = replace_expected_head_marker(
        context.p_token,
        context.pseudo_args.p_head,
        good_marker,
        bad_marker,
    )

    return {
        "bad": " ".join(bad_tokens),
        "target_role": "ANC_P",
        "target_index": context.p_index,
        "target_token": context.p_token,
        "anc_verb_token": context.base.anc_verb.token,
        "anc_verb_index": context.base.anc_verb.index,
        "expected_a_head": context.pseudo_args.a_head,
        "expected_p_head": context.pseudo_args.p_head,
        "a_validation_index": context.a_index,
        "a_validation_token": context.a_token,
        "good_value": marker_value(good_marker),
        "bad_value": bad_marker,
        **anc_template_metadata(context.base.template, language_config, empty_overt_arguments=""),
        "perturbation": "replace_anc_p_marker_with_ungrammatical_value",
    }
