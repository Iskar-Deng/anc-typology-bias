#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# evaluation/pairs_building/phenomena/6_2_anc_omit_A/rules.py

"""
Define perturbation rules for 6.2 ANC omitted A.

Usage:
python -m evaluation.pairs_building.apply_perturbation \
  --phenomenon evaluation/pairs_building/phenomena/6_2_anc_omit_A \
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
    finite_from_anc_verb,
    load_anc_templates,
    marker_value,
    prepare_anc_single_overt_tv_context,
    replace_expected_head_marker,
)


PHENOMENON_ID = "6.2"
PHENOMENON_NAME = "anc_omit_A"
TEMPLATE_PATH = Path(__file__).with_name("templates.json")


TEMPLATES = load_anc_templates(TEMPLATE_PATH)


def perturb(
    good_sentence: str,
    language_config: Dict[str, Any],
    source_index: int,
    row: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    context, skip = prepare_anc_single_overt_tv_context(
        TEMPLATES,
        PHENOMENON_ID,
        good_sentence,
        source_index,
        row,
        language_config,
        "P",
        pseudo_missing_reason="pseudo_english_missing_p_only_tv_anc",
        head_missing_reason="anc_p_head_match_count_not_one",
    )
    if skip is not None or context is None:
        return skip

    bad_tokens = context.base.tokens[:]
    bad_tokens[context.base.anc_verb.index] = finite_from_anc_verb(context.base.anc_verb)
    bad_tokens[context.index] = replace_expected_head_marker(
        context.token,
        context.head,
        context.good_marker,
        context.bad_marker,
    )

    return {
        "bad": " ".join(bad_tokens),
        "target_role": "ANC_OMITTED_A",
        "target_index": context.base.anc_verb.index,
        "target_token": context.base.anc_verb.token,
        "target_stem": context.base.anc_verb.stem.lower(),
        "target_external_case_marker": marker_value(context.base.anc_verb.marker),
        "p_index": context.index,
        "p_token": context.token,
        "expected_p_head": context.head,
        "p_good_marker": marker_value(context.good_marker),
        "p_bad_marker": marker_value(context.bad_marker),
        "good_value": "tv_anc_with_p_and_omitted_a",
        "bad_value": "finite_tv_predicate_with_p_without_a",
        **anc_template_metadata(context.base.template, language_config, empty_overt_arguments=""),
        "matrix_role": "P",
        "omitted_argument": "A",
        "perturbation": "replace_tv_p_anc_head_with_finite_tv_predicate_and_sync_p_marker",
    }
