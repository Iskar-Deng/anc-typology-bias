#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# evaluation/pairs_building/phenomena/6_1_anc_omit_S/rules.py

"""
Define perturbation rules for 6.1 ANC omitted S.

Usage:
python -m evaluation.pairs_building.apply_perturbation \
  --phenomenon evaluation/pairs_building/phenomena/6_1_anc_omit_S \
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
    prepare_anc_base_context,
)


PHENOMENON_ID = "6.1"
PHENOMENON_NAME = "anc_omit_S"
TEMPLATE_PATH = Path(__file__).with_name("templates.json")


TEMPLATES = load_anc_templates(TEMPLATE_PATH)


def perturb(
    good_sentence: str,
    language_config: Dict[str, Any],
    source_index: int,
    row: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    context, skip = prepare_anc_base_context(
        TEMPLATES,
        PHENOMENON_ID,
        good_sentence,
        source_index,
        row,
    )
    if skip is not None or context is None:
        return skip

    bad_tokens = context.tokens[:]
    bad_tokens[context.anc_verb.index] = finite_from_anc_verb(context.anc_verb)

    return {
        "bad": " ".join(bad_tokens),
        "target_role": "ANC_OMITTED_S",
        "target_index": context.anc_verb.index,
        "target_token": context.anc_verb.token,
        "target_stem": context.anc_verb.stem.lower(),
        "target_case_marker": context.anc_verb.marker or "0",
        "good_value": "bare_iv_anc_with_omitted_s",
        "bad_value": "finite_iv_predicate_without_s_no_external_case",
        **anc_template_metadata(
            context.template,
            language_config,
            empty_overt_arguments="none",
        ),
        "matrix_role": "P",
        "omitted_argument": "S",
        "perturbation": "replace_bare_iv_anc_head_with_finite_iv_predicate",
    }
