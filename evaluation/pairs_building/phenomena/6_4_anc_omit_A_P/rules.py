#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# evaluation/pairs_building/phenomena/6_4_anc_omit_A_P/rules.py

"""
Define perturbation rules for 6.4 ANC omitted A+P.

Usage:
python -m evaluation.pairs_building.apply_perturbation \
  --phenomenon evaluation/pairs_building/phenomena/6_4_anc_omit_A_P \
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
    pseudo_has_bare_anc,
)


PHENOMENON_ID = "6.4"
PHENOMENON_NAME = "anc_omit_A_P"
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

    if not pseudo_has_bare_anc(context.pseudo_english):
        return {
            "skip": True,
            "skip_reason": "pseudo_english_not_bare_anc",
            "good": good_sentence,
            "source_index": source_index,
            "profile": context.template.profile,
            "pseudo_english": context.pseudo_english,
        }

    bad_tokens = context.tokens[:]
    bad_tokens[context.anc_verb.index] = finite_from_anc_verb(context.anc_verb)

    return {
        "bad": " ".join(bad_tokens),
        "target_role": "ANC_OMITTED_A_P",
        "target_index": context.anc_verb.index,
        "target_token": context.anc_verb.token,
        "target_stem": context.anc_verb.stem.lower(),
        "target_external_case_marker": marker_value(context.anc_verb.marker),
        "good_value": "bare_tv_anc_with_omitted_a_and_p",
        "bad_value": "finite_tv_predicate_without_a_or_p_no_external_case",
        **anc_template_metadata(
            context.template,
            language_config,
            empty_overt_arguments="none",
        ),
        "matrix_role": "P",
        "omitted_argument": "A,P",
        "perturbation": "replace_bare_tv_anc_head_with_finite_tv_predicate",
    }
