#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# evaluation/pairs_building/phenomena/5_3_anc_ext_P_marker/rules.py

"""
Define perturbation rules for 5.3 ANC external P marker.

Usage:
python -m evaluation.pairs_building.apply_perturbation \
  --phenomenon evaluation/pairs_building/phenomena/5_3_anc_ext_P_marker \
  --input GENERATED_SELECTED_JSONL \
  --output PAIRS_JSONL \
  --sample-size 100 \
  --seed 42
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from evaluation.pairs_building.rule_utils import (
    alternating_noun_marker,
    anc_template_metadata,
    foil_marker_for_external_role,
    load_anc_templates,
    marker_value,
    prepare_anc_external_marker_context,
    replace_marker_on_anc_head,
)


PHENOMENON_ID = "5.3"
PHENOMENON_NAME = "anc_ext_P_marker"
TEMPLATE_PATH = Path(__file__).with_name("templates.json")


TEMPLATES = load_anc_templates(TEMPLATE_PATH)


def perturb(
    good_sentence: str,
    language_config: Dict[str, Any],
    source_index: int,
    row: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    good_marker = language_config["FIN_P_MARK"] or ""
    context, skip = prepare_anc_external_marker_context(
        TEMPLATES,
        PHENOMENON_ID,
        good_sentence,
        source_index,
        row,
        good_marker,
        "p",
    )
    if skip is not None or context is None:
        return skip

    bad_marker, perturbation = foil_marker_for_external_role(
        "p",
        good_marker,
        row,
        source_index,
    )
    base = context.base
    bad_tokens = base.tokens[:]
    bad_tokens[base.anc_verb.index] = replace_marker_on_anc_head(
        base.anc_verb.token,
        good_marker,
        bad_marker,
    )

    return {
        "bad": " ".join(bad_tokens),
        "target_role": "ANC_EXT_P",
        "target_index": base.anc_verb.index,
        "target_token": base.anc_verb.token,
        "target_stem": base.anc_verb.stem.lower(),
        "good_value": marker_value(good_marker),
        "bad_value": marker_value(bad_marker),
        **anc_template_metadata(base.template, language_config),
        "matrix_role": "P",
        "perturbation": perturbation,
    }
