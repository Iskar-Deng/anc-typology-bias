#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# evaluation/pairs_building/phenomena/4_6_anc_AV_order/rules.py

"""
Define perturbation rules for 4.6 ANC A--V order.

Usage:
python -m evaluation.pairs_building.apply_perturbation \
  --phenomenon evaluation/pairs_building/phenomena/4_6_anc_AV_order \
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


PHENOMENON_ID = "4.6"
PHENOMENON_NAME = "anc_AV_order"
TEMPLATE_PATH = Path(__file__).with_name("templates.json")


TEMPLATES = load_anc_templates(TEMPLATE_PATH)


def contiguous_span(indices: List[int]) -> tuple[int, int] | None:
    ordered = sorted(indices)
    if ordered != list(range(ordered[0], ordered[-1] + 1)):
        return None
    return ordered[0], ordered[-1] + 1


def swap_a_with_vp_chunk(
    tokens: List[str],
    a_index: int,
    p_index: int,
    v_index: int,
) -> tuple[List[str], str, str, List[str], List[str]] | None:
    vp_span = contiguous_span([p_index, v_index])
    if vp_span is None:
        return None

    a_span = (a_index, a_index + 1)
    if a_span[1] == vp_span[0]:
        good_value, bad_value = "AV", "VA"
        bad_tokens = (
            tokens[: a_span[0]]
            + tokens[vp_span[0] : vp_span[1]]
            + tokens[a_span[0] : a_span[1]]
            + tokens[vp_span[1] :]
        )
    elif vp_span[1] == a_span[0]:
        good_value, bad_value = "VA", "AV"
        bad_tokens = (
            tokens[: vp_span[0]]
            + tokens[a_span[0] : a_span[1]]
            + tokens[vp_span[0] : vp_span[1]]
            + tokens[a_span[1] :]
        )
    else:
        return None

    anc_span = contiguous_span([a_index, p_index, v_index])
    if anc_span is None:
        return None

    return (
        bad_tokens,
        good_value,
        bad_value,
        tokens[a_span[0] : a_span[1]],
        tokens[vp_span[0] : vp_span[1]],
    )


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
        a_missing_reason="anc_a_target_head_match_count_not_one",
        p_missing_reason="anc_p_validation_head_match_count_not_one",
    )
    if skip is not None or context is None:
        return skip

    good_a_marker = expected_internal_anc_marker("A", language_config)
    skip = anc_tv_marker_mismatch_skip(
        context,
        "A",
        good_a_marker,
        language_config,
        source_index,
        good_sentence,
    )
    if skip is not None:
        return skip

    good_p_marker = expected_internal_anc_marker("P", language_config)
    skip = anc_tv_marker_mismatch_skip(
        context,
        "P",
        good_p_marker,
        language_config,
        source_index,
        good_sentence,
    )
    if skip is not None:
        return skip

    swapped = swap_a_with_vp_chunk(
        context.base.tokens,
        context.a_index,
        context.p_index,
        context.base.anc_verb.index,
    )
    if swapped is None:
        return {
            "skip": True,
            "skip_reason": "anc_a_vp_order_not_contiguous_or_unexpected",
            "good": good_sentence,
            "tokens": context.base.tokens,
            "a_index": context.a_index,
            "p_index": context.p_index,
            "anc_verb_index": context.base.anc_verb.index,
            "anc_wo": language_config["anc_wo"],
            "anc_wo_choice": language_config.get("anc_wo_choice", language_config["anc_wo"]),
            "anc_iv_order": language_config.get("anc_iv_order", ""),
            "anc_tv_order": language_config.get("anc_tv_order", ""),
            "source_index": source_index,
            "profile": context.base.template.profile,
            "pseudo_english": context.base.pseudo_english,
        }

    bad_tokens, good_value, bad_value, a_span_tokens, vp_chunk_tokens = swapped

    return {
        "bad": " ".join(bad_tokens),
        "target_role": "ANC_AV_order",
        "target_index": context.a_index,
        "target_token": context.a_token,
        "anc_verb_token": context.base.anc_verb.token,
        "anc_verb_index": context.base.anc_verb.index,
        "expected_a_head": context.pseudo_args.a_head,
        "expected_p_head": context.pseudo_args.p_head,
        "p_validation_index": context.p_index,
        "p_validation_token": context.p_token,
        "a_span": " ".join(a_span_tokens),
        "p_span": context.p_token,
        "vp_chunk": " ".join(vp_chunk_tokens),
        "good_value": good_value,
        "bad_value": bad_value,
        **anc_template_metadata(context.base.template, language_config, empty_overt_arguments=""),
        "perturbation": "swap_anc_a_span_with_vp_chunk",
    }
