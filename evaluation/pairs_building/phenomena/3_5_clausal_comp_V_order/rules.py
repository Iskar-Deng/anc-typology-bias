#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# evaluation/pairs_building/phenomena/3_5_clausal_comp_V_order/rules.py

"""
Define perturbation rules for 3.5 Clausal complement--V order.

Usage:
python -m evaluation.pairs_building.apply_perturbation \
  --phenomenon evaluation/pairs_building/phenomena/3_5_clausal_comp_V_order \
  --input GENERATED_SELECTED_JSONL \
  --output PAIRS_JSONL \
  --sample-size 100 \
  --seed 42
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, NamedTuple

from evaluation.pairs_building.rule_utils import (
    ClausalMatch,
    expand_clausal_templates,
    find_clausal_match,
    load_json_templates,
)

PHENOMENON_ID = "3.5"
PHENOMENON_NAME = "clausal_comp_V_order"
TEMPLATE_PATH = Path(__file__).with_name("templates.json")


class EmbeddedSpan(NamedTuple):
    start: int
    end: int
    text: str


TEMPLATES = load_json_templates(TEMPLATE_PATH)
SHAPES = expand_clausal_templates(TEMPLATES)


def embedded_clause_span(parsed: ClausalMatch) -> EmbeddedSpan:
    if parsed.embedded_construction == "iv":
        assert parsed.embedded_s is not None
        start = min(parsed.embedded_s.start, parsed.embedded_verb_index)
        end = max(
            parsed.embedded_s.start + len(parsed.embedded_s.tokens),
            parsed.embedded_verb_index + 1,
        )
    else:
        assert parsed.embedded_a is not None
        assert parsed.embedded_p is not None
        start = min(parsed.embedded_a.start, parsed.embedded_p.start, parsed.embedded_verb_index)
        end = max(
            parsed.embedded_a.start + len(parsed.embedded_a.tokens),
            parsed.embedded_p.start + len(parsed.embedded_p.tokens),
            parsed.embedded_verb_index + 1,
        )

    return EmbeddedSpan(start=start, end=end, text="")


def swap_embedded_clause_and_matrix_verb(
    tokens: List[str],
    embedded_start: int,
    embedded_end: int,
    matrix_verb_index: int,
) -> List[str] | None:
    if embedded_start <= matrix_verb_index < embedded_end:
        return None

    matrix_verb = tokens[matrix_verb_index]

    if embedded_end <= matrix_verb_index:
        return (
            tokens[:embedded_start]
            + [matrix_verb]
            + tokens[embedded_start:embedded_end]
            + tokens[embedded_end:matrix_verb_index]
            + tokens[matrix_verb_index + 1 :]
        )

    if matrix_verb_index < embedded_start:
        return (
            tokens[:matrix_verb_index]
            + tokens[embedded_start:embedded_end]
            + [matrix_verb]
            + tokens[matrix_verb_index + 1 : embedded_start]
            + tokens[embedded_end:]
        )

    return None


def comp_v_order_value(embedded_start: int, matrix_verb_index: int) -> str:
    if embedded_start < matrix_verb_index:
        return "COMP_V"
    return "V_COMP"


def perturb(
    good_sentence: str,
    language_config: Dict[str, Any],
    source_index: int,
    row: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    tokens = good_sentence.strip().split()

    parsed = find_clausal_match(
        shapes=SHAPES,
        tokens=tokens,
        language_config=language_config,
        row=row,
    )
    if parsed is None:
        return {
            "skip": True,
            "skip_reason": "template_match_count_not_one",
            "good": good_sentence,
            "tokens": tokens,
            "clause_wo": language_config["clause_wo"],
            "np_wo": language_config["np_wo"],
            "comp_system": language_config["comp_system"],
            "good_value": None,
        }

    embedded = embedded_clause_span(parsed)
    bad_tokens = swap_embedded_clause_and_matrix_verb(
        tokens=tokens,
        embedded_start=embedded.start,
        embedded_end=embedded.end,
        matrix_verb_index=parsed.matrix_verb_index,
    )
    if bad_tokens is None:
        return {
            "skip": True,
            "skip_reason": "embedded_clause_overlaps_matrix_verb",
            "good": good_sentence,
            "tokens": tokens,
            "template": parsed.template_name,
        }

    good_order = comp_v_order_value(embedded.start, parsed.matrix_verb_index)
    bad_order = "V_COMP" if good_order == "COMP_V" else "COMP_V"

    metadata: Dict[str, Any] = {
        "bad": " ".join(bad_tokens),
        "target_role": "COMP_MATRIX_V_order",
        "target_index": min(embedded.start, parsed.matrix_verb_index),
        "target_token": parsed.matrix_verb_token,
        "matrix_a_span": parsed.matrix_a.text,
        "matrix_verb_token": parsed.matrix_verb_token,
        "matrix_verb_index": parsed.matrix_verb_index,
        "embedded_clause_span": " ".join(tokens[embedded.start : embedded.end]),
        "embedded_clause_start": embedded.start,
        "embedded_clause_end": embedded.end,
        "embedded_construction": parsed.embedded_construction,
        "embedded_verb_token": parsed.embedded_verb_token,
        "embedded_verb_index": parsed.embedded_verb_index,
        "good_value": good_order,
        "bad_value": bad_order,
        "template": parsed.template_name,
        "perturbation": "swap_clausal_complement_span_with_matrix_verb",
    }

    if parsed.embedded_construction == "iv":
        assert parsed.embedded_s is not None
        metadata["embedded_s_span"] = parsed.embedded_s.text
    else:
        assert parsed.embedded_a is not None
        assert parsed.embedded_p is not None
        metadata["embedded_a_span"] = parsed.embedded_a.text
        metadata["embedded_p_span"] = parsed.embedded_p.text

    return metadata
