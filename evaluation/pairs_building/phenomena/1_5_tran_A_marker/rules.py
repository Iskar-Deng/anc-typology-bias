#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# evaluation/pairs_building/phenomena/1_5_tran_A_marker/rules.py

"""
Define perturbation rules for 1.5 Transitive A marker.

Usage:
python -m evaluation.pairs_building.apply_perturbation \
  --phenomenon evaluation/pairs_building/phenomena/1_5_tran_A_marker \
  --input GENERATED_SELECTED_JSONL \
  --output PAIRS_JSONL \
  --sample-size 100 \
  --seed 42
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from evaluation.pairs_building.rule_utils import (
    find_transitive_template_match,
    load_json_templates,
    marker_value,
    stable_row_index,
    strip_nonempty_suffix,
    transitive_shape_from_pseudo,
)


PHENOMENON_ID = "1.5"
PHENOMENON_NAME = "tran_A_marker"
TEMPLATE_PATH = Path(__file__).with_name("templates.json")


TEMPLATES = load_json_templates(TEMPLATE_PATH)


def foil_for_a(row: Dict[str, Any] | None, fallback_index: int, alignment: str) -> tuple[str, str]:
    use_ge_foil = stable_row_index(row, fallback_index) % 2 == 0

    if use_ge_foil:
        return "ge", "replace_transitive_a_marker_with_ge"

    if alignment == "nom-acc":
        return "ca", "add_ca_to_transitive_a"

    if alignment == "erg-abs":
        return "0", "remove_ca_from_transitive_a"

    raise ValueError(f"Unsupported alignment: {alignment}")


def perturb(
    good_sentence: str,
    language_config: Dict[str, Any],
    source_index: int,
    row: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    tokens = good_sentence.strip().split()

    clause_wo = language_config["clause_wo"]
    np_wo = language_config["np_wo"]
    alignment = language_config["alignment"]
    good_a_mark = language_config["FIN_A_MARK"]
    good_p_mark = language_config["FIN_P_MARK"]

    parsed = find_transitive_template_match(
        templates=TEMPLATES,
        tokens=tokens,
        clause_wo=clause_wo,
        np_wo=np_wo,
        a_marker=good_a_mark,
        p_marker=good_p_mark,
        expected_shape=transitive_shape_from_pseudo(row),
    )
    if parsed is None:
        return {
            "skip": True,
            "skip_reason": "template_match_count_not_one",
            "good": good_sentence,
            "tokens": tokens,
            "clause_wo": clause_wo,
            "np_wo": np_wo,
            "alignment": alignment,
            "good_a_mark": marker_value(good_a_mark),
            "good_p_mark": marker_value(good_p_mark),
        }

    target_index = parsed.a.head_index
    target_token = tokens[target_index]
    bad_value, perturbation_label = foil_for_a(row, source_index, alignment)

    bad_tokens = tokens[:]
    if marker_value(good_a_mark) == "0":
        if bad_value == "0":
            raise ValueError("A is already zero-marked")
        bad_tokens[target_index] = target_token + bad_value
    elif good_a_mark == "ca":
        a_stem = strip_nonempty_suffix(target_token, "ca")
        if bad_value == "0":
            bad_tokens[target_index] = a_stem
        elif bad_value == "ge":
            bad_tokens[target_index] = a_stem + "ge"
        else:
            raise ValueError(f"Unsupported bad_value for ca-marked A: {bad_value}")
    else:
        raise ValueError(f"Unsupported GOOD A marker: {good_a_mark!r}")

    return {
        "bad": " ".join(bad_tokens),
        "target_role": "A",
        "target_index": target_index,
        "target_token": target_token,
        "a_span": parsed.a.text,
        "p_span": parsed.p.text,
        "verb_token": parsed.verb_token,
        "good_value": marker_value(good_a_mark),
        "bad_value": bad_value,
        "template": parsed.template_name,
        "perturbation": perturbation_label,
    }
