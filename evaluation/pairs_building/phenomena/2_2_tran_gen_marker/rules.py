#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# evaluation/pairs_building/phenomena/2_2_tran_gen_marker/rules.py

"""
Define perturbation rules for 2.2 Transitive genitive marker.

Usage:
python -m evaluation.pairs_building.apply_perturbation \
  --phenomenon evaluation/pairs_building/phenomena/2_2_tran_gen_marker \
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
    replace_token_suffix,
    stable_row_index,
    transitive_shape_from_pseudo,
)


PHENOMENON_ID = "2.2"
PHENOMENON_NAME = "tran_gen_marker"
TEMPLATE_PATH = Path(__file__).with_name("templates.json")


TEMPLATES = load_json_templates(TEMPLATE_PATH)


def target_role_from_source_index(source_index: int) -> str:
    if 1 <= source_index <= 40:
        return "A"
    if 41 <= source_index <= 80:
        return "P"
    if 81 <= source_index <= 120:
        return "A"
    if 121 <= source_index <= 160:
        return "P"
    return "A" if source_index % 2 == 1 else "P"


def bad_marker_from_source_index(source_index: int) -> tuple[str, str]:
    bad_marker = "ca" if source_index % 2 == 1 else "0"
    if bad_marker == "ca":
        return bad_marker, "replace_genitive_ge_with_ca"
    return bad_marker, "delete_genitive_ge"


def perturb(
    good_sentence: str,
    language_config: Dict[str, Any],
    source_index: int,
    row: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    tokens = good_sentence.strip().split()

    clause_wo = language_config["clause_wo"]
    np_wo = language_config["np_wo"]
    good_a_mark = language_config["FIN_A_MARK"]
    good_p_mark = language_config["FIN_P_MARK"]

    pseudo_english = row.get("pseudo_english") if row is not None else None
    if isinstance(pseudo_english, str) and "nmz" in pseudo_english:
        return {
            "skip": True,
            "skip_reason": "pseudo_english_contains_nominalization_artifact",
            "good": good_sentence,
            "tokens": tokens,
            "pseudo_english": pseudo_english,
        }

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
            "a_mark": marker_value(good_a_mark),
            "p_mark": marker_value(good_p_mark),
        }

    stable_index = stable_row_index(row, source_index)
    target_role = target_role_from_source_index(stable_index)
    target_np = parsed.a if target_role == "A" else parsed.p
    target_index = target_np.genitive_index
    if target_index is None:
        return {
            "skip": True,
            "skip_reason": f"target_{target_role.lower()}_is_not_genitive",
            "good": good_sentence,
            "tokens": tokens,
            "target_role": target_role,
            "template": parsed.template_name,
        }

    target_token = tokens[target_index]
    bad_marker, perturbation = bad_marker_from_source_index(stable_index)
    bad_tokens = tokens[:]
    if bad_marker == "ca":
        bad_tokens[target_index] = replace_token_suffix(target_token, "ge", "ca")
    elif bad_marker == "0":
        bad_tokens[target_index] = replace_token_suffix(target_token, "ge", "")
    else:
        raise ValueError(f"Unsupported bad marker: {bad_marker}")

    return {
        "bad": " ".join(bad_tokens),
        "target_role": f"{target_role}_genitive_marker",
        "target_argument": target_role,
        "target_index": target_index,
        "target_token": target_token,
        "a_span": parsed.a.text,
        "p_span": parsed.p.text,
        "verb_token": parsed.verb_token,
        "verb_index": parsed.verb_index,
        "good_value": "ge",
        "bad_value": bad_marker,
        "a_marker": marker_value(good_a_mark),
        "p_marker": marker_value(good_p_mark),
        "target_np_span": target_np.text,
        "template": parsed.template_name,
        "perturbation": perturbation,
    }
