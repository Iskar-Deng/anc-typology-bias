#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# evaluation/pairs_building/phenomena/2_1_intran_gen_marker/rules.py

"""
Define perturbation rules for 2.1 Intransitive genitive marker.

Usage:
python -m evaluation.pairs_building.apply_perturbation \
  --phenomenon evaluation/pairs_building/phenomena/2_1_intran_gen_marker \
  --input GENERATED_SELECTED_JSONL \
  --output PAIRS_JSONL \
  --sample-size 100 \
  --seed 42
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from evaluation.pairs_building.rule_utils import (
    find_intransitive_template_match,
    load_json_templates,
    replace_token_suffix,
)


PHENOMENON_ID = "2.1"
PHENOMENON_NAME = "intran_gen_marker"
TEMPLATE_PATH = Path(__file__).with_name("templates.json")


TEMPLATES = load_json_templates(TEMPLATE_PATH)


def perturbation_plan(source_index: int) -> tuple[str, str]:
    bad_marker = "ca" if source_index % 2 == 1 else "0"
    return bad_marker, "single_genitive"


def target_layer_label(gen_depth: int, target_layer: int) -> str:
    if gen_depth == 1:
        return "single_possessor"
    raise ValueError(f"Unsupported genitive depth: {gen_depth}")


def perturb(
    good_sentence: str,
    language_config: Dict[str, Any],
    source_index: int,
    row: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    tokens = good_sentence.strip().split()
    clause_wo = language_config["clause_wo"]
    np_wo = language_config["np_wo"]

    pseudo_english = row.get("pseudo_english") if row is not None else None
    if isinstance(pseudo_english, str) and "nmz" in pseudo_english:
        return {
            "skip": True,
            "skip_reason": "pseudo_english_contains_nominalization_artifact",
            "good": good_sentence,
            "tokens": tokens,
            "pseudo_english": pseudo_english,
        }

    template = find_intransitive_template_match(TEMPLATES, tokens, clause_wo, np_wo)
    if template is None:
        return {
            "skip": True,
            "skip_reason": "template_match_count_not_one",
            "good": good_sentence,
            "tokens": tokens,
            "clause_wo": clause_wo,
            "np_wo": np_wo,
        }

    gen_depth = template["gen_depth"]
    target_layer = 1
    bad_marker, plan_group = perturbation_plan(source_index)
    target_index = template["layer_to_gen_index"]["1"]
    target_token = tokens[target_index]

    bad_tokens = tokens[:]
    if bad_marker == "ca":
        bad_tokens[target_index] = replace_token_suffix(target_token, "ge", "ca")
        perturbation = "replace_genitive_ge_with_ca"
    elif bad_marker == "0":
        bad_tokens[target_index] = replace_token_suffix(target_token, "ge", "")
        perturbation = "delete_genitive_ge"
    else:
        raise ValueError(f"Unsupported bad marker: {bad_marker}")

    subject_start = template["subject_start"]
    subject_len = template["subject_len"]
    subject_tokens = tokens[subject_start : subject_start + subject_len]

    return {
        "bad": " ".join(bad_tokens),
        "target_role": "genitive_marker",
        "target_index": target_index,
        "target_token": target_token,
        "subject_span": " ".join(subject_tokens),
        "good_value": "ge",
        "bad_value": bad_marker,
        "gen_depth": gen_depth,
        "target_layer": target_layer,
        "target_layer_label": target_layer_label(gen_depth, target_layer),
        "plan_group": plan_group,
        "template": template["name"],
        "perturbation": perturbation,
    }
