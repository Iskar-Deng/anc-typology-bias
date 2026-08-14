#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# evaluation/pairs_building/phenomena/2_3_gen_order/rules.py

"""
Define perturbation rules for 2.3 Genitive possessor--head order.

Usage:
python -m evaluation.pairs_building.apply_perturbation \
  --phenomenon evaluation/pairs_building/phenomena/2_3_gen_order \
  --input GENERATED_SELECTED_JSONL \
  --output PAIRS_JSONL \
  --sample-size 100 \
  --seed 42
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from evaluation.pairs_building.rule_utils import (
    NpSpan,
    find_transitive_template_match,
    finite_s_verb,
    load_json_templates,
    marker_value,
    stable_row_index,
)


PHENOMENON_ID = "2.3"
PHENOMENON_NAME = "gen_order"
TEMPLATE_PATH = Path(__file__).with_name("templates.json")


TEMPLATES = load_json_templates(TEMPLATE_PATH)


def iter_templates(construction: str) -> list[Dict[str, Any]]:
    return [template for template in TEMPLATES if template["construction"] == construction]


def target_role_from_source_index(source_index: int) -> str:
    if 1 <= source_index <= 50:
        return "S"
    if 51 <= source_index <= 100:
        return "A"
    if 101 <= source_index <= 150:
        return "P"
    raise ValueError(f"Unexpected source index for 2.3: {source_index}")


def match_intransitive_template(
    tokens: List[str],
    clause_wo: str,
    np_wo: str,
) -> Dict[str, Any] | None:
    matches = []
    for template in iter_templates("intransitive"):
        if clause_wo not in template["clause_wo"]:
            continue
        if np_wo not in template["np_wo"]:
            continue
        if len(tokens) != template["token_count"]:
            continue
        if not finite_s_verb(tokens[template["verb_index"]], allow_noun_markers=False):
            continue

        target_start = template["target_start"]
        possessor_index = target_start + template["possessor_offset"]
        if not tokens[possessor_index].endswith("ge"):
            continue
        matches.append(template)

    if len(matches) == 1:
        return matches[0]
    return None


def swap_np_order(tokens: List[str], target_np: NpSpan) -> List[str]:
    if len(target_np.tokens) != 2:
        raise ValueError(f"Expected a two-token genitive NP, got: {target_np.text!r}")
    bad_tokens = tokens[:]
    first = target_np.start
    second = target_np.start + 1
    bad_tokens[first], bad_tokens[second] = bad_tokens[second], bad_tokens[first]
    return bad_tokens


def perturb(
    good_sentence: str,
    language_config: Dict[str, Any],
    source_index: int,
    row: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    tokens = good_sentence.strip().split()
    clause_wo = language_config["clause_wo"]
    np_wo = language_config["np_wo"]
    target_role = target_role_from_source_index(stable_row_index(row, source_index))

    pseudo_english = row.get("pseudo_english") if row is not None else None
    if isinstance(pseudo_english, str) and "nmz" in pseudo_english:
        return {
            "skip": True,
            "skip_reason": "pseudo_english_contains_nominalization_artifact",
            "good": good_sentence,
            "tokens": tokens,
            "pseudo_english": pseudo_english,
        }

    if target_role == "S":
        template = match_intransitive_template(tokens, clause_wo, np_wo)
        if template is None:
            return {
                "skip": True,
                "skip_reason": "intransitive_template_match_count_not_one",
                "good": good_sentence,
                "tokens": tokens,
                "clause_wo": clause_wo,
                "np_wo": np_wo,
            }

        target_start = template["target_start"]
        target_np = NpSpan(
            tokens=tokens[target_start : target_start + 2],
            start=target_start,
            head_offset=template["head_offset"],
        )
        verb_token = tokens[template["verb_index"]]
        verb_index = template["verb_index"]
        template_name = template["name"]
        a_span = None
        p_span = None
    else:
        parsed = find_transitive_template_match(
            templates=iter_templates("transitive"),
            tokens=tokens,
            clause_wo=clause_wo,
            np_wo=np_wo,
            a_marker=language_config["FIN_A_MARK"],
            p_marker=language_config["FIN_P_MARK"],
            target_role=target_role,
        )
        if parsed is None:
            return {
                "skip": True,
                "skip_reason": "transitive_template_match_count_not_one",
                "good": good_sentence,
                "tokens": tokens,
                "clause_wo": clause_wo,
                "np_wo": np_wo,
                "target_role": target_role,
                "a_mark": marker_value(language_config["FIN_A_MARK"]),
                "p_mark": marker_value(language_config["FIN_P_MARK"]),
            }

        target_np = parsed.a if target_role == "A" else parsed.p
        verb_token = parsed.verb_token
        verb_index = parsed.verb_index
        template_name = parsed.template_name
        a_span = parsed.a.text
        p_span = parsed.p.text

    possessor_index = target_np.possessor_index
    if possessor_index is None or not tokens[possessor_index].endswith("ge"):
        return {
            "skip": True,
            "skip_reason": "target_np_has_no_genitive_possessor",
            "good": good_sentence,
            "tokens": tokens,
            "target_role": target_role,
            "target_np_span": target_np.text,
        }

    bad_tokens = swap_np_order(tokens, target_np)
    good_value = "possessor_head" if np_wo == "gn" else "head_possessor"
    bad_value = "head_possessor" if np_wo == "gn" else "possessor_head"

    result = {
        "bad": " ".join(bad_tokens),
        "target_role": f"{target_role}_possessor_head_order",
        "target_argument": target_role,
        "target_index": target_np.start,
        "target_token": target_np.text,
        "target_np_span": target_np.text,
        "possessor_index": possessor_index,
        "head_index": target_np.head_index,
        "possessor_token": tokens[possessor_index],
        "head_token": tokens[target_np.head_index],
        "verb_token": verb_token,
        "verb_index": verb_index,
        "good_value": good_value,
        "bad_value": bad_value,
        "template": template_name,
        "perturbation": "swap_np_possessor_and_head_order",
    }

    if a_span is not None:
        result["a_span"] = a_span
    if p_span is not None:
        result["p_span"] = p_span

    return result
