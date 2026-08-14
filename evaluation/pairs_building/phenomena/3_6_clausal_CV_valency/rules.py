#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# evaluation/pairs_building/phenomena/3_6_clausal_CV_valency/rules.py

"""
Define perturbation rules for 3.6 Clausal complement CV valency.

Usage:
python -m evaluation.pairs_building.apply_perturbation \
  --phenomenon evaluation/pairs_building/phenomena/3_6_clausal_CV_valency \
  --input GENERATED_SELECTED_JSONL \
  --output PAIRS_JSONL \
  --sample-size 100 \
  --seed 42
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from evaluation.pairs_building.rule_utils import (
    expand_clausal_templates,
    find_clausal_match,
    load_json_templates,
)

PHENOMENON_ID = "3.6"
PHENOMENON_NAME = "clausal_CV_valency"
TEMPLATE_PATH = Path(__file__).with_name("templates.json")
PAIR_TABLE_PATH = Path(__file__).with_name("valency_pairs.tsv")


TEMPLATES = load_json_templates(TEMPLATE_PATH)
SHAPES = expand_clausal_templates(TEMPLATES)


def stem_from_finite(token: str) -> str | None:
    lower = token.lower()
    for suffix in ("ca", "ge"):
        if lower.endswith(suffix):
            return None
    if lower.endswith("ies") and len(lower) > 3:
        return lower[:-3] + "y"
    if lower.endswith("es") and len(lower) > 2:
        base = lower[:-2]
        if base.endswith(("s", "sh", "ch", "x", "z", "o")):
            return base
    if lower.endswith("s") and len(lower) > 1:
        return lower[:-1]
    return None


def finite_from_stem(stem: str, model_token: str) -> str:
    if stem == "say":
        out = "says"
    elif stem == "have":
        out = "has"
    elif stem == "do":
        out = "does"
    elif stem.endswith("y") and len(stem) > 1 and stem[-2] not in "aeiou":
        out = stem[:-1] + "ies"
    elif stem.endswith(("s", "sh", "ch", "x", "z", "o")):
        out = stem + "es"
    else:
        out = stem + "s"

    if model_token and model_token[0].isupper():
        return out.capitalize()
    return out


def load_pair_table() -> Dict[int, Dict[str, str]]:
    out: Dict[int, Dict[str, str]] = {}
    if not PAIR_TABLE_PATH.is_file():
        return out
    with PAIR_TABLE_PATH.open(encoding="utf-8") as infile:
        header = infile.readline().rstrip("\n").split("\t")
        for raw in infile:
            parts = raw.rstrip("\n").split("\t")
            if len(parts) != len(header):
                continue
            row = dict(zip(header, parts))
            try:
                pair_id = int(row.get("source_id") or row["pair_id"])
            except (KeyError, ValueError):
                continue
            out[pair_id] = row
    return out


PAIR_TABLE = load_pair_table()


def source_pair(row: Dict[str, Any] | None, source_index: int) -> Dict[str, str] | None:
    candidates: List[int] = []
    if row is not None:
        for key in ("source_id", "source_index", "id"):
            value = row.get(key)
            if isinstance(value, int):
                candidates.append(value)
            elif isinstance(value, str):
                try:
                    candidates.append(int(value))
                except ValueError:
                    pass
    candidates.append(source_index)

    for idx in candidates:
        if idx in PAIR_TABLE:
            return PAIR_TABLE[idx]
    return None


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

    pair = source_pair(row=row, source_index=source_index)
    if pair is None:
        return {
            "skip": True,
            "skip_reason": "missing_cv_valency_pair",
            "good": good_sentence,
            "tokens": tokens,
            "source_index": source_index,
            "template": parsed.template_name,
        }

    expected_cv_stem = pair["good_stem"].lower()
    tv_foil_stem = pair["bad_stem"].lower()
    actual_cv_stem = stem_from_finite(parsed.matrix_verb_token)
    if actual_cv_stem != expected_cv_stem:
        return {
            "skip": True,
            "skip_reason": "matrix_cv_stem_mismatch",
            "good": good_sentence,
            "tokens": tokens,
            "source_index": source_index,
            "expected_cv_stem": expected_cv_stem,
            "actual_cv_stem": actual_cv_stem,
            "matrix_verb_token": parsed.matrix_verb_token,
            "template": parsed.template_name,
        }

    bad_tokens = tokens[:]
    bad_tokens[parsed.matrix_verb_index] = finite_from_stem(
        tv_foil_stem,
        parsed.matrix_verb_token,
    )

    metadata: Dict[str, Any] = {
        "bad": " ".join(bad_tokens),
        "target_role": "MATRIX_CV_valency",
        "target_index": parsed.matrix_verb_index,
        "target_token": parsed.matrix_verb_token,
        "matrix_a_span": parsed.matrix_a.text,
        "matrix_verb_token": parsed.matrix_verb_token,
        "matrix_verb_index": parsed.matrix_verb_index,
        "embedded_construction": parsed.embedded_construction,
        "embedded_verb_token": parsed.embedded_verb_token,
        "embedded_verb_index": parsed.embedded_verb_index,
        "good_value": "clausal_complement_verb",
        "bad_value": "ordinary_transitive_verb",
        "source_uid": pair.get("source_uid"),
        "source_pair_id": pair.get("pair_id"),
        "good_stem": expected_cv_stem,
        "bad_stem": tv_foil_stem,
        "cv_stem": expected_cv_stem,
        "tv_foil_stem": tv_foil_stem,
        "cv_thatS_mean": pair.get("cv_thatS_mean"),
        "tv_NPVNP_mean": pair.get("tv_NPVNP_mean"),
        "tv_thatS_mean": pair.get("tv_thatS_mean"),
        "bad_source": pair.get("bad_source"),
        "template": parsed.template_name,
        "perturbation": "replace_matrix_cv_with_transitive_valency_foil",
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
