#!/usr/bin/env bash
# evaluation/scoring/score_all_pairs.sh
#
# Score all minimal-pair files for one model seed.
#
# Usage:
# bash evaluation/scoring/score_all_pairs.sh seed_42

set -euo pipefail

SEED_DIR="${1:?usage: bash evaluation/scoring/score_all_pairs.sh seed_42}"
MODEL_SIZE="gpt2-small"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

MODELS_DIR="$(python3 -c 'from utils import MODELS_DIR; print(MODELS_DIR)')"
EVAL_MATERIALS_DIR="$(python3 -c 'from utils import EVAL_MATERIALS_DIR; print(EVAL_MATERIALS_DIR)')"
PHENOMENA_MANIFEST="evaluation/phenomena_manifest.tsv"
LANGUAGE_MANIFEST="choices/manifest.tsv"

tail -n +2 "$PHENOMENA_MANIFEST" | while IFS=$'\t' read -r phenomenon _figure_stem _label _title; do
  echo
  echo "========== $phenomenon =========="

  tail -n +2 "$LANGUAGE_MANIFEST" | while IFS=$'\t' read -r _id language _rest; do
    echo "----- $language -----"

    python -m evaluation.scoring.score_pairs \
      --pairs "$EVAL_MATERIALS_DIR/$phenomenon/pairs/$language.pairs.jsonl" \
      --model "$MODELS_DIR/$MODEL_SIZE/$language/$SEED_DIR"
  done
done
