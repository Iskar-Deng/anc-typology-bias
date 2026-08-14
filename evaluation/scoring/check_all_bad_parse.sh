#!/usr/bin/env bash
# evaluation/scoring/check_all_bad_parse.sh
#
# Run BAD parse checks for all minimal-pair files.
#
# Usage:
# bash evaluation/scoring/check_all_bad_parse.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

GRAMMARS_DIR="$(python3 -c 'from utils import GRAMMARS_DIR; print(GRAMMARS_DIR)')"
EVAL_MATERIALS_DIR="$(python3 -c 'from utils import EVAL_MATERIALS_DIR; print(EVAL_MATERIALS_DIR)')"
PHENOMENA_MANIFEST="evaluation/phenomena_manifest.tsv"
LANGUAGE_MANIFEST="choices/manifest.tsv"

tail -n +2 "$PHENOMENA_MANIFEST" | while IFS=$'\t' read -r phenomenon _figure_stem _label _title; do
  echo
  echo "========== $phenomenon =========="

  tail -n +2 "$LANGUAGE_MANIFEST" | while IFS=$'\t' read -r _id language _rest; do
    echo "----- $language -----"

    python -m evaluation.scoring.check_bad_parse \
      --pairs "$EVAL_MATERIALS_DIR/$phenomenon/pairs/$language.pairs.jsonl" \
      --grammar "$GRAMMARS_DIR/$language/$language.dat"
  done
done
