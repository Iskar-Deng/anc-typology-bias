#!/usr/bin/env bash
set -euo pipefail

FREEZER_MEGABYTES=4096

if [[ "${1:-}" == "--freezer-megabytes" ]]; then
  FREEZER_MEGABYTES="$2"
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DATA_DIR="$(python3 -c 'from utils import DATA_DIR; print(DATA_DIR)')"
TOOLS_DIR="$(python3 -c 'from utils import TOOLS_DIR; print(TOOLS_DIR)')"
GRAMMARS_DIR="$(python3 -c 'from utils import GRAMMARS_DIR; print(GRAMMARS_DIR)')"

LEXICON="$DATA_DIR/train/train_lexicon.json"
CHOICES_DIR="$PROJECT_ROOT/choices"
MATRIX_DIR="$TOOLS_DIR/matrix"

mkdir -p "$GRAMMARS_DIR"

for choice in "$CHOICES_DIR"/[0-9][0-9]_*.choice; do
  language="$(basename "$choice" .choice)"
  grammar="$GRAMMARS_DIR/$language"
  dat="$grammar/$language.dat"

  echo
  echo "========== $language =========="

  python "$MATRIX_DIR/matrix.py" \
    --customizationroot "$MATRIX_DIR/gmcs" \
    customize-to-destination \
    "$choice" \
    "$grammar"

  python -m grammar_build.update_grammar_lexicon \
    --lexicon "$LEXICON" \
    --grammar "$grammar"

  python -m grammar_build.patch_anc_wo \
    --grammar "$grammar"

  bash grammar_build/compile_grammar.sh \
    "$dat" \
    --freezer-megabytes "$FREEZER_MEGABYTES"
done
