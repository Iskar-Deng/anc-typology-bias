#!/usr/bin/env bash
set -euo pipefail

SPLIT="$1"
WORKERS=8
CHUNKSIZE=50
MAX_GEN=20

shift
while [[ $# -gt 0 ]]; do
  case "$1" in
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    --chunksize)
      CHUNKSIZE="$2"
      shift 2
      ;;
    --max-gen)
      MAX_GEN="$2"
      shift 2
      ;;
  esac
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DATA_DIR="$(python3 -c 'from utils import DATA_DIR; print(DATA_DIR)')"
TOOLS_DIR="$(python3 -c 'from utils import TOOLS_DIR; print(TOOLS_DIR)')"
GRAMMARS_DIR="$(python3 -c 'from utils import GRAMMARS_DIR; print(GRAMMARS_DIR)')"

ACE_BIN="$TOOLS_DIR/ace/ace"
MRS="$DATA_DIR/$SPLIT/${SPLIT}_mrs.jsonl"
RAW_DIR="$DATA_DIR/$SPLIT/generated/raw"
SELECTED_DIR="$DATA_DIR/$SPLIT/generated/selected"
STATS_DIR="$DATA_DIR/$SPLIT/generated/stats"

mkdir -p "$RAW_DIR" "$SELECTED_DIR" "$STATS_DIR"

for grammar in "$GRAMMARS_DIR"/[0-9][0-9]_*; do
  language="$(basename "$grammar")"
  dat="$grammar/$language.dat"
  raw="$RAW_DIR/$language.jsonl"
  selected="$SELECTED_DIR/$language.jsonl"
  stats="$STATS_DIR/$language.json"

  echo
  echo "========== $language =========="

  python -m language_generation.generate_from_mrs_bank \
    --grammar "$dat" \
    --ace-bin "$ACE_BIN" \
    --input "$MRS" \
    --output "$raw" \
    --workers "$WORKERS" \
    --chunksize "$CHUNKSIZE" \
    --max-gen "$MAX_GEN"

  python -m language_generation.select_overgen \
    --input "$raw" \
    --output "$selected" \
    --language "$language" \
    --stats-output "$stats"
done
