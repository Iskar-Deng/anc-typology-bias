#!/usr/bin/env bash
set -euo pipefail

MODEL_SIZE="$1"
SEED="$2"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DATA_DIR="$(python3 -c 'from utils import DATA_DIR; print(DATA_DIR)')"
MODELS_DIR="$(python3 -c 'from utils import MODELS_DIR; print(MODELS_DIR)')"

TRAIN_DIR="$DATA_DIR/train/generated/selected"
DEV_DIR="$DATA_DIR/dev/generated/selected"

for train_input in "$TRAIN_DIR"/[0-9][0-9]_*.jsonl; do
  language="$(basename "$train_input" .jsonl)"
  dev_input="$DEV_DIR/$language.jsonl"

  echo
  echo "========== $language =========="

  python -m training.train_lm \
    --train-input "$train_input" \
    --dev-input "$dev_input" \
    --output-root "$MODELS_DIR" \
    --seed "$SEED" \
    --model-size "$MODEL_SIZE"
done
