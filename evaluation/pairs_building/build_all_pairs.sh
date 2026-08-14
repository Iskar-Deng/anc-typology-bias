#!/usr/bin/env bash
# evaluation/pairs_building/build_all_pairs.sh
#
# Build all minimal-pair files from source sentences and grammars.
#
# Usage:
# bash evaluation/pairs_building/build_all_pairs.sh

set -euo pipefail

SAMPLE_SIZE=100
SEED=42
PARSE_WORKERS=8
GEN_WORKERS=8
CHUNKSIZE=50
MAX_GEN=20

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

DATA_DIR="$(python3 -c 'from utils import DATA_DIR; print(DATA_DIR)')"
TOOLS_DIR="$(python3 -c 'from utils import TOOLS_DIR; print(TOOLS_DIR)')"
GRAMMARS_DIR="$(python3 -c 'from utils import GRAMMARS_DIR; print(GRAMMARS_DIR)')"
EVAL_MATERIALS_DIR="$(python3 -c 'from utils import EVAL_MATERIALS_DIR; print(EVAL_MATERIALS_DIR)')"

ACE_BIN="$TOOLS_DIR/ace/ace"
PSEUDO_GRAMMAR="$GRAMMARS_DIR/pseudo-english/pseudo-english.dat"
TRAIN_LEXICON="$DATA_DIR/train/train_lexicon.json"
PHENOMENA_MANIFEST="evaluation/phenomena_manifest.tsv"
LANGUAGE_MANIFEST="choices/manifest.tsv"

tail -n +2 "$PHENOMENA_MANIFEST" | while IFS=$'\t' read -r phenomenon _figure_stem _label _title; do
  phenomenon_dir="evaluation/pairs_building/phenomena/$phenomenon"
  material_dir="$EVAL_MATERIALS_DIR/$phenomenon"
  extract="$material_dir/${phenomenon}_extract.jsonl"
  pseudo="$material_dir/${phenomenon}_pseudo.jsonl"
  mrs="$material_dir/${phenomenon}_mrs.jsonl"
  raw_dir="$material_dir/generated/raw"
  selected_dir="$material_dir/generated/selected"
  pair_dir="$material_dir/pairs"
  mkdir -p "$material_dir" "$raw_dir" "$selected_dir" "$pair_dir"

  echo
  echo "========== $phenomenon =========="

  force_args=()
  case "$phenomenon" in
    6_1_anc_omit_S) force_args=(--force-anc-source-construction iv) ;;
    6_2_anc_omit_A|6_3_anc_omit_P|6_4_anc_omit_A_P) force_args=(--force-anc-source-construction tv) ;;
  esac

  python -m semantic_extraction.extract_basic \
    --input "$phenomenon_dir/source.txt" \
    --output "$extract"

  python -m semantic_extraction.generate_pseudo_english \
    --input "$extract" \
    --output "$material_dir" \
    --use-lexicon "$TRAIN_LEXICON" \
    "${force_args[@]}"

  python -m semantic_extraction.parse_pseudo_with_grammar \
    --ace-bin "$ACE_BIN" \
    --grammar "$PSEUDO_GRAMMAR" \
    --input "$pseudo" \
    --output "$mrs" \
    --workers "$PARSE_WORKERS" \
    --chunksize "$CHUNKSIZE" \
    --first-parse-only \
    --skip-failed

  tail -n +2 "$LANGUAGE_MANIFEST" | while IFS=$'\t' read -r _id language _rest; do
    grammar="$GRAMMARS_DIR/$language"
    raw="$raw_dir/$language.jsonl"
    selected="$selected_dir/$language.jsonl"
    pairs="$pair_dir/$language.pairs.jsonl"

    echo "----- $language -----"

    python -m language_generation.generate_from_mrs_bank \
      --grammar "$grammar/$language.dat" \
      --ace-bin "$ACE_BIN" \
      --input "$mrs" \
      --output "$raw" \
      --workers "$GEN_WORKERS" \
      --chunksize "$CHUNKSIZE" \
      --max-gen "$MAX_GEN"

    python -m language_generation.select_overgen \
      --input "$raw" \
      --output "$selected" \
      --language "$language" \
      --seed "$SEED"

    python -m evaluation.pairs_building.apply_perturbation \
      --phenomenon "$phenomenon_dir" \
      --input "$selected" \
      --output "$pairs" \
      --seed "$SEED" \
      --sample-size "$SAMPLE_SIZE"
  done
done
