#!/usr/bin/env bash
set -euo pipefail

DAT_PATH="$1"
shift
FREEZER_MEGABYTES=""

if [[ "${1:-}" == "--freezer-megabytes" ]]; then
  FREEZER_MEGABYTES="$2"
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="$(cd "$PROJECT_ROOT" && python3 -c 'from utils import TOOLS_DIR; print(TOOLS_DIR)')"
ACE_BIN="$TOOLS_DIR/ace/ace"

GRAMMAR_DIR="$(dirname "$DAT_PATH")"
DAT_NAME="$(basename "$DAT_PATH")"
CONFIG_PATH="$GRAMMAR_DIR/ace/config.tdl"

if [[ -n "$FREEZER_MEGABYTES" ]]; then
  if grep -q '^freezer-megabytes' "$CONFIG_PATH"; then
    sed -i "s/^freezer-megabytes.*/freezer-megabytes := ${FREEZER_MEGABYTES}./" "$CONFIG_PATH"
  else
    printf "\nfreezer-megabytes := %s.\n" "$FREEZER_MEGABYTES" >> "$CONFIG_PATH"
  fi
fi

cd "$GRAMMAR_DIR"
"$ACE_BIN" -g ace/config.tdl -G "$DAT_NAME"
