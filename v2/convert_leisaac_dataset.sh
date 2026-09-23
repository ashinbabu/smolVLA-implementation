#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"
LEROBOT_SRC="${LEROBOT_SRC:-$PROJECT_DIR/../v1/lerobot/src}"
DATASET_ID="${DATASET_ID:-LightwheelAI/leisaac-pick-orange}"
DATASET_ROOT="${DATASET_ROOT:-}"

if [[ ! -d "$LEROBOT_SRC/lerobot" ]]; then
  echo "LeRobot source directory not found: $LEROBOT_SRC" >&2
  exit 1
fi

ARGS=(
  --repo-id "$DATASET_ID"
  --push-to-hub false
)
if [[ -n "$DATASET_ROOT" ]]; then
  ARGS+=(--root "$DATASET_ROOT")
fi

PYTHONPATH="$LEROBOT_SRC${PYTHONPATH:+:$PYTHONPATH}" \
  python -m lerobot.scripts.convert_dataset_v21_to_v30 "${ARGS[@]}"
