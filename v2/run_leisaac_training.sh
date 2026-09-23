#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"
LEROBOT_SRC="${LEROBOT_SRC:-$PROJECT_DIR/../v1/lerobot/src}"
DATASET_ID="${DATASET_ID:-LightwheelAI/leisaac-pick-orange}"
DATASET_ROOT="${DATASET_ROOT:-${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}/${DATASET_ID}}"
PRETRAINED_PATH="${PRETRAINED_PATH:-lerobot/smolvla_base}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/train/leisaac_pick_orange}"
STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-32}"

if [[ ! -d "$LEROBOT_SRC/lerobot" ]]; then
  echo "LeRobot source directory not found: $LEROBOT_SRC" >&2
  exit 1
fi

if [[ "$DATASET_ID" == "LightwheelAI/leisaac-pick-orange" && ! -f "$DATASET_ROOT/meta/info.json" ]]; then
  echo "Converted local dataset not found at: $DATASET_ROOT" >&2
  echo "Run ./convert_leisaac_dataset.sh first, or set DATASET_ROOT explicitly." >&2
  exit 1
fi

if [[ "${CUDA_VISIBLE_DEVICES+x}" == x && -z "${CUDA_VISIBLE_DEVICES}" ]]; then
  echo "CUDA_VISIBLE_DEVICES is empty; unset it or choose a GPU before training." >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_DIR")"

PYTHONPATH="$LEROBOT_SRC${PYTHONPATH:+:$PYTHONPATH}" python -m lerobot.scripts.lerobot_train \
  --policy.type=smolvla \
  --policy.pretrained_path="$PRETRAINED_PATH" \
  --policy.load_vlm_weights=true \
  --policy.push_to_hub=false \
  --dataset.repo_id="$DATASET_ID" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.video_backend=pyav \
  --batch_size="$BATCH_SIZE" \
  --steps="$STEPS" \
  --save_freq=5000 \
  --policy.scheduler_decay_steps="$STEPS" \
  --output_dir="$OUTPUT_DIR" \
  --job_name=smolvla_leisaac_pick_orange \
  --policy.device=cuda \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true
