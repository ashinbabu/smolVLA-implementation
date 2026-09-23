#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"
CHECKPOINT="${CHECKPOINT:-$PROJECT_DIR/outputs/train/leisaac_pick_orange/checkpoints/020000/pretrained_model}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/eval_videos}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPISODES="${EPISODES:-256}"
SEED="${SEED:-1000}"
ISAACLAB_EVAL_CMD="${ISAACLAB_EVAL_CMD:-}"

if [[ ! -d "$CHECKPOINT" || ! -f "$CHECKPOINT/config.json" || ! -f "$CHECKPOINT/model.safetensors" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  echo "Set CHECKPOINT to a completed SmolVLA pretrained_model directory containing config.json and model.safetensors." >&2
  exit 1
fi

if [[ -z "$ISAACLAB_EVAL_CMD" ]]; then
  cat >&2 <<'EOF'
ISAACLAB_EVAL_CMD is not set. This repository does not contain a canonical
LeIsaac/IsaacLab policy-evaluation entry point, so provide the command supplied
by the LeIsaac task package. The command may use these placeholders:

  {checkpoint} {policy_path} {batch_size} {episodes} {output_dir} {seed}

For example:

  export ISAACLAB_EVAL_CMD='python path/to/evaluate_policy.py --checkpoint {checkpoint} --num_envs {batch_size} --episodes {episodes} --output_dir {output_dir} --seed {seed}'

Commands without placeholders retain the default LeIsaac flags.
EOF
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
if [[ "$ISAACLAB_EVAL_CMD" == *'{'* ]]; then
  RESOLVED_CMD="${ISAACLAB_EVAL_CMD//\{checkpoint\}/$(printf '%q' "$CHECKPOINT") }"
  RESOLVED_CMD="${RESOLVED_CMD//\{policy_path\}/$(printf '%q' "$CHECKPOINT") }"
  RESOLVED_CMD="${RESOLVED_CMD//\{batch_size\}/$(printf '%q' "$BATCH_SIZE") }"
  RESOLVED_CMD="${RESOLVED_CMD//\{episodes\}/$(printf '%q' "$EPISODES") }"
  RESOLVED_CMD="${RESOLVED_CMD//\{output_dir\}/$(printf '%q' "$OUTPUT_DIR") }"
  RESOLVED_CMD="${RESOLVED_CMD//\{seed\}/$(printf '%q' "$SEED") }"
else
  RESOLVED_CMD="$ISAACLAB_EVAL_CMD --checkpoint $(printf '%q' "$CHECKPOINT") --policy.path $(printf '%q' "$CHECKPOINT") --batch_size $(printf '%q' "$BATCH_SIZE") --eval.batch_size $(printf '%q' "$BATCH_SIZE") --episodes $(printf '%q' "$EPISODES") --eval.n_episodes $(printf '%q' "$EPISODES") --output_dir $(printf '%q' "$OUTPUT_DIR") --video_dir $(printf '%q' "$OUTPUT_DIR") --seed $(printf '%q' "$SEED")"
fi

COMMAND=(bash -lc "$RESOLVED_CMD")

printf '%q ' "${COMMAND[@]}" > "$OUTPUT_DIR/command.txt"
printf '\n' >> "$OUTPUT_DIR/command.txt"
"${COMMAND[@]}" 2>&1 | tee "$OUTPUT_DIR/evaluation.log"
