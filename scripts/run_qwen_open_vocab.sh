#!/usr/bin/env bash
# Run the stock LeRobot open-vocabulary subtask prompt against a local Qwen server.
# The source v3 dataset is copied first and is never modified.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_qwen_open_vocab.sh --dataset-root PATH --out-root PATH --episodes JSON [options]

Required:
  --dataset-root PATH   Source LeRobot v3 dataset (never modified)
  --out-root PATH       New annotated copy (must not exist)
  --episodes JSON       Example: '[0]' or '[0,5,7]'

Options:
  --model-id ID         Default: Qwen/Qwen2.5-VL-7B-Instruct
  --api-base URL        Default: http://127.0.0.1:8000/v1
  --camera-key KEY      Default: observation.images.cam_low
  --help

This deliberately does NOT pass --plan.subtask_prompt_path. LeRobot therefore
uses its bundled plan_subtasks.txt prompt, whose labels are free-form imperative
descriptions rather than the project's fixed five-label vocabulary.
EOF
}

DATASET_ROOT=""
OUT_ROOT=""
EPISODES=""
MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct"
API_BASE="http://127.0.0.1:8000/v1"
CAMERA_KEY="observation.images.cam_low"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --episodes) EPISODES="$2"; shift 2 ;;
    --model-id) MODEL_ID="$2"; shift 2 ;;
    --api-base) API_BASE="$2"; shift 2 ;;
    --camera-key) CAMERA_KEY="$2"; shift 2 ;;
    --help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$DATASET_ROOT" && -n "$OUT_ROOT" && -n "$EPISODES" ]] || { usage >&2; exit 2; }
[[ -f "$DATASET_ROOT/meta/info.json" ]] || { echo "Not a LeRobot dataset: $DATASET_ROOT" >&2; exit 2; }
[[ ! -e "$OUT_ROOT" ]] || { echo "Refusing to overwrite existing output: $OUT_ROOT" >&2; exit 2; }

echo "Checking Qwen server at $API_BASE/models"
curl --fail --silent --show-error "$API_BASE/models" >/dev/null
echo "Copying source dataset to: $OUT_ROOT"
cp -a "$DATASET_ROOT" "$OUT_ROOT"

echo "Annotating with LeRobot's stock OPEN-VOCABULARY prompt."
lerobot-annotate \
  --root="$OUT_ROOT" \
  --only_episodes="$EPISODES" \
  --push_to_hub=false \
  --vlm.model_id="$MODEL_ID" \
  --vlm.api_base="$API_BASE" \
  --vlm.auto_serve=false \
  --vlm.camera_key="$CAMERA_KEY" \
  --vlm.client_concurrency=1 \
  --vlm.max_new_tokens=1024 \
  --executor.episode_parallelism=1 \
  --plan.enabled=true \
  --plan.frames_per_second=1.0 \
  --plan.max_frames_per_prompt=40 \
  --plan.subtask_describe_first=true \
  --plan.plan_max_steps=8 \
  --plan.derive_task_from_video=off \
  --plan.emit_plan=false \
  --plan.emit_memory=false \
  --plan.n_task_rephrasings=0 \
  --interjections.enabled=false \
  --vqa.enabled=false
