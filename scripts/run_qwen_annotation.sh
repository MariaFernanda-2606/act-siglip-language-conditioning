#!/usr/bin/env bash
# Run LeRobot's local annotation pipeline against an already-running Qwen server.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_qwen_annotation.sh --dataset-root PATH --out-root PATH --model-id ID [options]

Required:
  --dataset-root PATH  Source LeRobot v3 dataset (never modified)
  --out-root PATH      New directory for the annotated copy (must not exist)
  --model-id ID        Exact model id exposed by the OpenAI-compatible server

Options:
  --api-base URL       Default: http://localhost:8000/v1
  --camera-key KEY     Default: observation.images.cam_low
  --prompt-path PATH   Default: project prompt for cube/drawer subtasks
  --frames-per-second N  Default: 2.0
  --max-frames-per-prompt N  Default: 80 (avoids a tiny final window)
  --episodes JSON      Example: '[0,5,7]'; omit to annotate every episode
  --help
EOF
}

DATASET_ROOT=""
OUT_ROOT=""
MODEL_ID=""
API_BASE="http://localhost:8000/v1"
CAMERA_KEY="observation.images.cam_low"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROMPT_PATH="$SCRIPT_DIR/../prompts/cube_drawer_subtasks.txt"
FRAMES_PER_SECOND="2.0"
MAX_FRAMES_PER_PROMPT="80"
EPISODES=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --model-id) MODEL_ID="$2"; shift 2 ;;
    --api-base) API_BASE="$2"; shift 2 ;;
    --camera-key) CAMERA_KEY="$2"; shift 2 ;;
    --prompt-path) PROMPT_PATH="$2"; shift 2 ;;
    --frames-per-second) FRAMES_PER_SECOND="$2"; shift 2 ;;
    --max-frames-per-prompt) MAX_FRAMES_PER_PROMPT="$2"; shift 2 ;;
    --episodes) EPISODES="$2"; shift 2 ;;
    --help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$DATASET_ROOT" && -n "$OUT_ROOT" && -n "$MODEL_ID" ]] || { usage >&2; exit 2; }
[[ -f "$DATASET_ROOT/meta/info.json" ]] || { echo "Not a LeRobot dataset: $DATASET_ROOT" >&2; exit 2; }
[[ -f "$PROMPT_PATH" ]] || { echo "Prompt not found: $PROMPT_PATH" >&2; exit 2; }
[[ ! -e "$OUT_ROOT" ]] || { echo "Refusing to overwrite existing output: $OUT_ROOT" >&2; exit 2; }

echo "Checking Qwen server at $API_BASE/models"
curl --fail --silent --show-error "$API_BASE/models" >/dev/null

echo "Copying source dataset to: $OUT_ROOT"
cp -a "$DATASET_ROOT" "$OUT_ROOT"

args=(
  --root="$OUT_ROOT"
  --vlm.api_base="$API_BASE"
  --vlm.auto_serve=false
  --vlm.model_id="$MODEL_ID"
  --vlm.camera_key="$CAMERA_KEY"
  --plan.subtask_prompt_path="$PROMPT_PATH"
  --plan.enabled=true
  --plan.plan_max_steps=5
  --plan.frames_per_second="$FRAMES_PER_SECOND"
  --plan.max_frames_per_prompt="$MAX_FRAMES_PER_PROMPT"
  --plan.emit_plan=false
  --plan.emit_memory=false
  --plan.n_task_rephrasings=0
  --interjections.enabled=false
  --vqa.enabled=false
  --push_to_hub=false
)

if [[ -n "$EPISODES" ]]; then
  args+=(--only_episodes="$EPISODES")
fi

echo "Starting annotation. The source is untouched; annotations will be written to the output copy."
lerobot-annotate "${args[@]}"
