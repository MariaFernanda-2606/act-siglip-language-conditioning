#!/usr/bin/env bash
# One-command Qwen + kinematic hybrid annotation and evaluation pipeline.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_hybrid_pipeline.sh --dataset-root PATH --qwen-out-root PATH --ground-truth PATH --split PATH --model-id ID --output-prefix PATH [options]

The Qwen output directory and every output-prefix file must not already exist.
The local Qwen server must already be running.

Options:
  --subset NAME       Split section to annotate/evaluate; default: test
  --all-episodes      Annotate every episode in the dataset (ignores --subset)
  --camera-key KEY    Default: observation.images.cam_low
  --api-base URL      Default: http://localhost:8000/v1
  --help
EOF
}

DATASET_ROOT=""; QWEN_OUT_ROOT=""; GROUND_TRUTH=""; SPLIT=""; MODEL_ID=""; OUTPUT_PREFIX=""
SUBSET="test"; ALL_EPISODES=false; CAMERA_KEY="observation.images.cam_low"; API_BASE="http://localhost:8000/v1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --qwen-out-root) QWEN_OUT_ROOT="$2"; shift 2 ;;
    --ground-truth) GROUND_TRUTH="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --model-id) MODEL_ID="$2"; shift 2 ;;
    --output-prefix) OUTPUT_PREFIX="$2"; shift 2 ;;
    --subset) SUBSET="$2"; shift 2 ;;
    --all-episodes) ALL_EPISODES=true; shift ;;
    --camera-key) CAMERA_KEY="$2"; shift 2 ;;
    --api-base) API_BASE="$2"; shift 2 ;;
    --help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$DATASET_ROOT" && -n "$QWEN_OUT_ROOT" && -n "$GROUND_TRUTH" && -n "$SPLIT" && -n "$MODEL_ID" && -n "$OUTPUT_PREFIX" ]] || { usage >&2; exit 2; }
[[ ! -e "$QWEN_OUT_ROOT" ]] || { echo "Refusing to overwrite $QWEN_OUT_ROOT" >&2; exit 2; }
for suffix in _qwen_predictions.json _kinematic_predictions.json _hybrid_predictions.json _metrics.json; do
  [[ ! -e "${OUTPUT_PREFIX}${suffix}" ]] || { echo "Refusing to overwrite ${OUTPUT_PREFIX}${suffix}" >&2; exit 2; }
done

if "$ALL_EPISODES"; then
  EPISODES="$(python -c 'import json,sys; n=int(json.load(open(sys.argv[1]))["total_episodes"]); print(json.dumps(list(range(n)), separators=(",", ":")))' "$DATASET_ROOT/meta/info.json")"
  SEGMENT_TARGET=(--episodes $(python -c 'import json,sys; n=int(json.load(open(sys.argv[1]))["total_episodes"]); print(*range(n))' "$DATASET_ROOT/meta/info.json"))
  RUN_NAME="all dataset"
else
  EPISODES="$(python -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))[sys.argv[2]]["episodes"], separators=(",", ":")))' "$SPLIT" "$SUBSET")"
  SEGMENT_TARGET=(--subset "$SUBSET")
  RUN_NAME="${SUBSET}"
fi
EVALUATION_EPISODES=( $(python -c 'import json,sys; print(*json.load(open(sys.argv[1]))["test"]["episodes"])' "$SPLIT") )

echo "[1/4] Qwen annotations for ${RUN_NAME}: $EPISODES"
"$SCRIPT_DIR/run_qwen_annotation.sh" \
  --dataset-root "$DATASET_ROOT" --out-root "$QWEN_OUT_ROOT" \
  --model-id "$MODEL_ID" --api-base "$API_BASE" --camera-key "$CAMERA_KEY" \
  --episodes "$EPISODES"

echo "[2/4] Exporting Qwen predictions"
python "$SCRIPT_DIR/export_lerobot_qwen_predictions.py" \
  --root "$QWEN_OUT_ROOT" \
  --output "${OUTPUT_PREFIX}_qwen_predictions.json"

echo "[3/4] Kinematic boundaries and hybrid labels"
python "$SCRIPT_DIR/segment_kinematic_subtasks.py" \
  --root "$DATASET_ROOT" --ground-truth "$GROUND_TRUTH" --split "$SPLIT" \
  "${SEGMENT_TARGET[@]}" --smoothing-s 1.0 --min-duration-s 2.5 \
  --output "${OUTPUT_PREFIX}_kinematic_predictions.json"
python "$SCRIPT_DIR/combine_qwen_semantics_kinematics.py" \
  --kinematic "${OUTPUT_PREFIX}_kinematic_predictions.json" \
  --qwen "${OUTPUT_PREFIX}_qwen_predictions.json" \
  --output "${OUTPUT_PREFIX}_hybrid_predictions.json"

echo "[4/4] Evaluation"
python "$SCRIPT_DIR/evaluate_subtasks.py" \
  --ground-truth "$GROUND_TRUTH" \
  --predictions "${OUTPUT_PREFIX}_hybrid_predictions.json" \
  --episodes "${EVALUATION_EPISODES[@]}" \
  --output "${OUTPUT_PREFIX}_metrics.json"
