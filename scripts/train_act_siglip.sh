#!/usr/bin/env bash
# Trains the final act_siglip model (v7 = act_siglip_progress_v4, checkpoint 015000) with LeRobot 0.5.2.
# Four runs in sequence; each fine-tunes the checkpoint of the previous one (policy.pretrained_path):
#   01_mask_aug_pilot  50k steps from scratch (v3_annotated, image masks + augmentation)
#   02_open_ft         10k from 01/050000   (no affine augmentation, more weight on "open")
#   03_place_ft        10k from 02/010000   (more weight on "place"/"close")
#   04_progress_v4     15k from 03/007500   (dataset v4 = 300 eps, progress levels)  -> final model
#
# Usage (from anywhere; runs in the repo root):
#   DATA_ROOT=~/.cache/huggingface/lerobot/MariaFPG ./scripts/train_act_siglip.sh            # all 4
#   ./scripts/train_act_siglip.sh 03_place_ft 04_progress_v4                                  # some stages
# DATA_ROOT = folder that contains place_the_cubes_in_the_drawers_v3_annotated/ and _v4/.
# Needs the act_siglip env (lerobot 0.5.2 + `pip install -e lerobot_policy_act_siglip`). Do not run it from a
# folder holding another `lerobot` package (it would shadow the installed one).
set -euo pipefail
cd "$(dirname "$0")/.."

: "${DATA_ROOT:=$HOME/.cache/huggingface/lerobot/MariaFPG}"
DATA_ROOT="${DATA_ROOT%/}"
export ACCELERATE_MIXED_PRECISION="${ACCELERATE_MIXED_PRECISION:-bf16}"  # lerobot 0.5.2 ignores --policy.use_amp

STAGES=("$@")
[ ${#STAGES[@]} -eq 0 ] && STAGES=(01_mask_aug_pilot 02_open_ft 03_place_ft 04_progress_v4)
mkdir -p outputs/train/configs

for s in "${STAGES[@]}"; do
  src="configs/train/$s.json"
  [ -f "$src" ] || { echo "no config $src"; exit 1; }
  cfg="outputs/train/configs/$s.json"  # copy with the real dataset path
  sed "s#\${DATA_ROOT}#$DATA_ROOT#g" "$src" > "$cfg"
  read -r out pre root < <(python -c "import json; c=json.load(open('$cfg')); \
print(c['output_dir'], c['policy']['pretrained_path'] or '-', c['dataset']['root'])")
  [ -d "$root" ] || { echo "dataset not found: $root (set DATA_ROOT)"; exit 1; }
  [ "$pre" = "-" ] || [ -d "$pre" ] || { echo "$s needs $pre: train the previous stage first"; exit 1; }
  [ -e "$out" ] && { echo "$out already exists: delete it or skip stage $s"; exit 1; }
  echo "=== $s -> $out (log $out.log)"
  lerobot-train --config_path="$cfg" 2>&1 | tee "$out.log"
done
