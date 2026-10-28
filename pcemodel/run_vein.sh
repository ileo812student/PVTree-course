#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Modified for PVTree-course; copyright 2026 PVTree-course contributors
# applies to modifications only. Original authors retain their rights.
# Upstream: Sunniva-Shang/PVTree-palmvein-generation, commit 5eaf80f.
# Changes: working-directory independence, input validation, configurable output,
# seed/GPU selection, resume support, quoting and strict shell error handling.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

dataset_name="${1:?usage: run_vein.sh DATASET_NAME_OR_PATH CHECKPOINT_NAME [RESULTS_DIR] [SEED] [--resume]}"
checkpoint_name="${2:?usage: run_vein.sh DATASET_NAME CHECKPOINT_NAME}"
results_dir="${3:-./results/}"
seed="${4:-2026}"
resume="${5:-}"

if [[ -d "$dataset_name" ]]; then
    dataroot="$(cd "$dataset_name" && pwd)"
elif [[ -d "../pv_pattern_results/$dataset_name" ]]; then
    dataroot="../pv_pattern_results/$dataset_name"
else
    echo "Input dataset not found: $dataset_name" >&2
    exit 2
fi

extra_args=()
if [[ "$resume" == "--resume" ]]; then
    extra_args+=(--resume)
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python test.py \
    --dataroot "$dataroot" \
    --name "$checkpoint_name" \
    --results_dir "$results_dir" \
    --seed "$seed" \
    --n_samples 7  \
    --no_flip \
    --load_size 128 \
    --crop_size 128 \
    --input_nc 1 \
    --output_nc 1 \
    --use_dropout \
    --upsample bilinear \
    --netG unet_128 \
    --netD basic_128_multi \
    --netE resnet_128 \
    --dataset_mode single \
    "${extra_args[@]}"
