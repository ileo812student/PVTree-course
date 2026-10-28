#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Modified for PVTree-course; copyright 2026 PVTree-course contributors
# applies to modifications only. Original authors retain their rights.
# Upstream: Sunniva-Shang/PVTree-palmvein-generation, commit 5eaf80f.
# Changes: resolve the script directory and enable strict shell error handling.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dataname=palmvein
ckpname=palm_vein_128_black

bash "$script_dir/run_vein.sh" "$dataname" "$ckpname"
