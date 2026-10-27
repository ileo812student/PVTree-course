#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dataname=palmvein
ckpname=palm_vein_128_black

bash "$script_dir/run_vein.sh" "$dataname" "$ckpname"
