# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PVTree-course contributors.
# Course-project addition; see LICENSE for terms and README for upstream attribution.
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PCEMODEL = ROOT / "pcemodel"

for path in (ROOT, PCEMODEL):
    path_string = str(path)
    if path_string not in sys.path:
        sys.path.insert(0, path_string)
