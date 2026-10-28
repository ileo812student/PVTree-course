#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PVTree-course contributors.
# Course-project addition; see LICENSE for terms and README for upstream attribution.
"""Verify source/A/B image correspondence, hashes, formats and manifests."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
from PIL import Image

from optical_perturb import IMAGE_EXTENSIONS, infer_identity, sha256_file


def image_map(root: Path) -> dict[str, Path]:
    pairs = [
        (path.relative_to(root).as_posix(), path)
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if len(pairs) != len({relative for relative, _ in pairs}):
        raise ValueError(f"Duplicate relative image path under {root}")
    return dict(sorted(pairs))


def read_manifest(path: Path) -> dict[str, dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    result = {record["relative_path"]: record for record in records}
    if len(result) != len(records):
        raise ValueError(f"Duplicate manifest relative_path in {path}")
    return result


def verify(source: Path, group_a: Path, group_b: Path) -> dict[str, Any]:
    roots = {"source": source.resolve(), "A": group_a.resolve(), "B": group_b.resolve()}
    maps = {name: image_map(root) for name, root in roots.items()}
    relative_lists = {name: list(mapping) for name, mapping in maps.items()}
    checks: dict[str, bool] = {
        "recursive_image_lists_equal": relative_lists["source"] == relative_lists["A"] == relative_lists["B"],
        "no_empty_files": all(path.stat().st_size > 0 for mapping in maps.values() for path in mapping.values()),
    }

    unreadable: list[str] = []
    format_mismatches: list[str] = []
    a_hash_mismatches: list[str] = []
    identities: Counter[str] = Counter()
    source_hashes: dict[str, str] = {}
    output_hashes: dict[str, dict[str, str]] = {"A": {}, "B": {}}

    if checks["recursive_image_lists_equal"]:
        for relative in relative_lists["source"]:
            source_path = maps["source"][relative]
            source_pixels = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
            if source_pixels is None or source_pixels.size == 0:
                unreadable.append(f"source:{relative}")
                continue
            with Image.open(source_path) as image:
                source_spec = (image.format, image.mode, image.size, str(source_pixels.dtype), source_pixels.shape)
            identities[infer_identity(Path(relative))] += 1
            source_hash = sha256_file(source_path)
            source_hashes[relative] = source_hash
            for group in ("A", "B"):
                path = maps[group][relative]
                pixels = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if pixels is None or pixels.size == 0:
                    unreadable.append(f"{group}:{relative}")
                    continue
                with Image.open(path) as image:
                    spec = (image.format, image.mode, image.size, str(pixels.dtype), pixels.shape)
                if spec != source_spec:
                    format_mismatches.append(f"{group}:{relative}")
                digest = sha256_file(path)
                output_hashes[group][relative] = digest
                if group == "A" and digest != source_hash:
                    a_hash_mismatches.append(relative)

    checks["all_images_decodable"] = not unreadable
    checks["format_mode_size_dtype_channels_preserved"] = not format_mismatches
    checks["group_a_byte_identical_to_source"] = not a_hash_mismatches

    manifests = {
        group: read_manifest(roots[group] / "manifest.jsonl") for group in ("A", "B")
    }
    manifest_errors: list[str] = []
    for group in ("A", "B"):
        if list(manifests[group]) != relative_lists[group]:
            manifest_errors.append(f"{group}:relative paths differ")
            continue
        for relative, record in manifests[group].items():
            if record["group"] != group:
                manifest_errors.append(f"{group}:{relative}:wrong group")
            if record["identity"] != infer_identity(Path(relative)):
                manifest_errors.append(f"{group}:{relative}:wrong identity")
            if record["source_sha256"] != source_hashes.get(relative):
                manifest_errors.append(f"{group}:{relative}:wrong source hash")
            if record["output_sha256"] != output_hashes[group].get(relative):
                manifest_errors.append(f"{group}:{relative}:wrong output hash")
            changed = record["source_sha256"] != record["output_sha256"]
            if bool(record["changed"]) != changed:
                manifest_errors.append(f"{group}:{relative}:wrong changed flag")
    checks["manifests_complete_and_current"] = not manifest_errors

    b_operator_counts = Counter(
        "+".join(record["operators"]) or "none" for record in manifests["B"].values()
    )
    b_unchanged = sum(not record["changed"] for record in manifests["B"].values())
    image_count = len(relative_lists["source"])
    retain_ratio = b_unchanged / image_count if image_count else 0.0
    b_config = json.loads((roots["B"] / "config_resolved.json").read_text(encoding="utf-8"))
    if b_config.get("retain_per_identity") is not None:
        expected_unchanged = int(b_config["retain_per_identity"]) * len(identities)
    else:
        expected_unchanged = int(round(float(b_config["retain_probability"]) * image_count))
    checks["group_b_retain_ratio_matches_config"] = b_unchanged == expected_unchanged
    checks["identity_counts_match"] = all(
        Counter(infer_identity(Path(relative)) for relative in relative_lists[name]) == identities
        for name in ("A", "B")
    )

    result = {
        "source": str(roots["source"]),
        "group_a": str(roots["A"]),
        "group_b": str(roots["B"]),
        "image_count": image_count,
        "identity_count": len(identities),
        "identity_sample_counts": dict(sorted(identities.items())),
        "image_bytes": {name: sum(path.stat().st_size for path in mapping.values()) for name, mapping in maps.items()},
        "group_b_unchanged_count": b_unchanged,
        "group_b_expected_unchanged_count": expected_unchanged,
        "group_b_actual_retain_ratio": retain_ratio,
        "group_b_operator_counts": dict(sorted(b_operator_counts.items())),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "errors": {
            "unreadable": unreadable,
            "format_mismatches": format_mismatches,
            "group_a_hash_mismatches": a_hash_mismatches,
            "manifest_errors": manifest_errors,
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--group-a", type=Path, required=True)
    parser.add_argument("--group-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.source, args.group_a, args.group_b)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
