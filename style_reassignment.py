#!/usr/bin/env python3
"""Prepare and audit deterministic PCE latent reassignment experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw


EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SAMPLE_RE = re.compile(r"^(?P<identity>c[1-4]_\d+)_sample(?P<sample>\d+)\.[^.]+$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vector_sha256(vector: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(vector, dtype="<f4").tobytes(order="C")).hexdigest()


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def scan_explicit_identities(root: Path, samples_per_identity: int = 7) -> dict[str, list[Path]]:
    if not root.is_dir():
        raise ValueError(f"Input root does not exist: {root}")
    grouped: dict[str, list[Path]] = {}
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        paths = sorted(path for path in directory.iterdir()
                       if path.is_file() and path.suffix.lower() in EXTENSIONS)
        if len(paths) != samples_per_identity:
            raise ValueError(f"{directory.name}: expected {samples_per_identity} images, found {len(paths)}")
        indices = []
        for path in paths:
            match = SAMPLE_RE.match(path.name)
            if not match or match.group("identity") != directory.name:
                raise ValueError(f"Identity filename mismatch: {path}")
            indices.append(int(match.group("sample")))
        if sorted(indices) != list(range(samples_per_identity)):
            raise ValueError(f"{directory.name}: sample indices are not 0..{samples_per_identity - 1}")
        grouped[directory.name] = sorted(paths, key=lambda path: int(SAMPLE_RE.match(path.name).group("sample")))
    if not grouped:
        raise ValueError(f"No explicit identity directories under {root}")
    return grouped


def sample_latent_pool(identity_count: int, samples_per_identity: int, dimension: int,
                       seed: int, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Match the legacy get_z_random2 RandomState sampling formula identity by identity."""
    rng = np.random.RandomState(seed)
    centers, vectors = [], []
    for _ in range(identity_count):
        center = rng.normal(loc=0.0, scale=1.0, size=(1, dimension))
        center /= np.linalg.norm(center)
        directions = rng.normal(loc=0.0, scale=1.0, size=(samples_per_identity, dimension))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        samples = center + directions * radius
        centers.append(center[0].astype(np.float32))
        vectors.extend(samples.astype(np.float32))
    return np.stack(centers), np.stack(vectors)


def assignment_row(*, assignment: str, input_relative: str, output_relative: str,
                   target_identity: str, target_sample_index: int, latent_id: int,
                   source_center_id: str, source_sample_index: int,
                   original_identity: str, original_sample_index: int,
                   vector: np.ndarray) -> dict:
    return {
        "assignment": assignment,
        "input_relative": input_relative,
        "output_relative": output_relative,
        "target_identity": target_identity,
        "target_sample_index": target_sample_index,
        "latent_id": latent_id,
        "source_center_id": source_center_id,
        "source_sample_index": source_sample_index,
        "original_identity": original_identity,
        "original_sample_index": original_sample_index,
        "z_sha256": vector_sha256(vector),
    }


def create_assignments(grouped: dict[str, list[Path]], input_root: Path, vectors: np.ndarray,
                       reassignment_seed: int) -> tuple[list[dict], list[dict], dict]:
    identities = list(grouped)
    samples = len(next(iter(grouped.values())))
    index_by_identity = {identity: index for index, identity in enumerate(identities)}
    a0 = []
    for identity in identities:
        center_index = index_by_identity[identity]
        for sample_index, path in enumerate(grouped[identity]):
            latent_id = center_index * samples + sample_index
            relative = path.relative_to(input_root).as_posix()
            a0.append(assignment_row(
                assignment="A0", input_relative=relative, output_relative=relative,
                target_identity=identity, target_sample_index=sample_index, latent_id=latent_id,
                source_center_id=identity, source_sample_index=sample_index,
                original_identity=identity, original_sample_index=sample_index,
                vector=vectors[latent_id],
            ))

    rng = np.random.default_rng(reassignment_seed)
    target_order = [identities[index] for index in rng.permutation(len(identities))]
    source_order = [identities[index] for index in rng.permutation(len(identities))]
    offsets = [int(value) for value in rng.choice(len(identities), size=samples, replace=False)]
    source_sample_permutations = {
        identity: [int(value) for value in rng.permutation(samples)] for identity in identities
    }
    target_sample_permutations = {
        identity: [int(value) for value in rng.permutation(samples)] for identity in identities
    }
    c1 = []
    for target_position, target_identity in enumerate(target_order):
        for slot in range(samples):
            source_identity = source_order[(target_position + offsets[slot]) % len(identities)]
            source_sample = source_sample_permutations[source_identity][slot]
            target_sample = target_sample_permutations[target_identity][slot]
            latent_id = index_by_identity[source_identity] * samples + source_sample
            input_path = grouped[target_identity][target_sample]
            original_path = grouped[source_identity][source_sample]
            c1.append(assignment_row(
                assignment="C1", input_relative=input_path.relative_to(input_root).as_posix(),
                output_relative=input_path.relative_to(input_root).as_posix(),
                target_identity=target_identity, target_sample_index=target_sample, latent_id=latent_id,
                source_center_id=source_identity, source_sample_index=source_sample,
                original_identity=source_identity, original_sample_index=source_sample,
                vector=vectors[latent_id],
            ))
            c1[-1]["original_input_relative"] = original_path.relative_to(input_root).as_posix()
    c1.sort(key=lambda row: row["input_relative"])
    return a0, c1, {
        "target_identity_order": target_order,
        "source_center_order": source_order,
        "cyclic_offsets": offsets,
        "source_sample_permutations": source_sample_permutations,
        "target_sample_permutations": target_sample_permutations,
    }


def audit_assignments(a0: list[dict], c1: list[dict], vector_count: int) -> dict:
    result = {}
    for name, rows in (("A0", a0), ("C1", c1)):
        latent_ids = [int(row["latent_id"]) for row in rows]
        by_target: dict[str, list[dict]] = defaultdict(list)
        by_source: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_target[row["target_identity"]].append(row)
            by_source[row["source_center_id"]].append(row)
        result[name] = {
            "rows": len(rows),
            "unique_latent_ids": len(set(latent_ids)),
            "latent_ids_exactly_once": sorted(latent_ids) == list(range(vector_count)),
            "target_identities": len(by_target),
            "target_sample_counts": sorted(set(len(items) for items in by_target.values())),
            "source_center_counts": sorted(set(len(items) for items in by_source.values())),
            "distinct_source_centers_per_target": sorted(set(
                len({item["source_center_id"] for item in items}) for items in by_target.values()
            )),
            "distinct_targets_per_source_center": sorted(set(
                len({item["target_identity"] for item in items}) for items in by_source.values()
            )),
            "unique_input_paths": len({row["input_relative"] for row in rows}),
            "unique_output_paths": len({row["output_relative"] for row in rows}),
        }
    result["constraints_passed"] = (
        result["A0"]["latent_ids_exactly_once"]
        and result["C1"]["latent_ids_exactly_once"]
        and result["A0"]["target_sample_counts"] == [7]
        and result["C1"]["target_sample_counts"] == [7]
        and result["C1"]["distinct_source_centers_per_target"] == [7]
        and result["C1"]["distinct_targets_per_source_center"] == [7]
        and result["C1"]["unique_input_paths"] == vector_count
        and result["C1"]["unique_output_paths"] == vector_count
    )
    return result


def select_preview_identities(identities: list[str]) -> list[str]:
    by_case = defaultdict(list)
    for identity in identities:
        by_case[identity.split("_")[0]].append(identity)
    selected = []
    for case, count in (("c1", 13), ("c2", 13), ("c3", 12), ("c4", 12)):
        values = by_case[case]
        selected.extend(values[index] for index in np.linspace(0, len(values) - 1, count, dtype=int))
    return selected


def prepare(args: argparse.Namespace) -> None:
    input_root, output = args.input.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    grouped = scan_explicit_identities(input_root, args.samples_per_identity)
    identities = list(grouped)
    centers, vectors = sample_latent_pool(len(identities), args.samples_per_identity,
                                           args.dimension, args.latent_seed, args.radius)
    np.save(output / "centers.npy", centers, allow_pickle=False)
    np.save(output / "latent_pool.npy", vectors, allow_pickle=False)
    a0, c1, reversible = create_assignments(grouped, input_root, vectors, args.reassignment_seed)
    write_jsonl(output / "assignments_A0.jsonl", a0)
    write_jsonl(output / "assignments_C1.jsonl", c1)

    latent_rows = []
    for row in a0:
        latent_rows.append({
            "latent_id": row["latent_id"], "source_center_id": row["source_center_id"],
            "source_sample_index": row["source_sample_index"],
            "original_identity": row["original_identity"],
            "original_sample_index": row["original_sample_index"],
            "original_input_relative": row["input_relative"], "dtype": "float32",
            "dimension": args.dimension, "z": vectors[row["latent_id"]].tolist(),
            "z_sha256": row["z_sha256"],
            "A0_target_relative": row["output_relative"],
        })
    c1_by_latent = {row["latent_id"]: row for row in c1}
    for row in latent_rows:
        mapped = c1_by_latent[row["latent_id"]]
        row["C1_target_relative"] = mapped["output_relative"]
        row["C1_target_identity"] = mapped["target_identity"]
        row["C1_target_sample_index"] = mapped["target_sample_index"]
    write_jsonl(output / "latent_manifest.jsonl", latent_rows)
    (output / "reversible_mapping_state.json").write_text(json.dumps(reversible, indent=2), encoding="utf-8")

    preview_identities = select_preview_identities(identities)
    preview_set = set(preview_identities)
    write_jsonl(output / "preview_assignments_A0.jsonl",
                [row for row in a0 if row["target_identity"] in preview_set])
    write_jsonl(output / "preview_assignments_C1.jsonl",
                [row for row in c1 if row["target_identity"] in preview_set])
    (output / "preview_identities.txt").write_text("\n".join(preview_identities) + "\n", encoding="utf-8")

    fixed_identity = preview_identities[0]
    fixed_input = grouped[fixed_identity][0].relative_to(input_root).as_posix()
    fixed_pattern_rows = []
    for index, row in enumerate([item for item in c1 if item["target_identity"] == fixed_identity]):
        item = dict(row)
        item.update({"assignment": "fixed_pattern", "input_relative": fixed_input,
                     "output_relative": f"fixed_pattern/style_{index:02d}_{row['source_center_id']}.png"})
        fixed_pattern_rows.append(item)
    fixed_style_rows = []
    chosen = []
    for case in ("c1", "c2", "c3", "c4"):
        values = [identity for identity in preview_identities if identity.startswith(case + "_")]
        chosen.extend([values[0], values[-1]])
    for index, identity in enumerate(chosen):
        path = grouped[identity][0].relative_to(input_root).as_posix()
        item = dict(a0[0])
        item.update({"assignment": "fixed_style", "input_relative": path,
                     "output_relative": f"fixed_style/pattern_{index:02d}_{identity}.png",
                     "target_identity": identity, "target_sample_index": 0})
        fixed_style_rows.append(item)
    write_jsonl(output / "diagnostic_fixed_pattern.jsonl", fixed_pattern_rows)
    write_jsonl(output / "diagnostic_fixed_style.jsonl", fixed_style_rows)

    audit = audit_assignments(a0, c1, len(vectors))
    audit.update({
        "input_root": str(input_root), "identities": len(identities),
        "samples_per_identity": args.samples_per_identity,
        "dimension": args.dimension, "radius": args.radius,
        "latent_seed": args.latent_seed, "reassignment_seed": args.reassignment_seed,
        "centers_sha256": sha256_file(output / "centers.npy"),
        "latent_pool_sha256": sha256_file(output / "latent_pool.npy"),
        "latent_manifest_sha256": sha256_file(output / "latent_manifest.jsonl"),
        "assignments_A0_sha256": sha256_file(output / "assignments_A0.jsonl"),
        "assignments_C1_sha256": sha256_file(output / "assignments_C1.jsonl"),
        "preview_identity_counts": dict(Counter(identity.split("_")[0] for identity in preview_identities)),
    })
    (output / "latent_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    config = {
        "input_root": str(input_root), "output": str(output), "dimension": args.dimension,
        "radius": args.radius, "samples_per_identity": args.samples_per_identity,
        "latent_seed": args.latent_seed, "reassignment_seed": args.reassignment_seed,
        "formula": "normalize(N(0,1) center) + 0.5 * normalize(N(0,1) direction)",
        "rng": {"latent": "numpy RandomState (MT19937; legacy compatibility)",
                "reassignment": "numpy default_rng (PCG64; independent stream)"},
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if not audit["constraints_passed"]:
        raise SystemExit(2)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def audit_generated(args: argparse.Namespace) -> None:
    input_root, a0_root, c1_root = args.input.resolve(), args.a0.resolve(), args.c1.resolve()
    latent_dir, output = args.latent_dir.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    pool = np.load(latent_dir / "latent_pool.npy", allow_pickle=False)
    assignments = {name: read_jsonl(latent_dir / f"assignments_{name}.jsonl") for name in ("A0", "C1")}
    result = {"groups": {}, "latent_pool_sha256": sha256_file(latent_dir / "latent_pool.npy")}
    input_paths = {path.relative_to(input_root).as_posix(): path for paths in scan_explicit_identities(input_root).values() for path in paths}
    for name, root in (("A0", a0_root), ("C1", c1_root)):
        rows = assignments[name]
        expected = {row["output_relative"] for row in rows}
        actual_paths = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in EXTENSIONS)
        actual = {path.relative_to(root).as_posix() for path in actual_paths}
        manifest_path = root / "generation_records.jsonl"
        records = read_jsonl(manifest_path)
        record_map = {row["output_relative"]: row for row in records}
        decode_errors, spec_errors, hash_errors, empty_files = [], [], [], []
        mode_counts, size_counts, format_counts = Counter(), Counter(), Counter()
        per_identity = Counter(Path(relative).parts[0] for relative in actual)
        for relative in sorted(actual):
            path = root / relative
            if path.stat().st_size == 0:
                empty_files.append(relative)
            try:
                with Image.open(path) as image:
                    mode_counts[image.mode] += 1
                    size_counts[f"{image.size[0]}x{image.size[1]}"] += 1
                    format_counts[str(image.format)] += 1
                    if image.size != (128, 128) or image.mode != "RGB" or image.format != "PNG":
                        spec_errors.append({"path": relative, "size": image.size, "mode": image.mode})
            except Exception as exc:
                decode_errors.append({"path": relative, "error": str(exc)})
            record = record_map.get(relative)
            if not record or record.get("output_sha256") != sha256_file(path):
                hash_errors.append(relative)
        result["groups"][name] = {
            "root": str(root), "images": len(actual), "expected_images": len(expected),
            "identities": len(per_identity), "samples_per_identity_values": sorted(set(per_identity.values())),
            "mode_counts": dict(mode_counts), "size_counts": dict(size_counts),
            "format_counts": dict(format_counts), "empty_files": empty_files,
            "relative_paths_match": actual == expected == set(input_paths),
            "generation_records": len(records), "records_complete": set(record_map) == expected,
            "record_paths_unique": len(record_map) == len(records),
            "input_output_relative_mapping_preserved": all(
                row["input_relative"] == row["output_relative"] for row in rows
            ),
            "latent_ids_unique_and_complete": len({int(row["latent_id"]) for row in rows}) == len(pool),
            "decode_errors": decode_errors, "spec_errors": spec_errors, "hash_errors": hash_errors,
        }
    result["a0_c1_relative_paths_match"] = (
        {path.relative_to(a0_root).as_posix() for path in a0_root.rglob("*.png")}
        == {path.relative_to(c1_root).as_posix() for path in c1_root.rglob("*.png")}
    )
    result["a0_c1_specs_match"] = all(
        result["groups"]["A0"][key] == result["groups"]["C1"][key]
        for key in ("mode_counts", "size_counts", "format_counts", "samples_per_identity_values")
    )
    result["all_checks_passed"] = result["a0_c1_relative_paths_match"] and result["a0_c1_specs_match"] and all(
        group["relative_paths_match"] and group["records_complete"] and not group["decode_errors"]
        and group["record_paths_unique"] and group["input_output_relative_mapping_preserved"]
        and group["latent_ids_unique_and_complete"] and not group["empty_files"]
        and not group["spec_errors"] and not group["hash_errors"] for group in result["groups"].values()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--input", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--latent-seed", type=int, default=2026)
    prepare_parser.add_argument("--reassignment-seed", type=int, default=2027)
    prepare_parser.add_argument("--dimension", type=int, default=8)
    prepare_parser.add_argument("--radius", type=float, default=0.5)
    prepare_parser.add_argument("--samples-per-identity", type=int, default=7)
    prepare_parser.set_defaults(func=prepare)
    audit_parser = subparsers.add_parser("audit-generated")
    audit_parser.add_argument("--input", type=Path, required=True)
    audit_parser.add_argument("--a0", type=Path, required=True)
    audit_parser.add_argument("--c1", type=Path, required=True)
    audit_parser.add_argument("--latent-dir", type=Path, required=True)
    audit_parser.add_argument("--output", type=Path, required=True)
    audit_parser.set_defaults(func=audit_generated)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
