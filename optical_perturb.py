#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PVTree-course contributors.
# Course-project addition; see LICENSE for terms and README for upstream attribution.
"""Offline optical perturbations for PVTree / PCE-Palm grayscale outputs.

The operators approximate acquisition-condition changes. They are deliberately
lightweight and are not a physically validated skin-scattering model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import time
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
OPERATOR_ORDER = ("illumination", "blur", "noise")

PRESETS: dict[str, dict[str, Any]] = {
    "none": {
        "enabled_operators": [],
        "retain_probability": 1.0,
        "combination_probability": 0.0,
        "operator_weights": {"illumination": 1.0, "blur": 1.0, "noise": 1.0},
        "illumination": {
            "amplitude_range": [0.0, 0.0],
            "global_bias_range": [0.0, 0.0],
            "gain_clip": [1.0, 1.0],
            "control_grid": 5,
        },
        "blur": {
            "sigma_fraction_range": [0.0, 0.0],
            "sigma_clip_px": [0.0, 0.0],
        },
        "noise": {"std_255_range": [0.0, 0.0]},
    },
    "mild": {
        "enabled_operators": list(OPERATOR_ORDER),
        "retain_probability": 0.30,
        "combination_probability": 0.15,
        "operator_weights": {"illumination": 0.45, "blur": 0.30, "noise": 0.25},
        "illumination": {
            "amplitude_range": [0.05, 0.12],
            "global_bias_range": [-0.025, 0.025],
            "gain_clip": [0.86, 1.14],
            "control_grid": 5,
        },
        "blur": {
            "sigma_fraction_range": [0.002, 0.0045],
            "sigma_clip_px": [0.35, 1.20],
        },
        "noise": {"std_255_range": [1.0, 3.0]},
    },
    "moderate": {
        "enabled_operators": list(OPERATOR_ORDER),
        "retain_probability": 0.20,
        "combination_probability": 0.25,
        "operator_weights": {"illumination": 0.45, "blur": 0.30, "noise": 0.25},
        "illumination": {
            "amplitude_range": [0.10, 0.20],
            "global_bias_range": [-0.04, 0.04],
            "gain_clip": [0.76, 1.24],
            "control_grid": 5,
        },
        "blur": {
            "sigma_fraction_range": [0.0035, 0.007],
            "sigma_clip_px": [0.55, 1.80],
        },
        "noise": {"std_255_range": [2.5, 6.0]},
    },
}


def _deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def resolve_config(preset: str = "mild", config_path: Path | None = None) -> dict[str, Any]:
    user_config: dict[str, Any] = {}
    if config_path is not None:
        with config_path.open("r", encoding="utf-8") as handle:
            user_config = json.load(handle)
        preset = user_config.get("preset", preset)
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}")
    config = _deep_update(PRESETS[preset], {k: v for k, v in user_config.items() if k != "preset"})
    config["preset"] = preset
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    enabled = config["enabled_operators"]
    unknown = set(enabled) - set(OPERATOR_ORDER)
    if unknown:
        raise ValueError(f"Unknown operators: {sorted(unknown)}")
    if len(enabled) != len(set(enabled)):
        raise ValueError("enabled_operators contains duplicates")
    for name in ("retain_probability", "combination_probability"):
        value = float(config[name])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    for operator in enabled:
        if float(config["operator_weights"].get(operator, 0.0)) < 0.0:
            raise ValueError(f"Negative operator weight: {operator}")
    if config.get("retain_per_identity") is not None and int(config["retain_per_identity"]) < 0:
        raise ValueError("retain_per_identity must be non-negative")


def stable_seed(global_seed: int, relative_path: Path, salt: str = "") -> int:
    payload = f"{global_seed}|{relative_path.as_posix()}|{salt}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def list_images(input_dir: Path) -> list[Path]:
    return sorted(
        (path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.relative_to(input_dir).as_posix().casefold(),
    )


def infer_identity(relative_path: Path) -> str:
    if relative_path.parent != Path("."):
        return relative_path.parent.as_posix()
    match = re.match(r"(.+?)_sample\d+$", relative_path.stem, flags=re.IGNORECASE)
    return match.group(1) if match else relative_path.stem


def load_uint8(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {image.dtype}: {path}")
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] in (3, 4)):
        return image
    raise ValueError(f"Unsupported image shape {image.shape}: {path}")


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Unsupported image shape {image.shape}")


def load_grayscale_uint8(path: Path) -> np.ndarray:
    return to_grayscale(load_uint8(path))


def restore_channel_layout(gray: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if reference.ndim == 2:
        return gray
    color = np.repeat(gray[:, :, np.newaxis], 3, axis=2)
    if reference.shape[2] == 3:
        return color
    return np.dstack((color, reference[:, :, 3]))


def image_metrics(image: np.ndarray) -> dict[str, float]:
    values = image.astype(np.float32)
    low = float(np.mean(image <= 5))
    high = float(np.mean(image >= 250))
    mean = float(np.mean(values))
    std = float(np.std(values))
    return {
        "brightness_mean": mean,
        "intensity_std": std,
        "contrast_rms": std / max(mean, 1e-8),
        "contrast_std": std,
        "saturated_low_ratio": low,
        "saturated_high_ratio": high,
        "saturated_total_ratio": low + high,
        "sharpness_laplacian_var": float(cv2.Laplacian(values, cv2.CV_32F).var()),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_operators(rng: np.random.Generator, config: dict[str, Any]) -> list[str]:
    enabled = [name for name in OPERATOR_ORDER if name in config["enabled_operators"]]
    if not enabled or rng.random() < float(config["retain_probability"]):
        return []
    count = 2 if len(enabled) >= 2 and rng.random() < float(config["combination_probability"]) else 1
    weights = np.asarray([float(config["operator_weights"].get(name, 1.0)) for name in enabled], dtype=np.float64)
    if float(weights.sum()) <= 0.0:
        raise ValueError("At least one enabled operator must have a positive weight")
    weights /= weights.sum()
    count = min(count, int(np.count_nonzero(weights)))
    chosen = rng.choice(enabled, size=count, replace=False, p=weights).tolist()
    return [name for name in OPERATOR_ORDER if name in chosen]


def apply_illumination(
    image: np.ndarray, rng: np.random.Generator, config: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = image.shape
    grid_size = max(3, int(config["control_grid"]))
    coarse = rng.normal(0.0, 1.0, size=(grid_size, grid_size)).astype(np.float32)
    field = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    sigma = max(height, width) * 0.12
    field = cv2.GaussianBlur(field, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
    field -= float(field.mean())
    max_abs = float(np.max(np.abs(field)))
    if max_abs > 1e-8:
        field /= max_abs
    amplitude = float(rng.uniform(*config["amplitude_range"]))
    bias = float(rng.uniform(*config["global_bias_range"]))
    gain = np.clip(1.0 + bias + amplitude * field, *config["gain_clip"])
    output = image * gain
    return output, {
        "amplitude": amplitude,
        "global_bias": bias,
        "gain_min": float(gain.min()),
        "gain_max": float(gain.max()),
        "control_grid": grid_size,
    }


def apply_blur(
    image: np.ndarray, rng: np.random.Generator, config: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    fraction = float(rng.uniform(*config["sigma_fraction_range"]))
    sigma = float(np.clip(fraction * min(image.shape), *config["sigma_clip_px"]))
    if sigma <= 0.0:
        return image.copy(), {"sigma_fraction": fraction, "sigma_px": sigma}
    output = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
    return output, {"sigma_fraction": fraction, "sigma_px": sigma}


def apply_noise(
    image: np.ndarray, rng: np.random.Generator, config: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    std_255 = float(rng.uniform(*config["std_255_range"]))
    noise = rng.normal(0.0, std_255 / 255.0, size=image.shape).astype(np.float32)
    return image + noise, {"std_255": std_255}


def apply_operators(
    image_uint8: np.ndarray,
    operators: Iterable[str],
    rng: np.random.Generator,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, dict[str, Any]]]:
    image = image_uint8.astype(np.float32) / 255.0
    parameters: dict[str, dict[str, Any]] = {}
    for operator in operators:
        if operator == "illumination":
            image, parameters[operator] = apply_illumination(image, rng, config[operator])
        elif operator == "blur":
            image, parameters[operator] = apply_blur(image, rng, config[operator])
        elif operator == "noise":
            image, parameters[operator] = apply_noise(image, rng, config[operator])
        else:
            raise ValueError(f"Unknown operator: {operator}")
        image = np.clip(image, 0.0, 1.0)
    return np.rint(image * 255.0).astype(np.uint8), parameters


def _aggregate_metrics(records: list[dict[str, Any]], key: str) -> dict[str, float]:
    if not records:
        return {}
    metric_names = records[0][key].keys()
    return {name: float(np.mean([record[key][name] for record in records])) for name in metric_names}


def _operator_metric_summary(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped["+".join(record["operators"]) or "none"].append(record)
    summary = {}
    for name, items in sorted(grouped.items()):
        before = _aggregate_metrics(items, "before_metrics")
        after = _aggregate_metrics(items, "after_metrics")
        summary[name] = {
            "count": len(items),
            "before": before,
            "after": after,
            "delta": {key: after[key] - before[key] for key in before},
        }
    return summary


def write_records(output_dir: Path, records: list[dict[str, Any]]) -> None:
    with (output_dir / "augmentation_records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    manifest_fields = (
        "identity",
        "relative_path",
        "source_path",
        "group",
        "source_sha256",
        "output_sha256",
        "changed",
        "operators",
        "parameters",
        "global_seed",
        "file_seed",
    )
    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            manifest = {key: record[key] for key in manifest_fields}
            handle.write(json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n")

    metric_names = list(records[0]["before_metrics"].keys()) if records else []
    fieldnames = ["input_relative", "output_relative", "identity", "operators", "file_seed"]
    fieldnames += [f"before_{name}" for name in metric_names]
    fieldnames += [f"after_{name}" for name in metric_names]
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row: dict[str, Any] = {
                "input_relative": record["input_relative"],
                "output_relative": record["output_relative"],
                "identity": record["identity"],
                "operators": "+".join(record["operators"]) or "none",
                "file_seed": record["file_seed"],
            }
            row.update({f"before_{key}": value for key, value in record["before_metrics"].items()})
            row.update({f"after_{key}": value for key, value in record["after_metrics"].items()})
            writer.writerow(row)


def diagnostic_flags(before: dict[str, float], after: dict[str, float]) -> list[str]:
    flags = []
    if after["saturated_low_ratio"] > before["saturated_low_ratio"] + 0.02:
        flags.append("Low-end saturation increased by more than 2 percentage points; inspect for underexposure.")
    if after["saturated_high_ratio"] > before["saturated_high_ratio"] + 0.02:
        flags.append("High-end saturation increased by more than 2 percentage points; inspect for overexposure.")
    if before["sharpness_laplacian_var"] > 0:
        sharpness_ratio = after["sharpness_laplacian_var"] / before["sharpness_laplacian_var"]
        if sharpness_ratio < 0.5:
            flags.append("Mean Laplacian sharpness fell by more than 50%; inspect whether fine veins were erased.")
    return flags


def write_markdown_report(output_dir: Path, report: dict[str, Any]) -> None:
    labels = {
        "brightness_mean": "Brightness mean",
        "intensity_std": "Intensity standard deviation",
        "contrast_rms": "RMS contrast (std / mean)",
        "contrast_std": "Contrast std",
        "saturated_low_ratio": "Low saturation ratio",
        "saturated_high_ratio": "High saturation ratio",
        "saturated_total_ratio": "Total saturation ratio",
        "sharpness_laplacian_var": "Laplacian sharpness variance",
    }
    lines = [
        "# Optical perturbation report",
        "",
        f"- Images: {report['image_count']}",
        f"- Identities: {report['identity_count']}",
        f"- Seed: {report['seed']}",
        f"- Byte-identical retained files: {report['unchanged_file_count']} ({report['actual_retain_ratio']:.2%})",
        f"- Elapsed: {report['elapsed_seconds']:.3f} s ({report['images_per_second']:.2f} images/s)",
        "",
        "## Operator distribution",
        "",
    ]
    for name, count in report["operator_counts"].items():
        lines.append(f"- `{name}`: {count}")
    lines.extend(["", "## Aggregate image diagnostics", "", "| Metric | Before | After | Delta |", "|---|---:|---:|---:|"])
    for key, label in labels.items():
        lines.append(
            f"| {label} | {report['aggregate_before'][key]:.6f} | "
            f"{report['aggregate_after'][key]:.6f} | {report['aggregate_delta'][key]:+.6f} |"
        )
    lines.extend(["", "## Diagnostic flags", ""])
    if report["diagnostic_flags"]:
        lines.extend(f"- {flag}" for flag in report["diagnostic_flags"])
    else:
        lines.append("- No automatic clipping or excessive-blur threshold was triggered.")
    lines.extend(
        [
            "",
            "> These diagnostics detect overly strong perturbations only. They are not evidence of recognition improvement.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def save_visualizations(
    input_dir: Path,
    output_dir: Path,
    source_paths: list[Path],
    records: list[dict[str, Any]],
    config: dict[str, Any],
    seed: int,
    visual_dir: Path,
) -> list[str]:
    if not source_paths:
        return []
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    visual_dir.mkdir(parents=True, exist_ok=True)
    source = source_paths[0]
    relative = source.relative_to(input_dir)
    original = load_grayscale_uint8(source)
    variants: list[tuple[str, np.ndarray]] = [("original", original)]
    for label, operators in (
        ("illumination", ["illumination"]),
        ("blur", ["blur"]),
        ("noise", ["noise"]),
        ("combined", ["illumination", "blur"]),
    ):
        rng = np.random.default_rng(stable_seed(seed, relative, f"preview-{label}"))
        image, _ = apply_operators(original, operators, rng, config)
        variants.append((label, image))

    figure, axes = plt.subplots(1, len(variants), figsize=(15, 3.2), constrained_layout=True)
    for axis, (label, image) in zip(axes, variants):
        axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        axis.set_title(label)
        axis.axis("off")
    operator_path = visual_dir / "operator_comparison.png"
    figure.savefig(operator_path, dpi=160)
    plt.close(figure)

    by_identity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_identity[record["identity"]].append(record)
    selected_identities = sorted(by_identity)[:4]
    samples_per_identity = min(3, max(len(by_identity[name]) for name in selected_identities))
    figure, axes = plt.subplots(
        len(selected_identities),
        samples_per_identity,
        figsize=(3 * samples_per_identity, 3 * len(selected_identities)),
        squeeze=False,
        constrained_layout=True,
    )
    for row, identity in enumerate(selected_identities):
        identity_records = sorted(by_identity[identity], key=lambda item: item["output_relative"])
        for column in range(samples_per_identity):
            axis = axes[row, column]
            if column < len(identity_records):
                record = identity_records[column]
                original_image = load_grayscale_uint8(input_dir / record["input_relative"])
                enhanced_image = load_grayscale_uint8(output_dir / record["output_relative"])
                separator = np.full((original_image.shape[0], 3), 255, dtype=np.uint8)
                pair = np.hstack((original_image, separator, enhanced_image))
                axis.imshow(pair, cmap="gray", vmin=0, vmax=255)
                operators = "+".join(record["operators"]) or "none"
                axis.set_title(f"{identity} · original | {operators}", fontsize=9)
            axis.axis("off")
    identity_path = visual_dir / "identity_samples.png"
    figure.savefig(identity_path, dpi=160)
    plt.close(figure)

    detail_size = max(24, min(original.shape) // 2)
    response = np.abs(cv2.Laplacian(original.astype(np.float32), cv2.CV_32F))
    best_score, best_x, best_y = -1.0, 0, 0
    stride = max(4, detail_size // 8)
    for y in range(0, original.shape[0] - detail_size + 1, stride):
        for x in range(0, original.shape[1] - detail_size + 1, stride):
            score = float(response[y : y + detail_size, x : x + detail_size].mean())
            if score > best_score:
                best_score, best_x, best_y = score, x, y
    figure, axes = plt.subplots(1, len(variants), figsize=(15, 3.2), constrained_layout=True)
    for axis, (label, image) in zip(axes, variants):
        crop = image[best_y : best_y + detail_size, best_x : best_x + detail_size]
        axis.imshow(crop, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axis.set_title(f"{label} detail")
        axis.axis("off")
    detail_path = visual_dir / "fine_vein_zoom.png"
    figure.savefig(detail_path, dpi=200)
    plt.close(figure)
    return [str(operator_path.resolve()), str(identity_path.resolve()), str(detail_path.resolve())]


def process_dataset(
    input_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
    seed: int = 2026,
    max_images: int | None = None,
    overwrite: bool = False,
    visualize: bool = True,
    group: str | None = None,
    visualization_dir: Path | None = None,
) -> dict[str, Any]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    if output_dir == input_dir or output_dir.is_relative_to(input_dir):
        raise ValueError("Output directory must be separate from and outside the input directory")
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = list_images(input_dir)
    if max_images is not None:
        paths = paths[:max_images]
    if not paths:
        raise ValueError(f"No supported images found in: {input_dir}")

    enabled = bool(config["enabled_operators"])
    retain_per_identity = config.get("retain_per_identity")
    if not enabled:
        retained_indices = set(range(len(paths)))
    elif retain_per_identity is not None:
        identity_indices: dict[str, list[int]] = defaultdict(list)
        for index, path in enumerate(paths):
            identity_indices[infer_identity(path.relative_to(input_dir))].append(index)
        retained_indices = set()
        for identity, indices in sorted(identity_indices.items()):
            count = int(retain_per_identity)
            if count > len(indices):
                raise ValueError(
                    f"retain_per_identity={count} exceeds {len(indices)} samples for identity {identity}"
                )
            ranking = sorted(
                indices,
                key=lambda index: stable_seed(seed, paths[index].relative_to(input_dir), "retain-rank"),
            )
            retained_indices.update(ranking[:count])
    else:
        retain_count = int(round(float(config["retain_probability"]) * len(paths)))
        retain_ranking = sorted(
            range(len(paths)),
            key=lambda index: stable_seed(seed, paths[index].relative_to(input_dir), "retain-rank"),
        )
        retained_indices = set(retain_ranking[:retain_count])

    start = time.perf_counter()
    records: list[dict[str, Any]] = []
    resolved_group = group or ("A" if config.get("preset") == "none" else "B")
    for index, source in enumerate(paths):
        relative = source.relative_to(input_dir)
        file_seed = stable_seed(seed, relative)
        rng = np.random.default_rng(file_seed)
        if index in retained_indices:
            operators = []
        else:
            augmented_config = _deep_update(config, {"retain_probability": 0.0})
            operators = choose_operators(rng, augmented_config)
        source_pixels = load_uint8(source)
        original = to_grayscale(source_pixels)
        result_gray, parameters = apply_operators(original, operators, rng, config)
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite: {destination}")
        if operators:
            result_pixels = restore_channel_layout(result_gray, source_pixels)
            if not cv2.imwrite(str(destination), result_pixels):
                raise OSError(f"Unable to write image: {destination}")
        else:
            shutil.copy2(source, destination)
            result_pixels = source_pixels
        source_sha256 = sha256_file(source)
        output_sha256 = sha256_file(destination)
        records.append(
            {
                "input_relative": relative.as_posix(),
                "output_relative": relative.as_posix(),
                "relative_path": relative.as_posix(),
                "source_path": str(source.resolve()),
                "identity": infer_identity(relative),
                "group": resolved_group,
                "operators": operators,
                "parameters": parameters,
                "global_seed": int(seed),
                "file_seed": int(file_seed),
                "source_sha256": source_sha256,
                "output_sha256": output_sha256,
                "changed": source_sha256 != output_sha256,
                "shape": list(source_pixels.shape),
                "dtype": str(source_pixels.dtype),
                "before_metrics": image_metrics(original),
                "after_metrics": image_metrics(to_grayscale(result_pixels)),
            }
        )

    write_records(output_dir, records)
    if visualization_dir is None:
        visualization_dir = output_dir.parent / f"{output_dir.name}_visualizations"
    visualizations = (
        save_visualizations(input_dir, output_dir, paths, records, config, seed, visualization_dir.resolve())
        if visualize
        else []
    )
    elapsed = time.perf_counter() - start
    before = _aggregate_metrics(records, "before_metrics")
    after = _aggregate_metrics(records, "after_metrics")
    unchanged_count = sum(not record["changed"] for record in records)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "seed": int(seed),
        "group": resolved_group,
        "image_count": len(records),
        "identity_count": len({record["identity"] for record in records}),
        "identity_sample_counts": dict(sorted(Counter(record["identity"] for record in records).items())),
        "operator_counts": dict(sorted(Counter("+".join(record["operators"]) or "none" for record in records).items())),
        "operator_metric_summary": _operator_metric_summary(records),
        "unchanged_file_count": unchanged_count,
        "actual_retain_ratio": unchanged_count / len(records),
        "config": config,
        "aggregate_before": before,
        "aggregate_after": after,
        "aggregate_delta": {key: after[key] - before[key] for key in before},
        "diagnostic_flags": diagnostic_flags(before, after),
        "elapsed_seconds": elapsed,
        "images_per_second": len(records) / max(elapsed, 1e-9),
        "visualizations": visualizations,
        "interpretation_note": (
            "These image statistics are diagnostic only and are not evidence of recognition improvement. "
            "The optical parameters are conservative starting points and have not been calibrated on real sensors."
        ),
    }
    with (output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
    with (output_dir / "config_resolved.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2, sort_keys=True)
    write_markdown_report(output_dir, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="PCE output image root")
    parser.add_argument("--output", type=Path, required=True, help="new output root; never overwritten by default")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="mild")
    parser.add_argument("--config", type=Path, help="JSON overrides; may include a preset key")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-images", type=int, help="process the first N images in stable path order")
    parser.add_argument("--operators", nargs="+", choices=OPERATOR_ORDER, help="override enabled operators")
    parser.add_argument("--group", choices=("A", "B"), help="dataset group recorded in manifest")
    parser.add_argument("--visualization-dir", type=Path, help="visual output directory; defaults beside dataset root")
    parser.add_argument("--overwrite", action="store_true", help="allow writing into an existing output directory")
    parser.add_argument("--skip-visualizations", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = resolve_config(args.preset, args.config)
    if args.operators is not None:
        config["enabled_operators"] = args.operators
        validate_config(config)
    report = process_dataset(
        args.input,
        args.output,
        config,
        seed=args.seed,
        max_images=args.max_images,
        overwrite=args.overwrite,
        visualize=not args.skip_visualizations,
        group=args.group,
        visualization_dir=args.visualization_dir,
    )
    print(
        f"Processed {report['image_count']} images / {report['identity_count']} identities "
        f"in {report['elapsed_seconds']:.2f}s -> {report['output_dir']}"
    )


if __name__ == "__main__":
    main()
