# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PVTree-course contributors.
# Course-project addition; see LICENSE for terms and README for upstream attribution.
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytest

from optical_perturb import infer_identity, process_dataset, resolve_config


def make_dataset(root: Path, identities: int = 2, samples: int = 3) -> list[Path]:
    paths = []
    y, x = np.mgrid[0:96, 0:128]
    for identity_index in range(identities):
        identity = f"c{identity_index + 1}_0"
        identity_dir = root / identity
        identity_dir.mkdir(parents=True)
        for sample_index in range(samples):
            base = 80.0 + x * 0.7 + y * 0.25
            vein = 35.0 * np.exp(-((y - (35 + sample_index * 6)) ** 2) / 10.0)
            image = np.clip(base - vein + identity_index * 5, 0, 255).astype(np.uint8)
            path = identity_dir / f"{identity}_sample{sample_index}.png"
            assert cv2.imwrite(str(path), image)
            paths.append(path)
    return paths


def read_pixels(root: Path) -> dict[str, np.ndarray]:
    return {
        path.relative_to(root).as_posix(): cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        for path in sorted(root.rglob("*.png"))
        if "visualizations" not in path.parts
    }


def test_none_mode_is_pixel_identical_and_preserves_labels(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    make_dataset(input_dir)

    report = process_dataset(input_dir, output_dir, resolve_config("none"), seed=17, visualize=True)

    before = read_pixels(input_dir)
    after = read_pixels(output_dir)
    assert before.keys() == after.keys()
    for relative_path in before:
        np.testing.assert_array_equal(before[relative_path], after[relative_path])
    assert report["image_count"] == 6
    assert report["identity_count"] == 2
    assert report["identity_sample_counts"] == {"c1_0": 3, "c2_0": 3}
    assert report["actual_retain_ratio"] == 1.0
    assert (output_dir / "report.md").is_file()
    visual_dir = output_dir.parent / "output_visualizations"
    assert (visual_dir / "operator_comparison.png").is_file()
    assert (visual_dir / "fine_vein_zoom.png").is_file()
    manifests = [json.loads(line) for line in (output_dir / "manifest.jsonl").read_text().splitlines()]
    assert all(item["group"] == "A" and not item["changed"] for item in manifests)
    assert all(item["source_sha256"] == item["output_sha256"] for item in manifests)
    assert len(list(output_dir.rglob("*.png"))) == 6
    records = [json.loads(line) for line in (output_dir / "augmentation_records.jsonl").read_text().splitlines()]
    assert all(record["operators"] == [] for record in records)


def test_fixed_seed_is_reproducible(tmp_path):
    input_dir = tmp_path / "input"
    make_dataset(input_dir)
    config = resolve_config("mild")
    config["retain_probability"] = 0.0
    config["combination_probability"] = 0.5

    output_a = tmp_path / "output_a"
    output_b = tmp_path / "output_b"
    process_dataset(input_dir, output_a, config, seed=2026, visualize=False)
    process_dataset(input_dir, output_b, config, seed=2026, visualize=False)

    pixels_a = read_pixels(output_a)
    pixels_b = read_pixels(output_b)
    assert pixels_a.keys() == pixels_b.keys()
    for relative_path in pixels_a:
        np.testing.assert_array_equal(pixels_a[relative_path], pixels_b[relative_path])
    assert (output_a / "augmentation_records.jsonl").read_text() == (
        output_b / "augmentation_records.jsonl"
    ).read_text()


def test_retain_per_identity_is_exact(tmp_path):
    input_dir = tmp_path / "input"
    make_dataset(input_dir, identities=2, samples=7)
    config = resolve_config("mild")
    config["retain_per_identity"] = 3
    output_dir = tmp_path / "output"

    report = process_dataset(input_dir, output_dir, config, seed=2026, visualize=False)
    records = [json.loads(line) for line in (output_dir / "manifest.jsonl").read_text().splitlines()]
    retained = Counter(record["identity"] for record in records if not record["changed"])

    assert report["unchanged_file_count"] == 6
    assert retained == {"c1_0": 3, "c2_0": 3}


@pytest.mark.parametrize(
    "config_name,expected_operator",
    [
        ("optical_illumination_only.json", "illumination"),
        ("optical_blur_only.json", "blur"),
        ("optical_noise_only.json", "noise"),
    ],
)
def test_single_operator_configs_keep_shape_dtype_and_range(tmp_path, config_name, expected_operator):
    input_dir = tmp_path / "input"
    make_dataset(input_dir, identities=1, samples=1)
    config_path = Path(__file__).resolve().parents[1] / "configs" / config_name
    output_dir = tmp_path / "output"

    process_dataset(input_dir, output_dir, resolve_config(config_path=config_path), seed=9, visualize=False)

    source = next(input_dir.rglob("*.png"))
    result = next(output_dir.rglob("*.png"))
    source_pixels = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    result_pixels = cv2.imread(str(result), cv2.IMREAD_GRAYSCALE)
    assert result_pixels.shape == source_pixels.shape
    assert result_pixels.dtype == np.uint8
    assert int(result_pixels.min()) >= 0
    assert int(result_pixels.max()) <= 255
    record = json.loads((output_dir / "augmentation_records.jsonl").read_text().strip())
    assert record["identity"] == "c1_0"
    assert record["operators"] == [expected_operator]
    assert expected_operator in record["parameters"]


def test_output_is_not_overwritten_by_default(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    make_dataset(input_dir, identities=1, samples=1)
    output_dir.mkdir()
    with pytest.raises(FileExistsError):
        process_dataset(input_dir, output_dir, resolve_config("none"), visualize=False)


def test_flat_filename_identity_rule():
    assert infer_identity(Path("c3_12_sample6.png")) == "c3_12"
