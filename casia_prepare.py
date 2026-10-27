#!/usr/bin/env python3
"""Inventory CASIA Multi-Spectral images, extract 940 nm palm ROIs and make splits.

The source tree is read-only.  Filenames must match
``<subject>_<l|r>_<wavelength>_<sample>.<ext>``; ambiguous files are reported.
Samples 01--03 and 04--06 are mapped to sessions 1 and 2 respectively by an
explicit, configurable boundary because the flat filenames contain no session
token.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
NAME_RE = re.compile(
    r"^(?P<subject>\d+)_(?P<hand>[lr])_(?P<wavelength>\d+|wht)_(?P<sample>\d+)\.(?P<ext>jpe?g|png|bmp|tiff?)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SourceRecord:
    subject_id: str
    hand: str
    palm_identity: str
    session: int
    sample_index: int
    wavelength: str
    source_path: str
    image_size: list[int]
    image_mode: str
    dtype: str


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def scan_source(root: Path, session_boundary: int = 3) -> tuple[list[SourceRecord], list[str]]:
    records: list[SourceRecord] = []
    ignored: list[str] = []
    for path in sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.as_posix().casefold()):
        match = NAME_RE.match(path.name)
        if not match:
            ignored.append(str(path.resolve()))
            continue
        fields = match.groupdict()
        subject = fields["subject"].zfill(3)
        hand = "left" if fields["hand"].lower() == "l" else "right"
        sample = int(fields["sample"])
        with Image.open(path) as image:
            mode, size = image.mode, list(image.size)
            dtype = str(np.asarray(image).dtype)
        records.append(SourceRecord(
            subject_id=subject,
            hand=hand,
            palm_identity=f"subject_{subject}_{hand}",
            session=1 if sample <= session_boundary else 2,
            sample_index=sample,
            wavelength=fields["wavelength"].upper(),
            source_path=str(path.resolve()),
            image_size=size,
            image_mode=mode,
            dtype=dtype,
        ))
    return records, ignored


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records), encoding="utf-8")


def largest_hand_mask(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    threshold, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("no foreground contour")
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < gray.size * 0.12:
        raise ValueError("largest foreground contour is too small")
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    return mask, contour, float(threshold)


def extract_roi(gray: np.ndarray, hand: str) -> tuple[np.ndarray, dict, dict[str, np.ndarray]]:
    mask, contour, threshold = largest_hand_mask(gray)
    normalized = gray.copy()
    normalized_mask = mask.copy()
    vertical_flip = hand == "right"
    if vertical_flip:
        normalized = cv2.flip(normalized, 0)
        normalized_mask = cv2.flip(normalized_mask, 0)

    distance = cv2.distanceTransform(normalized_mask, cv2.DIST_L2, 5)
    _, radius, _, center_tuple = cv2.minMaxLoc(distance)
    cx, cy = map(float, center_tuple)
    if radius < min(gray.shape) * 0.08:
        raise ValueError(f"palm radius too small: {radius:.2f}")

    ys, xs = np.nonzero(normalized_mask)
    wrist_select = xs > cx + 0.70 * radius
    if int(wrist_select.sum()) < 50:
        raise ValueError("unable to locate wrist-side landmark")
    wx = float(np.mean(xs[wrist_select]))
    wy = float(np.mean(ys[wrist_select]))
    angle = math.degrees(math.atan2(wy - cy, wx - cx))
    matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    aligned = cv2.warpAffine(normalized, matrix, (gray.shape[1], gray.shape[0]), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    aligned_mask = cv2.warpAffine(normalized_mask, matrix, (gray.shape[1], gray.shape[0]), flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # A compact central-palm square avoids finger roots and bright wrist hardware.
    side = int(round(1.90 * radius))
    side = max(96, min(side, min(gray.shape)))
    crop_cx = cx - 0.03 * radius
    crop_cy = cy
    x0 = int(round(crop_cx - side / 2))
    y0 = int(round(crop_cy - side / 2))
    x1, y1 = x0 + side, y0 + side
    out_of_bounds = x0 < 0 or y0 < 0 or x1 > gray.shape[1] or y1 > gray.shape[0]
    if out_of_bounds:
        raise ValueError(f"ROI outside image: {(x0, y0, x1, y1)}")
    crop = aligned[y0:y1, x0:x1]
    crop_mask = aligned_mask[y0:y1, x0:x1]
    roi = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA)
    occupancy = float(np.mean(crop_mask > 0))
    border_background = float(np.mean(np.concatenate([
        crop_mask[0], crop_mask[-1], crop_mask[:, 0], crop_mask[:, -1]
    ]) == 0))
    flags = []
    if occupancy < 0.60:
        flags.append("low_palm_occupancy")
    if occupancy > 0.99:
        flags.append("crop_may_be_too_tight")
    if border_background > 0.80:
        flags.append("large_background_border")
    if float(np.std(roi)) < 4.0:
        flags.append("low_contrast_roi")

    metadata = {
        "segmentation_threshold": threshold,
        "palm_center_xy": [cx, cy],
        "wrist_landmark_xy": [wx, wy],
        "alignment_angle_degrees": angle,
        "rotation_180_for_right_hand": False,
        "horizontal_flip": False,
        "vertical_flip_for_right_hand": vertical_flip,
        "crop_box_aligned_xyxy": [x0, y0, x1, y1],
        "palm_radius_px": float(radius),
        "palm_occupancy": occupancy,
        "border_background_ratio": border_background,
        "quality_flags": flags,
    }
    debug = {"mask": mask, "normalized": normalized, "aligned": aligned, "aligned_mask": aligned_mask,
             "crop": crop, "roi": roi, "contour": contour}
    return roi, metadata, debug


def diagnostic_image(gray: np.ndarray, metadata: dict, debug: dict[str, np.ndarray]) -> np.ndarray:
    original = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    contour_panel = original.copy()
    cv2.drawContours(contour_panel, [debug["contour"]], -1, (0, 255, 0), 3)
    normalized_panel = cv2.cvtColor(debug["normalized"], cv2.COLOR_GRAY2BGR)
    cx, cy = metadata["palm_center_xy"]
    wx, wy = metadata["wrist_landmark_xy"]
    cv2.circle(normalized_panel, (round(cx), round(cy)), 8, (0, 0, 255), -1)
    cv2.circle(normalized_panel, (round(wx), round(wy)), 8, (255, 0, 0), -1)
    cv2.line(normalized_panel, (round(cx), round(cy)), (round(wx), round(wy)), (0, 255, 255), 3)
    aligned_panel = cv2.cvtColor(debug["aligned"], cv2.COLOR_GRAY2BGR)
    x0, y0, x1, y1 = metadata["crop_box_aligned_xyxy"]
    cv2.rectangle(aligned_panel, (x0, y0), (x1, y1), (0, 255, 0), 3)
    final = cv2.resize(cv2.cvtColor(debug["roi"], cv2.COLOR_GRAY2BGR), (gray.shape[1], gray.shape[0]))
    panels = [original, contour_panel, normalized_panel, aligned_panel, final]
    labels = ["original", "foreground contour", "palm/wrist landmarks", "aligned + ROI", "final 128x128 ROI"]
    for panel, label in zip(panels, labels):
        cv2.putText(panel, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    return np.concatenate(panels, axis=1)


def make_pairs(success_records: list[dict], output_root: Path, seed: int) -> dict:
    subjects = sorted({record["subject_id"] for record in success_records})
    rng = random.Random(seed)
    rng.shuffle(subjects)
    midpoint = len(subjects) // 2
    split_subjects = {"dev": sorted(subjects[:midpoint]), "test": sorted(subjects[midpoint:])}
    summary = {}
    for split, selected_subjects in split_subjects.items():
        identities = sorted({record["palm_identity"] for record in success_records if record["subject_id"] in selected_subjects})
        (output_root / f"casia_{split}_identities.txt").write_text("\n".join(identities) + "\n", encoding="utf-8")
        grouped: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
        for record in success_records:
            if record["subject_id"] in selected_subjects:
                grouped[record["palm_identity"]][int(record["session"])].append(record)
        pairs = []
        for identity in identities:
            for first in sorted(grouped[identity][1], key=lambda r: r["sample_index"]):
                for second in sorted(grouped[identity][2], key=lambda r: r["sample_index"]):
                    pairs.append((first["output_path"], second["output_path"], 1, identity, identity))
        for left_index, identity_a in enumerate(identities):
            for identity_b in identities[left_index + 1:]:
                for first in sorted(grouped[identity_a][1], key=lambda r: r["sample_index"]):
                    for second in sorted(grouped[identity_b][2], key=lambda r: r["sample_index"]):
                        pairs.append((first["output_path"], second["output_path"], 0, identity_a, identity_b))
        pair_path = output_root / f"casia_{split}_pairs.csv"
        with pair_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["path1", "path2", "label", "identity1", "identity2"])
            writer.writerows(pairs)
        counts = Counter(pair[2] for pair in pairs)
        summary[split] = {"subjects": len(selected_subjects), "identities": len(identities),
                          "genuine_pairs": counts[1], "impostor_pairs": counts[0],
                          "pair_file": str(pair_path.resolve())}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--casia-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("datasets/casia_multi_940_roi"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--session-boundary", type=int, default=3,
                        help="samples <= boundary map to session 1; later samples map to session 2")
    parser.add_argument("--limit", type=int, default=0, help="process only N deterministic 940 nm images")
    parser.add_argument("--visualizations", type=int, default=40)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    source_root = args.casia_root.resolve()
    output_root = args.output.resolve()
    if not source_root.is_dir():
        raise SystemExit(f"CASIA root does not exist: {source_root}")
    if output_root.exists() and any(output_root.iterdir()) and not args.resume:
        raise SystemExit(f"Output is not empty: {output_root}; use --resume or a new directory")
    started = time.time()
    records, ignored = scan_source(source_root, args.session_boundary)
    manifests = output_root / "manifests"
    reports = output_root / "reports"
    visuals = output_root / "visualizations"
    images_root = output_root / "images"
    for directory in (manifests, reports, visuals, images_root):
        directory.mkdir(parents=True, exist_ok=True)
    write_jsonl(manifests / "casia_source_manifest.jsonl", [asdict(record) for record in records])

    inventory = {
        "source_root": str(source_root), "parsed_images": len(records), "ignored_files": ignored,
        "subjects": len({r.subject_id for r in records}),
        "palm_identities": len({r.palm_identity for r in records}),
        "hands": dict(Counter(r.hand for r in records)),
        "wavelength_counts": dict(sorted(Counter(r.wavelength for r in records).items())),
        "session_mapping": f"sample 01-{args.session_boundary:02d} -> session 1; later samples -> session 2",
        "session_counts": dict(sorted(Counter(str(r.session) for r in records).items())),
        "size_counts": dict(Counter(f"{r.image_size[0]}x{r.image_size[1]}" for r in records)),
        "mode_counts": dict(Counter(r.image_mode for r in records)),
        "dtype_counts": dict(Counter(r.dtype for r in records)),
    }
    atomic_json(reports / "casia_inventory.json", inventory)
    selected = [record for record in records if record.wavelength == "940"]
    if args.limit:
        # Round-robin groups deliberately cover both hands and both sessions.
        groups = defaultdict(list)
        for record in selected:
            groups[(record.hand, record.session)].append(record)
        for values in groups.values():
            values.sort(key=lambda r: (r.subject_id, r.sample_index))
        interleaved = []
        keys = [("left", 1), ("right", 1), ("left", 2), ("right", 2)]
        offset = 0
        while len(interleaved) < args.limit and any(offset < len(groups[key]) for key in keys):
            for key in keys:
                if offset < len(groups[key]) and len(interleaved) < args.limit:
                    interleaved.append(groups[key][offset])
            offset += 1
        selected = interleaved

    successes, failures = [], []
    for index, record in enumerate(selected):
        source_path = Path(record.source_path)
        relative = Path(record.palm_identity) / f"{source_path.stem}.png"
        destination = images_root / relative
        diagnostic_path = visuals / f"{source_path.stem}.jpg"
        try:
            if destination.exists() and args.resume:
                with Image.open(destination) as image:
                    if image.size != (128, 128) or image.mode != "L":
                        raise ValueError("existing ROI has wrong specification")
                metadata = {"resumed_existing": True, "quality_flags": []}
            else:
                gray = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    raise ValueError("image cannot be decoded")
                roi, metadata, debug = extract_roi(gray, record.hand)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(destination), roi):
                    raise ValueError("failed to save PNG")
                if index < args.visualizations or metadata["quality_flags"]:
                    cv2.imwrite(str(diagnostic_path), diagnostic_image(gray, metadata, debug),
                                [cv2.IMWRITE_JPEG_QUALITY, 92])
            result = asdict(record)
            result.update(metadata)
            result["output_path"] = str(destination.resolve())
            successes.append(result)
        except Exception as exc:
            failures.append({**asdict(record), "error": str(exc), "diagnostic_path": str(diagnostic_path.resolve())})
    write_jsonl(manifests / "casia_940_roi_manifest.jsonl", successes)
    write_jsonl(manifests / "casia_940_roi_failures.jsonl", failures)
    pair_summary = make_pairs(successes, manifests, args.seed) if not args.limit else {}
    quality_counts = Counter(flag for record in successes for flag in record.get("quality_flags", []))
    report = {
        "inventory": inventory,
        "selected_940": len(selected), "roi_success": len(successes), "roi_failed": len(failures),
        "success_rate": len(successes) / max(len(selected), 1), "quality_flag_counts": dict(quality_counts),
        "pair_summary": pair_summary, "seed": args.seed, "session_boundary": args.session_boundary,
        "elapsed_seconds": time.time() - started,
        "limitations": [
            "Session labels are derived from the configurable sample-number boundary; filenames contain no session token.",
            "ROI alignment uses a segmented palm-center and wrist-side landmark, not learned keypoints.",
            "No CLAHE, sharpening, random augmentation or non-940-nm imagery is used."
        ],
    }
    atomic_json(reports / "casia_roi_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
