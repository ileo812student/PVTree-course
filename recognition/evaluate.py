#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PVTree-course contributors.
# Course-project addition; see LICENSE for terms and README for upstream attribution.
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from recognition.data import PairImageDataset, image_transform
from recognition.metrics import verification_metrics
from recognition.model import ResNet18Embedding


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    if config.get("require_cuda", True) and device.type != "cuda":
        raise RuntimeError("CUDA is required; refusing CPU evaluation")
    model = ResNet18Embedding(int(config["embedding_dim"])).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    dataset = PairImageDataset(args.pairs, image_transform(int(config["image_size"])))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    embeddings = {}
    started = time.perf_counter()
    amp_enabled = bool(config.get("amp", True)) and device.type == "cuda"
    with torch.inference_mode():
        for images, paths in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                values = model(images)
            values = values.float().cpu().numpy()
            embeddings.update(zip(paths, values))
    with args.pairs.open(newline="", encoding="utf-8") as handle:
        pairs = list(csv.DictReader(handle))
    scores = [float(np.dot(embeddings[row["path1"]], embeddings[row["path2"]])) for row in pairs]
    labels = [int(row["label"]) for row in pairs]
    result = verification_metrics(scores, labels)
    roc = result.pop("roc")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Evaluation output is non-empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "roc.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["fpr", "tpr", "threshold"])
        writer.writerows(zip(roc["fpr"], roc["tpr"], roc["thresholds"]))
    with (args.output / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["label", "score", "path1", "path2"])
        for row, label, score in zip(pairs, labels, scores):
            writer.writerow([label, score, row["path1"], row["path2"]])
    genuine = np.asarray([score for score, label in zip(scores, labels) if label == 1])
    impostor = np.asarray([score for score, label in zip(scores, labels) if label == 0])
    def summary(values):
        return {"mean": float(values.mean()), "std": float(values.std()),
                "min": float(values.min()), "max": float(values.max()),
                "q01": float(np.quantile(values, 0.01)), "q50": float(np.quantile(values, 0.50)),
                "q99": float(np.quantile(values, 0.99))}
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    pair_digest = hashlib.sha256(args.pairs.read_bytes()).hexdigest()
    result.update({"label": "SMOKE TEST" if args.smoke_test else "EVALUATION",
                   "checkpoint": str(args.checkpoint.resolve()), "pairs": str(args.pairs.resolve()),
                   "checkpoint_sha256": digest, "pairs_sha256": pair_digest,
                   "unique_images": len(embeddings), "device": str(device),
                   "model_on_cuda": next(model.parameters()).device.type == "cuda",
                   "amp_enabled": amp_enabled, "elapsed_seconds": time.perf_counter() - started,
                   "similarity_distribution": {"genuine": summary(genuine), "impostor": summary(impostor)},
                   "note": "Smoke metrics are not experimental conclusions." if args.smoke_test else
                           "Formal evaluation; no parameter selection was performed on this pair file."})
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
