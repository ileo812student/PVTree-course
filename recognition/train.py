#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PVTree-course contributors.
# Course-project addition; see LICENSE for terms and README for upstream attribution.
"""Train the fixed ResNet18 + ArcFace recognition baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from recognition.data import IdentityFolderDataset, PairImageDataset, image_transform
from recognition.metrics import verification_metrics
from recognition.model import ArcFaceHead, ResNet18Embedding


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(module: nn.Module) -> str:
    """Hash tensor names, metadata and values in a serialization-independent order."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8")); digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii")); digest.update(b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii")); digest.update(b"\0")
        digest.update(value.numpy().tobytes(order="C")); digest.update(b"\n")
    return digest.hexdigest()


class GpuMonitor:
    """Sample utilization without adding an NVML dependency."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if torch.cuda.is_available():
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def _run(self) -> None:
        command = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw",
                   "--format=csv,noheader,nounits", "--id=0"]
        while not self._stop.is_set():
            try:
                output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL, timeout=2).strip()
                values = [float(value.strip()) for value in output.split(",")]
                if len(values) == 3:
                    self.samples.append({"gpu_util_percent": values[0], "memory_used_mib": values[1],
                                         "power_watts": values[2]})
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            self._stop.wait(self.interval)

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        if not self.samples:
            return {"sample_count": 0}
        result = {"sample_count": len(self.samples)}
        for key in self.samples[0]:
            values = [sample[key] for sample in self.samples]
            result[f"{key}_mean"] = float(np.mean(values))
            result[f"{key}_max"] = float(np.max(values))
        return result


def evaluate_dev(model: nn.Module, pairs_path: Path, device: torch.device, image_size: int,
                 batch_size: int, num_workers: int, amp_enabled: bool) -> dict:
    dataset = PairImageDataset(pairs_path, image_transform(image_size))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=device.type == "cuda", persistent_workers=num_workers > 0)
    embeddings: dict[str, np.ndarray] = {}
    model.eval()
    with torch.inference_mode():
        for images, paths in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=amp_enabled and device.type == "cuda"):
                values = model(images)
            embeddings.update(zip(paths, values.float().cpu().numpy()))
    with pairs_path.open(newline="", encoding="utf-8") as handle:
        pairs = list(csv.DictReader(handle))
    scores = [float(np.dot(embeddings[row["path1"]], embeddings[row["path2"]])) for row in pairs]
    labels = [int(row["label"]) for row in pairs]
    result = verification_metrics(scores, labels)
    result.pop("roc")
    result["unique_images"] = len(embeddings)
    return result


def save_checkpoint(path: Path, *, epoch: int, model: nn.Module, head: nn.Module,
                    optimizer, scheduler, scaler, class_to_idx: dict, config: dict,
                    logs: list[dict], best_dev_eer: float, best_epoch: int,
                    smoke_test: bool) -> None:
    torch.save({"epoch": epoch, "model": model.state_dict(), "head": head.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(), "class_to_idx": class_to_idx,
                "config": config, "logs": logs, "best_dev_eer": best_dev_eer,
                "best_epoch": best_epoch, "smoke_test": smoke_test}, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=0, help="override config for smoke/benchmark runs")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--max-identities", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--no-dev-eval", action="store_true")
    parser.add_argument("--train-root", type=Path, help="override train_root")
    parser.add_argument("--output-dir", type=Path, help="override output_dir")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.train_root:
        config["train_root"] = str(args.train_root.resolve())
    if args.output_dir:
        config["output_dir"] = str(args.output_dir.resolve())
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.num_workers >= 0:
        config["num_workers"] = args.num_workers
    epochs = args.epochs or int(config["epochs"])
    config["epochs_effective"] = epochs
    config["command_mode"] = "benchmark" if args.benchmark else ("smoke_test" if args.smoke_test else "formal")

    seed = int(config["seed"])
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.get("require_cuda", True) and device.type != "cuda":
        raise RuntimeError("CUDA is required; refusing CPU formal training")

    output = Path(config["output_dir"]).resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"Output directory is non-empty: {output}; use a new path or --resume")
    output.mkdir(parents=True, exist_ok=True)
    dataset = IdentityFolderDataset(config["train_root"], image_transform(config["image_size"], True,
                                    config["augmentation"]["rotation_degrees"]), args.max_identities)
    generator = torch.Generator().manual_seed(seed)
    workers = int(config["num_workers"])
    pin_memory = bool(config.get("pin_memory", True)) and device.type == "cuda"
    persistent_workers = bool(config.get("persistent_workers", True)) and workers > 0
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=True,
                        num_workers=workers, generator=generator, pin_memory=pin_memory,
                        persistent_workers=persistent_workers)

    model = ResNet18Embedding(int(config["embedding_dim"])).to(device)
    head = ArcFaceHead(int(config["embedding_dim"]), len(dataset.class_to_idx),
                       config["arcface"]["scale"], config["arcface"]["margin"]).to(device)
    if next(model.parameters()).device.type != "cuda" or next(head.parameters()).device.type != "cuda":
        raise RuntimeError("Model or ArcFace head is not on CUDA")
    initial_model_sha256 = state_dict_sha256(model)
    initial_head_sha256 = state_dict_sha256(head)
    initial_combined_sha256 = hashlib.sha256(
        (initial_model_sha256 + initial_head_sha256).encode("ascii")
    ).hexdigest()
    config["initial_state_sha256"] = {
        "model": initial_model_sha256,
        "arcface_head": initial_head_sha256,
        "combined": initial_combined_sha256,
    }
    (output / "config_resolved.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    optimizer = AdamW(list(model.parameters()) + list(head.parameters()), lr=float(config["learning_rate"]),
                      weight_decay=float(config["weight_decay"]))
    warmup = int(config["warmup_epochs"])
    scheduler = LambdaLR(optimizer, lambda epoch: (epoch + 1) / max(warmup, 1) if epoch < warmup
                         else 0.5 * (1 + math.cos(math.pi * (epoch - warmup) / max(epochs - warmup, 1))))
    amp_enabled = bool(config.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_epoch, logs, best_dev_eer, best_epoch = 0, [], float("inf"), 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        head.load_state_dict(checkpoint["head"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        logs = list(checkpoint.get("logs", []))
        best_dev_eer = float(checkpoint.get("best_dev_eer", float("inf")))
        best_epoch = int(checkpoint.get("best_epoch", 0))

    dev_pairs = Path(config["dev_pairs"]).resolve() if config.get("dev_pairs") and not args.no_dev_eval else None
    if dev_pairs and not dev_pairs.is_file():
        raise FileNotFoundError(f"Development pair file not found: {dev_pairs}")

    monitor = GpuMonitor()
    monitor.start()
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    input_device_verified = False
    for epoch in range(start_epoch, epochs):
        model.train()
        head.train()
        epoch_started = time.perf_counter()
        total_loss, count, data_seconds, compute_seconds = 0.0, 0, 0.0, 0.0
        batch_finished = time.perf_counter()
        for images, labels, _ in loader:
            loaded = time.perf_counter()
            data_seconds += loaded - batch_finished
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            input_device_verified = input_device_verified or (images.device.type == "cuda" and labels.device.type == "cuda")
            compute_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                embeddings = model(images)
                logits = head(embeddings, labels)
                loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}: {loss.item()}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().item()) * len(labels)
            count += len(labels)
            compute_seconds += time.perf_counter() - compute_started
            batch_finished = time.perf_counter()
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - epoch_started
        scheduler.step()
        row = {"epoch": epoch + 1, "loss": total_loss / count,
               "learning_rate": optimizer.param_groups[0]["lr"], "train_seconds": train_seconds,
               "images_per_second": count / train_seconds, "data_seconds": data_seconds,
               "compute_seconds": compute_seconds,
               "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
               "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024 ** 3}
        if dev_pairs:
            evaluated = evaluate_dev(model, dev_pairs, device, int(config["image_size"]),
                                     int(config.get("eval_batch_size", config["batch_size"])),
                                     workers, amp_enabled)
            row.update({f"dev_{key}": value for key, value in evaluated.items()})
        logs.append(row)
        improved = dev_pairs is not None and float(row["dev_eer"]) < best_dev_eer
        if improved:
            best_dev_eer, best_epoch = float(row["dev_eer"]), epoch + 1
        save_checkpoint(output / "checkpoint_last.pt", epoch=epoch, model=model, head=head,
                        optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                        class_to_idx=dataset.class_to_idx, config=config, logs=logs,
                        best_dev_eer=best_dev_eer, best_epoch=best_epoch,
                        smoke_test=args.smoke_test)
        if improved:
            save_checkpoint(output / "checkpoint_best_dev.pt", epoch=epoch, model=model, head=head,
                            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                            class_to_idx=dataset.class_to_idx, config=config, logs=logs,
                            best_dev_eer=best_dev_eer, best_epoch=best_epoch,
                            smoke_test=args.smoke_test)
        with (output / "train_log.jsonl").open("w", encoding="utf-8") as handle:
            for item in logs:
                handle.write(json.dumps(item, sort_keys=True) + "\n")
        print(json.dumps(row), flush=True)

    elapsed = time.perf_counter() - started
    gpu_monitor = monitor.stop()
    selected = output / ("checkpoint_best_dev.pt" if dev_pairs else "checkpoint_last.pt")
    report = {
        "label": "BENCHMARK" if args.benchmark else ("SMOKE TEST" if args.smoke_test else "FORMAL TRAINING"),
        "config_source": str(args.config.resolve()), "resolved_config": config,
        "device": str(device), "gpu_name": torch.cuda.get_device_name(0),
        "model_on_cuda": next(model.parameters()).device.type == "cuda",
        "arcface_on_cuda": next(head.parameters()).device.type == "cuda",
        "input_on_cuda_verified": input_device_verified, "amp_enabled": amp_enabled,
        "initial_model_sha256": initial_model_sha256,
        "initial_arcface_head_sha256": initial_head_sha256,
        "initial_combined_sha256": initial_combined_sha256,
        "pin_memory": pin_memory, "persistent_workers": persistent_workers,
        "identities": len(dataset.class_to_idx), "images": len(dataset),
        "epochs_run": max(0, epochs - start_epoch), "logs": logs,
        "elapsed_seconds": elapsed,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024 ** 3,
        "gpu_monitor": gpu_monitor, "best_dev_eer": None if not dev_pairs else best_dev_eer,
        "best_epoch": None if not dev_pairs else best_epoch,
        "checkpoint_last": str((output / "checkpoint_last.pt").resolve()),
        "selected_checkpoint": str(selected.resolve()),
        "selected_checkpoint_sha256": sha256_file(selected),
        "dev_pairs_sha256": sha256_file(dev_pairs) if dev_pairs else None,
    }
    (output / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
