#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PVTree-course contributors.
# Course-project addition; see LICENSE for terms and README for upstream attribution.
"""Generate PCE images from an explicit, hashed input/latent assignment manifest."""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from options.test_options import TestOptions
from data.base_dataset import get_transform
from models import create_model
from util import util


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vector_sha256(vector: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(vector, dtype="<f4").tobytes(order="C")).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class StyleGenerationOptions(TestOptions):
    def initialize(self, parser):
        parser = super().initialize(parser)
        parser.add_argument("--assignment_manifest", required=True)
        parser.add_argument("--latent_pool", required=True)
        parser.add_argument("--inference_seed", type=int, default=2028)
        parser.add_argument("--generator_mode", choices=("eval", "batch_stats_no_dropout"),
                            default="eval")
        parser.add_argument("--sample_noise_mode", choices=("global_stream", "path_sha256_v1"),
                            default="global_stream")
        parser.add_argument("--trace_noise_calls", action="store_true",
                            help="record exact torch.randn tensors (small acceptance runs only)")
        parser.add_argument("--stop_after", type=int, default=0,
                            help="controlled interruption after N newly generated rows (acceptance testing only)")
        return parser


def parse_without_checkpoint_write():
    options = StyleGenerationOptions()
    opt = options.gather_options()
    opt.isTrain = False
    if opt.suffix:
        opt.name = opt.name + "_" + opt.suffix.format(**vars(opt))
    gpu_ids = []
    for value in opt.gpu_ids.split(","):
        parsed = int(value)
        if parsed >= 0:
            gpu_ids.append(parsed)
    opt.gpu_ids = gpu_ids
    if gpu_ids:
        torch.cuda.set_device(gpu_ids[0])
    return opt


def combined_input_hash(dataroot: Path, rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for relative in sorted({row["input_relative"] for row in rows}):
        path = dataroot / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(relative.encode("utf-8")); digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii")); digest.update(b"\n")
    return digest.hexdigest()


def stable_sample_seed(inference_seed: int, relative_path: str) -> int:
    """Derive a stable signed-63-bit seed without Python's process-randomized hash()."""
    normalized = relative_path.replace("\\", "/")
    payload = f"pce-path-noise-v1\0{int(inference_seed)}\0{normalized}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)


def tensor_bytes_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def named_tensors_sha256(items) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(items):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8")); digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii")); digest.update(b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii")); digest.update(b"\0")
        digest.update(value.tobytes(order="C")); digest.update(b"\n")
    return digest.hexdigest()


def model_state_hashes(model) -> dict[str, str]:
    parameters, buffers = [], []
    for network_name in model.model_names:
        if not isinstance(network_name, str):
            continue
        network = getattr(model, "net" + network_name)
        parameters.extend((f"{network_name}.{name}", value) for name, value in network.named_parameters())
        buffers.extend((f"{network_name}.{name}", value) for name, value in network.named_buffers())
    return {
        "parameters_sha256": named_tensors_sha256(parameters),
        "buffers_sha256": named_tensors_sha256(buffers),
    }


@contextmanager
def isolated_sample_rng(inference_seed: int, relative_path: str, gpu_ids: list[int]):
    """Bind generator randomness to one stable sample key and restore caller RNG state."""
    sample_seed = stable_sample_seed(inference_seed, relative_path)
    cpu_outer = torch.get_rng_state().clone()
    cuda_outer = [torch.cuda.get_rng_state(device).clone() for device in gpu_ids]
    with torch.random.fork_rng(devices=gpu_ids, enabled=True, device_type="cuda"):
        torch.manual_seed(sample_seed)
        for device in gpu_ids:
            with torch.cuda.device(device):
                torch.cuda.manual_seed(sample_seed)
        local_before = [tensor_bytes_sha256(torch.cuda.get_rng_state(device)) for device in gpu_ids]
        evidence = {
            "sample_noise_seed": sample_seed,
            "rng_scheme": "sha256('pce-path-noise-v1\\0' + inference_seed + '\\0' + relative_path)",
            "cuda_rng_before_sha256": local_before,
        }
        yield evidence
        evidence["cuda_rng_after_sha256"] = [
            tensor_bytes_sha256(torch.cuda.get_rng_state(device)) for device in gpu_ids
        ]
    if not torch.equal(cpu_outer, torch.get_rng_state()):
        raise RuntimeError(f"CPU RNG state leaked while generating {relative_path}")
    for device, expected in zip(gpu_ids, cuda_outer):
        if not torch.equal(expected, torch.cuda.get_rng_state(device)):
            raise RuntimeError(f"CUDA RNG state leaked on cuda:{device} while generating {relative_path}")


@contextmanager
def trace_torch_randn_calls():
    """Acceptance-only observer for the exact random tensors consumed by G."""
    original = torch.randn
    calls = []

    def traced(*args, **kwargs):
        value = original(*args, **kwargs)
        calls.append({
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "sha256": tensor_bytes_sha256(value),
        })
        return value

    torch.randn = traced
    try:
        yield calls
    finally:
        torch.randn = original


def module_inventory(model) -> dict:
    inventory = {}
    for network_name in model.model_names:
        if not isinstance(network_name, str):
            continue
        network = getattr(model, "net" + network_name)
        modules = list(network.modules())
        inventory[network_name] = {
            "training": bool(network.training),
            "dropout_modules": sum(isinstance(module, torch.nn.modules.dropout._DropoutNd) for module in modules),
            "batchnorm_modules": sum(isinstance(module, torch.nn.modules.batchnorm._BatchNorm) for module in modules),
            "batchnorm_training": sum(
                isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.training for module in modules
            ),
            "batchnorm_tracking_stats": sum(
                isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.track_running_stats
                for module in modules
            ),
            "instancenorm_modules": sum(isinstance(module, torch.nn.modules.instancenorm._InstanceNorm) for module in modules),
            "any_dropout_training": any(
                isinstance(module, torch.nn.modules.dropout._DropoutNd) and module.training for module in modules
            ),
        }
    return inventory


def set_generator_mode(model, mode: str) -> None:
    model.eval()
    if mode == "batch_stats_no_dropout":
        for module in model.netG.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.train()
                module.track_running_stats = False


def checkpoint_path(opt, network: str) -> Path:
    return Path(opt.checkpoints_dir).resolve() / opt.name / f"{opt.epoch}_net_{network}.pth"


def validate_resume(output: Path, state: dict, rows: list[dict]) -> dict[str, dict]:
    state_path, records_path = output / "generation_state.json", output / "generation_records.jsonl"
    if not state_path.is_file() or not records_path.is_file():
        raise RuntimeError("Resume requires generation_state.json and generation_records.jsonl")
    existing_state = json.loads(state_path.read_text(encoding="utf-8"))
    if existing_state != state:
        raise RuntimeError("Resume state mismatch: configuration, weights, inputs or latent mapping changed")
    records = read_jsonl(records_path)
    by_output = {row["output_relative"]: row for row in records}
    if len(by_output) != len(records):
        raise RuntimeError("Duplicate paths in generation_records.jsonl")
    expected = {row["output_relative"] for row in rows}
    actual = {
        path.relative_to(output).as_posix() for path in output.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    }
    unknown = actual - set(by_output)
    if unknown:
        raise RuntimeError(f"Unknown existing output files; refusing to skip: {sorted(unknown)[:3]}")
    if not set(by_output).issubset(expected):
        raise RuntimeError("Recorded output does not belong to current assignment")
    for relative, record in by_output.items():
        path = output / relative
        if not path.is_file() or sha256_file(path) != record["output_sha256"]:
            raise RuntimeError(f"Existing output hash mismatch: {path}")
    return by_output


def main() -> None:
    opt = parse_without_checkpoint_write()
    if not opt.eval:
        raise RuntimeError("Style experiment requires --eval for both groups")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU PCE generation")
    if opt.preprocess != "resize" or opt.load_size != 128 or opt.crop_size != 128 or not opt.no_flip:
        raise ValueError("Expected deterministic --preprocess resize --load_size 128 --crop_size 128 --no_flip")
    if opt.sample_noise_mode == "path_sha256_v1" and opt.batch_size != 1:
        raise ValueError("path_sha256_v1 requires --batch_size 1")

    dataroot = Path(opt.dataroot).resolve()
    assignment_path = Path(opt.assignment_manifest).resolve()
    latent_path = Path(opt.latent_pool).resolve()
    output = Path(opt.results_dir).resolve()
    rows = read_jsonl(assignment_path)
    pool = np.load(latent_path, allow_pickle=False)
    if pool.dtype != np.float32 or pool.ndim != 2 or pool.shape[1] != opt.nz:
        raise ValueError(f"Unexpected latent pool specification: {pool.shape} {pool.dtype}")
    latent_ids = [int(row["latent_id"]) for row in rows]
    if len(rows) != len({row["output_relative"] for row in rows}):
        raise ValueError("Assignment output paths are not unique")
    for row in rows:
        latent_id = int(row["latent_id"])
        if not 0 <= latent_id < len(pool) or vector_sha256(pool[latent_id]) != row["z_sha256"]:
            raise ValueError(f"Latent hash mismatch for ID {latent_id}")
    sample_noise_seeds = [stable_sample_seed(opt.inference_seed, row["output_relative"]) for row in rows]
    if opt.sample_noise_mode == "path_sha256_v1" and len(sample_noise_seeds) != len(set(sample_noise_seeds)):
        raise ValueError("Stable sample-noise seed collision in assignment manifest")

    random.seed(opt.inference_seed)
    np.random.seed(opt.inference_seed)
    torch.manual_seed(opt.inference_seed)
    torch.cuda.manual_seed_all(opt.inference_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    model = create_model(opt)
    model.setup(opt)
    # BaseModel enables cuDNN benchmarking during construction; restore the
    # frozen deterministic inference contract afterwards.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    set_generator_mode(model, opt.generator_mode)
    inventory = module_inventory(model)
    if any(item["any_dropout_training"] for item in inventory.values()):
        raise RuntimeError(f"Model did not enter deterministic eval mode: {inventory}")
    if opt.generator_mode == "eval" and any(item["batchnorm_training"] for item in inventory.values()):
        raise RuntimeError(f"BatchNorm unexpectedly active in eval mode: {inventory}")
    if opt.generator_mode == "batch_stats_no_dropout":
        generator_inventory = inventory["G"]
        if (generator_inventory["batchnorm_training"] != generator_inventory["batchnorm_modules"]
                or generator_inventory["batchnorm_tracking_stats"]):
            raise RuntimeError(f"Batch-stat compatibility mode was not applied completely: {inventory}")
    checkpoint_hashes = {
        name: {"path": str(checkpoint_path(opt, name)), "sha256": sha256_file(checkpoint_path(opt, name))}
        for name in model.model_names if isinstance(name, str)
    }
    source_paths = {
        "style_generate": Path(__file__).resolve(),
        "networks": Path(__file__).resolve().parent / "models/networks.py",
        "rpg_model": Path(__file__).resolve().parent / "models/rpg_model.py",
        "base_model": Path(__file__).resolve().parent / "models/base_model.py",
    }
    source_hashes = {name: {"path": str(path), "sha256": sha256_file(path)}
                     for name, path in source_paths.items()}
    state_hashes_before = model_state_hashes(model)
    relevant_options = {
        key: getattr(opt, key) for key in (
            "name", "epoch", "model", "netG", "netE", "input_nc", "output_nc", "nz", "ngf", "nef",
            "norm", "nl", "where_add", "upsample", "use_dropout", "preprocess", "load_size", "crop_size",
            "no_flip", "eval", "inference_seed", "batch_size", "sample_noise_mode",
            "trace_noise_calls",
        )
    }
    relevant_options["generator_mode"] = opt.generator_mode
    state = {
        "schema_version": 2,
        "dataroot": str(dataroot),
        "assignment_manifest": str(assignment_path),
        "assignment_manifest_sha256": sha256_file(assignment_path),
        "latent_pool": str(latent_path),
        "latent_pool_sha256": sha256_file(latent_path),
        "combined_input_sha256": combined_input_hash(dataroot, rows),
        "assignment_rows": len(rows),
        "unique_latent_ids": len(set(latent_ids)),
        "checkpoint_hashes": checkpoint_hashes,
        "source_hashes": source_hashes,
        "options": relevant_options,
        "module_inventory_after_eval": inventory,
        "model_state_before_generation": state_hashes_before,
        "explicit_latent_encode": False,
        "encoder_called": False,
        "forward_context": "torch.inference_mode",
        "sample_noise_seed_unique_count": len(set(sample_noise_seeds)),
        "cuda_device": torch.cuda.get_device_name(opt.gpu_ids[0]),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }

    if output.exists():
        if not opt.resume:
            raise FileExistsError(f"Output exists; use a new directory or verified --resume: {output}")
        completed = validate_resume(output, state, rows)
    else:
        if opt.resume:
            raise FileNotFoundError(f"Cannot resume missing output directory: {output}")
        output.mkdir(parents=True)
        (output / "generation_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
        (output / "generation_records.jsonl").touch()
        completed = {}

    transform = get_transform(opt, grayscale=True)
    started, generated, controlled_stop = time.perf_counter(), 0, False
    records_path = output / "generation_records.jsonl"
    with records_path.open("a", encoding="utf-8", buffering=1) as record_file:
        for row in tqdm(rows, desc=f"PCE {rows[0].get('assignment', 'style')}"):
            relative = row["output_relative"]
            if relative in completed:
                continue
            input_path, output_path = dataroot / row["input_relative"], output / relative
            if output_path.exists():
                raise RuntimeError(f"Unknown existing file; refusing overwrite: {output_path}")
            with Image.open(input_path) as image:
                tensor = transform(image.convert("RGB")).unsqueeze(0)
            model.set_input({"A": tensor, "B": tensor.clone(), "A_paths": [str(input_path)],
                             "B_paths": [str(input_path)], "B_path": str(input_path)})
            latent = torch.from_numpy(pool[int(row["latent_id"])][None]).to(model.device)
            if opt.sample_noise_mode == "path_sha256_v1":
                rng_context = isolated_sample_rng(opt.inference_seed, relative, opt.gpu_ids)
            else:
                rng_context = nullcontext({"sample_noise_seed": None, "rng_scheme": "legacy_global_stream"})
            with rng_context as rng_evidence:
                trace_context = trace_torch_randn_calls() if opt.trace_noise_calls else nullcontext(None)
                with trace_context as noise_calls:
                    with torch.inference_mode():
                        fake, _ = model.test(latent, encode=False)
            pixels = util.tensor2im(fake)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            util.save_image(pixels, str(output_path))
            record = {
                **row,
                "input_path": str(input_path), "output_path": str(output_path),
                "input_sha256": sha256_file(input_path), "output_sha256": sha256_file(output_path),
                "shape": list(pixels.shape), "dtype": str(pixels.dtype),
                "input_tensor_dtype": str(tensor.dtype),
                "input_tensor_min": float(tensor.min().item()),
                "input_tensor_max": float(tensor.max().item()),
                "latent_dtype": str(latent.dtype),
                "latent_min": float(latent.min().item()),
                "latent_max": float(latent.max().item()),
                "raw_output_dtype": str(fake.dtype),
                "raw_output_min": float(fake.min().item()),
                "raw_output_max": float(fake.max().item()),
                "raw_output_finite": bool(torch.isfinite(fake).all().item()),
                "sample_noise_seed": rng_evidence["sample_noise_seed"],
                "noise_rng_scheme": rng_evidence["rng_scheme"],
                "noise_cuda_rng_before_sha256": rng_evidence.get("cuda_rng_before_sha256"),
                "noise_cuda_rng_after_sha256": rng_evidence.get("cuda_rng_after_sha256"),
                "noise_call_trace": noise_calls,
                "noise_call_trace_sha256": (
                    hashlib.sha256(json.dumps(noise_calls, sort_keys=True).encode("utf-8")).hexdigest()
                    if noise_calls is not None else None
                ),
                "generator_checkpoint_sha256": checkpoint_hashes["G"]["sha256"],
                "encoder_checkpoint_sha256": checkpoint_hashes["E"]["sha256"],
                "generator_mode": opt.generator_mode,
                "generation_batch_size": 1,
                "torch_version": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cuda_device": torch.cuda.get_device_name(opt.gpu_ids[0]),
            }
            record_file.write(json.dumps(record, sort_keys=True) + "\n")
            generated += 1
            if opt.stop_after and generated >= opt.stop_after:
                controlled_stop = True
                break

    torch.cuda.synchronize()
    records = read_jsonl(records_path)
    state_hashes_after = model_state_hashes(model)
    state_unchanged = state_hashes_before == state_hashes_after
    report = {
        **state,
        "generated_this_run": generated,
        "total_completed": len(records),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024 ** 3,
        "model_state_after_generation": state_hashes_after,
        "model_parameters_and_buffers_unchanged": state_unchanged,
        "controlled_stop": controlled_stop,
        "complete": len(records) == len(rows),
    }
    (output / "generation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not state_unchanged:
        raise RuntimeError("PCE parameters or buffers changed during supposedly immutable inference")
    if not report["complete"] and not controlled_stop:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
