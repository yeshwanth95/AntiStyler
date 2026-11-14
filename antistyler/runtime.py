"""Seed management and experiment provenance."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import secrets
import subprocess
import sys


def seed_everything(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_runtime(config):
    # Must precede CUDA initialization, including any model/device allocation.
    if config.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if config.cache_dir is not None:
        os.environ["TORCH_HOME"] = config.cache_dir

    import torch

    if config.seed is None:
        config.seed = secrets.randbits(32)
    torch.set_num_threads(config.num_threads)
    torch.use_deterministic_algorithms(config.deterministic)
    torch.backends.cudnn.deterministic = config.deterministic
    torch.backends.cudnn.benchmark = not config.deterministic
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    seed_everything(config.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if config.device == "auto" else torch.device(config.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Supported devices are cpu, cuda, and cuda:N")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_manifest(output_dir, config, command, device):
    checkpoints = {}
    for name, section in (("detector", config.detector), ("defense", config.defense)):
        if section.checkpoint:
            checkpoints[name] = file_hash(section.checkpoint)
    write_provenance(output_dir, config, command, device, checkpoints, {
        "input_sha256": file_hash(config.input_image),
        "rng_policy": "Reset to the run seed before each model construction and stage; both defense inputs use the same seed.",
    })


def write_provenance(output_dir, config, command, device, checkpoints, extra):
    import torch

    versions = {}
    for package in ("torch", "torchvision", "numpy", "Pillow", "ultralytics", "pycocotools", "opencv-python"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        root = Path(__file__).resolve().parents[1]
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True,
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    # Also fingerprint uncommitted implementation files for useful local-run provenance.
    sources = {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    write_json(output_dir / "resolved_config.json", config.to_dict())
    write_json(output_dir / "metadata.json", {
        "command": command, "argv": sys.argv,
        "seed": config.runtime.seed, "deterministic": config.runtime.deterministic,
        "device": str(device), "python": platform.python_version(),
        "platform": platform.platform(), "packages": versions,
        "cuda_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "git_revision": revision, "git_dirty": dirty, "source_sha256": sources,
        "checkpoint_sha256": checkpoints, **extra,
    })
    (output_dir / "environment.txt").write_text(
        "".join(f"{name}=={version}\n" for name, version in versions.items())
    )
