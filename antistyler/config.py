"""Strict JSON configuration, with paths relative to the configuration file."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path


@dataclass
class RuntimeConfig:
    seed: int | None = 42
    deterministic: bool = True
    device: str = "auto"
    num_threads: int = 1
    cache_dir: str | None = None


@dataclass
class DetectorConfig:
    weights: str | None = "COCO_V1"
    checkpoint: str | None = None
    min_size: int = 800
    max_size: int = 1333
    score_threshold: float = 0.7


@dataclass
class AttackConfig:
    patch_size: int = 100
    # null preserves the notebook's int((dimension + patch_size) / 2) placement.
    position: list[int] | None = None  # [x, y], top-left corner
    steps: int = 200
    learning_rate: float = 0.1
    target_label: str | int = "person"
    target_boxes: list[list[float]] | None = None  # original-image xyxy coordinates
    target_score_threshold: float = 0.7
    target_selection: str = "all"  # all or highest_score
    loss_weights: dict[str, float] = field(default_factory=lambda: {
        "loss_classifier": 1.0,
        "loss_box_reg": 1.0,
        "loss_objectness": 1.0,
        "loss_rpn_box_reg": 1.0,
    })


@dataclass
class DefenseConfig:
    weights: str | None = "IMAGENET1K_V1"
    checkpoint: str | None = None  # full torchvision VGG19 state_dict
    content_layers: list[str] = field(default_factory=lambda: ["conv_4"])
    style_layers: list[str] = field(default_factory=lambda: [
        "conv_1", "conv_2", "conv_3", "conv_4", "conv_5",
    ])
    content_weight: float = 1.0
    style_weight: float = 1000.0
    steps: int = 1
    learning_rate: float = 0.001
    image_size: int = 600
    padding: int = 10
    tau: float = 0.99
    mask_threshold: float = 0.5
    dilation_kernel: int = 11
    erosion_kernel: int = 11
    smoothing_kernel: int = 51
    final_dilation_kernel: int = 11


@dataclass
class Config:
    input_image: str = "../image.jpg"
    output_dir: str = "../outputs/experiment"
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    defense: DefenseConfig = field(default_factory=DefenseConfig)

    def to_dict(self):
        return asdict(self)


def _merge(defaults, supplied, prefix=""):
    if not isinstance(supplied, dict):
        raise ValueError(f"{prefix or 'config'} must be an object")
    result = copy.deepcopy(defaults)
    for key, value in supplied.items():
        if key not in defaults:
            raise ValueError(f"Unknown configuration key: {prefix}{key}")
        if isinstance(defaults[key], dict):
            result[key] = _merge(defaults[key], value, f"{prefix}{key}.")
        else:
            result[key] = value
    return result


def _integer(name, value, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _number(name, value, minimum=0, maximum=None, positive=False):
    if (type(value) not in (float, int) or not math.isfinite(value)
            or value < minimum or (positive and value == minimum)
            or (maximum is not None and value > maximum)):
        raise ValueError(f"Invalid {name}: {value!r}")


def validate(config: Config):
    r, a, d, detector = config.runtime, config.attack, config.defense, config.detector
    if r.seed is not None:
        _integer("runtime.seed", r.seed, 0)
        if r.seed >= 2**32:
            raise ValueError("runtime.seed must be < 2**32")
    if type(r.deterministic) is not bool:
        raise ValueError("runtime.deterministic must be a boolean")
    if not isinstance(r.device, str) or not r.device:
        raise ValueError("runtime.device must be a device string")
    _integer("runtime.num_threads", r.num_threads)
    for name in ("min_size", "max_size"):
        _integer(f"detector.{name}", getattr(detector, name))
    if detector.max_size < detector.min_size:
        raise ValueError("detector.max_size must be >= min_size")
    _number("detector.score_threshold", detector.score_threshold, maximum=1)
    for name in ("patch_size", "steps"):
        _integer(f"attack.{name}", getattr(a, name))
    _number("attack.learning_rate", a.learning_rate, positive=True)
    _number("attack.target_score_threshold", a.target_score_threshold, maximum=1)
    if a.position is not None:
        if not isinstance(a.position, list) or len(a.position) != 2:
            raise ValueError("attack.position must be null or [x, y]")
        for value in a.position:
            _integer("attack.position coordinate", value, 0)
    if a.target_selection not in ("all", "highest_score"):
        raise ValueError("attack.target_selection must be all or highest_score")
    if not ((isinstance(a.target_label, str) and a.target_label)
            or (type(a.target_label) is int and a.target_label > 0)):
        raise ValueError("attack.target_label must be a class name or positive class ID")
    if a.target_boxes is not None:
        if not isinstance(a.target_boxes, list) or not a.target_boxes:
            raise ValueError("attack.target_boxes must be null or a nonempty list")
        for box in a.target_boxes:
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError("Each target box must be [x1, y1, x2, y2]")
            for value in box:
                _number("attack.target_boxes coordinate", value)
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError("Target boxes must have positive width and height")
    for name, value in a.loss_weights.items():
        _number(f"attack.loss_weights.{name}", value)
    if not any(a.loss_weights.values()):
        raise ValueError("At least one attack loss weight must be positive")
    for name in ("steps", "image_size"):
        _integer(f"defense.{name}", getattr(d, name))
    _integer("defense.padding", d.padding, 0)
    if 2 * d.padding >= d.image_size:
        raise ValueError("defense.image_size must be greater than twice the padding")
    for name in ("dilation_kernel", "erosion_kernel", "smoothing_kernel", "final_dilation_kernel"):
        value = getattr(d, name)
        _integer(f"defense.{name}", value)
        if value % 2 != 1:
            raise ValueError(f"defense.{name} must be odd")
    _number("defense.tau", d.tau, maximum=1, positive=True)
    _number("defense.mask_threshold", d.mask_threshold, maximum=1)
    _number("defense.learning_rate", d.learning_rate, positive=True)
    for name in ("content_weight", "style_weight"):
        _number(f"defense.{name}", getattr(d, name))
    for name in ("content_layers", "style_layers"):
        layers = getattr(d, name)
        if (not isinstance(layers, list) or not layers
                or not all(isinstance(layer, str) for layer in layers)
                or len(set(layers)) != len(layers)):
            raise ValueError(f"defense.{name} must contain unique layer names")
    for section in (detector, d):
        if section.weights is not None and not isinstance(section.weights, str):
            raise ValueError("Model weights must be a weight enum name or null")
        if section.checkpoint is not None and section.weights is not None:
            raise ValueError("Set weights to null when supplying a local checkpoint")


def load_config(path, *, seed_override=None, override_seed=False, output_dir=None):
    path = Path(path).resolve()
    data = _merge(Config().to_dict(), json.loads(path.read_text()))
    if override_seed:
        data["runtime"]["seed"] = seed_override
    if output_dir is not None:
        # CLI paths are relative to the working directory; JSON paths to its file.
        data["output_dir"] = str(Path(output_dir).resolve())
    for section, key in ((data, "input_image"), (data, "output_dir"),
                         (data["runtime"], "cache_dir"),
                         (data["detector"], "checkpoint"),
                         (data["defense"], "checkpoint")):
        value = section[key]
        if value is not None:
            if not isinstance(value, str) or not value:
                raise ValueError(f"{key} must be a nonempty path string")
            section[key] = str((path.parent / Path(value).expanduser()).resolve())
    if data["input_image"] is None or data["output_dir"] is None:
        raise ValueError("input_image and output_dir cannot be null")
    config = Config(
        input_image=data["input_image"], output_dir=data["output_dir"],
        runtime=RuntimeConfig(**data["runtime"]), detector=DetectorConfig(**data["detector"]),
        attack=AttackConfig(**data["attack"]), defense=DefenseConfig(**data["defense"]),
    )
    validate(config)
    return config
