"""Strict JSON configuration for universal patch attacks on YOLO-format datasets.

Uses only the standard library so --help and --validate-config stay lightweight.
Paths are relative to the configuration file, as in config.py.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

from .config import RuntimeConfig, _integer, _merge, _number


@dataclass
class VictimConfig:
    weights: str = "../models/yolo11/VisDrone_YOLO11m.pt"  # Ultralytics checkpoint
    imgsz: int = 640


@dataclass
class DatasetConfig:
    root: str = "../data/visdrone_yolo"  # contains <split>/images and <split>/labels
    train_split: str = "train"
    eval_split: str = "val"
    eval_images: int | None = None  # evenly spaced subset of the evaluation split; null uses all
    # Names must match the victim's class names.
    target_classes: list[str] = field(default_factory=lambda: ["car", "van", "truck", "bus"])


@dataclass
class PatchConfig:
    size: int = 64  # side of the learnable square patch
    scale: float = 0.5  # pasted side = scale * reference length of the box
    reference: str = "area"  # area: sqrt(width * height); diagonal: sqrt(width^2 + height^2)
    max_box_fraction: float = 1.0  # pasted side <= fraction of the shorter box side
    min_side: int = 2  # objects that would get a smaller patch are left unpatched
    init: str = "random"  # random, gray, or a saved .pt patch to start from


@dataclass
class AugmentConfig:
    position_jitter: float = 0.1  # center shift, as a fraction of the pasted side
    scale_jitter: float = 0.1  # relative change of the pasted side
    brightness: float = 0.1  # additive shift range
    contrast: float = 0.1  # multiplicative range around 1
    noise: float = 0.03  # uniform per-pixel noise amplitude


@dataclass
class TrainConfig:
    images: int | None = 1024  # sampled training images with targets; null uses all
    epochs: int = 5  # 0 skips optimization (useful with patch.init set to a saved patch)
    batch_size: int = 8
    learning_rate: float = 0.03
    lr_schedule: str = "cosine"  # constant or cosine
    loss: str = "sum_above_threshold"  # sum_above_threshold or max
    score_threshold: float = 0.1  # anchors whose target score exceeds this are suppressed
    # target_classes: suppress target-class scores (hidden vehicles may be relabelled as other classes).
    # all_classes_on_objects: on anchors covering a patched object, suppress every class (vanishing).
    suppress: str = "target_classes"
    object_iou: float = 0.3  # an anchor covers an object if its point is inside it or its box IoU >= this
    tv_weight: float = 2.5
    augment: AugmentConfig = field(default_factory=AugmentConfig)


@dataclass
class OutputDatasetConfig:
    dir: str = "../data/visdrone_yolo_patch"
    image_format: str = "png"  # png is lossless; jpg quantizes the patch further
    # yolo: <split>/images and <split>/labels. variants: <split>/all_instance (attacked),
    # <split>/clean (original pixels, same format) and one shared <split>/labels.
    layout: str = "yolo"


@dataclass
class EvaluationConfig:
    conf: float = 0.001  # prediction threshold for COCO AP/AR
    iou: float = 0.7  # NMS IoU
    max_det: int = 300
    max_dets: list[int] = field(default_factory=lambda: [1, 10, 100])  # COCOeval maxDets
    asr_conf: float = 0.25  # a target object is "detected" by a prediction at least this confident
    asr_iou: float = 0.5
    control: bool = True  # also evaluate a uniform-noise patch with the same placements
    previews: int = 4  # annotated clean/control/attacked examples to save


@dataclass
class PatchAttackConfig:
    output_dir: str = "../outputs/attack_visdrone"
    runtime: RuntimeConfig = field(default_factory=lambda: RuntimeConfig(num_threads=6))
    victim: VictimConfig = field(default_factory=VictimConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    patch: PatchConfig = field(default_factory=PatchConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output_dataset: OutputDatasetConfig = field(default_factory=OutputDatasetConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self):
        return asdict(self)


def validate(config: PatchAttackConfig):
    r, p, t, a, e = config.runtime, config.patch, config.train, config.train.augment, config.evaluation
    if r.seed is not None:
        _integer("runtime.seed", r.seed, 0)
        if r.seed >= 2**32:
            raise ValueError("runtime.seed must be < 2**32")
    if type(r.deterministic) is not bool:
        raise ValueError("runtime.deterministic must be a boolean")
    if not isinstance(r.device, str) or not r.device:
        raise ValueError("runtime.device must be a device string")
    _integer("runtime.num_threads", r.num_threads)
    _integer("victim.imgsz", config.victim.imgsz, 32)
    if config.victim.imgsz % 32:
        raise ValueError("victim.imgsz must be a multiple of 32")
    d = config.dataset
    for name in ("train_split", "eval_split"):
        if not isinstance(getattr(d, name), str) or not getattr(d, name):
            raise ValueError(f"dataset.{name} must be a nonempty string")
    if d.eval_images is not None:
        _integer("dataset.eval_images", d.eval_images)
    if (not isinstance(d.target_classes, list) or not d.target_classes
            or not all(isinstance(name, str) and name for name in d.target_classes)
            or len(set(d.target_classes)) != len(d.target_classes)):
        raise ValueError("dataset.target_classes must be a nonempty list of unique class names")
    _integer("patch.size", p.size)
    _number("patch.scale", p.scale, positive=True)
    if p.reference not in ("area", "diagonal"):
        raise ValueError("patch.reference must be area or diagonal")
    _number("patch.max_box_fraction", p.max_box_fraction, maximum=1, positive=True)
    _integer("patch.min_side", p.min_side)
    if not isinstance(p.init, str) or not p.init:
        raise ValueError("patch.init must be random, gray, or a patch path")
    if t.images is not None:
        _integer("train.images", t.images)
    _integer("train.epochs", t.epochs, 0)
    _integer("train.batch_size", t.batch_size)
    _number("train.learning_rate", t.learning_rate, positive=True)
    if t.lr_schedule not in ("constant", "cosine"):
        raise ValueError("train.lr_schedule must be constant or cosine")
    if t.loss not in ("sum_above_threshold", "max"):
        raise ValueError("train.loss must be sum_above_threshold or max")
    _number("train.score_threshold", t.score_threshold, maximum=1)
    if t.suppress not in ("target_classes", "all_classes_on_objects"):
        raise ValueError("train.suppress must be target_classes or all_classes_on_objects")
    _number("train.object_iou", t.object_iou, maximum=1, positive=True)
    _number("train.tv_weight", t.tv_weight)
    for name in ("position_jitter", "scale_jitter", "brightness", "contrast", "noise"):
        _number(f"train.augment.{name}", getattr(a, name), maximum=1)
    if config.output_dataset.image_format not in ("png", "jpg"):
        raise ValueError("output_dataset.image_format must be png or jpg")
    if config.output_dataset.layout not in ("yolo", "variants"):
        raise ValueError("output_dataset.layout must be yolo or variants")
    _number("evaluation.conf", e.conf, maximum=1)
    _number("evaluation.iou", e.iou, maximum=1, positive=True)
    _integer("evaluation.max_det", e.max_det)
    if (not isinstance(e.max_dets, list) or len(e.max_dets) != 3
            or any(type(v) is not int or v < 1 for v in e.max_dets) or e.max_dets != sorted(e.max_dets)):
        raise ValueError("evaluation.max_dets must be three increasing positive integers")
    _number("evaluation.asr_conf", e.asr_conf, maximum=1)
    _number("evaluation.asr_iou", e.asr_iou, maximum=1, positive=True)
    if type(e.control) is not bool:
        raise ValueError("evaluation.control must be a boolean")
    _integer("evaluation.previews", e.previews, 0)


def load_patch_config(path, *, seed_override=None, override_seed=False,
                      output_dir=None, dataset_dir=None):
    path = Path(path).resolve()
    data = _merge(PatchAttackConfig().to_dict(), json.loads(path.read_text()))
    if override_seed:
        data["runtime"]["seed"] = seed_override
    # CLI paths are relative to the working directory; JSON paths to its file.
    if output_dir is not None:
        data["output_dir"] = str(Path(output_dir).resolve())
    if dataset_dir is not None:
        data["output_dataset"]["dir"] = str(Path(dataset_dir).resolve())
    paths = [(data, "output_dir"), (data["runtime"], "cache_dir"), (data["victim"], "weights"),
             (data["dataset"], "root"), (data["output_dataset"], "dir")]
    if data["patch"]["init"] not in ("random", "gray"):
        paths.append((data["patch"], "init"))
    for section, key in paths:
        value = section[key]
        if value is not None:
            if not isinstance(value, str) or not value:
                raise ValueError(f"{key} must be a nonempty path string")
            section[key] = str((path.parent / Path(value).expanduser()).resolve())
    for section, key in ((data, "output_dir"), (data["victim"], "weights"),
                         (data["dataset"], "root"), (data["output_dataset"], "dir")):
        if section[key] is None:
            raise ValueError(f"{key} cannot be null")
    train = dict(data["train"], augment=AugmentConfig(**data["train"]["augment"]))
    config = PatchAttackConfig(
        output_dir=data["output_dir"], runtime=RuntimeConfig(**data["runtime"]),
        victim=VictimConfig(**data["victim"]), dataset=DatasetConfig(**data["dataset"]),
        patch=PatchConfig(**data["patch"]), train=TrainConfig(**train),
        output_dataset=OutputDatasetConfig(**data["output_dataset"]),
        evaluation=EvaluationConfig(**data["evaluation"]),
    )
    validate(config)
    return config
