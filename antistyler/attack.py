"""The notebook's targeted patch attack, optimizing Faster R-CNN training losses."""

from dataclasses import dataclass

import torch
from torch import nn

from .images import validate_image


@dataclass
class AttackResult:
    image: torch.Tensor
    patch: torch.Tensor
    target: dict[str, torch.Tensor]
    position: tuple[int, int]
    losses: list[dict[str, float]]


def patch_position(image, config):
    height, width = image.shape[-2:]
    size = config.patch_size
    x, y = config.position if config.position is not None else (
        (width + size) // 2, (height + size) // 2,
    )
    if x < 0 or y < 0 or x + size > width or y + size > height:
        raise ValueError(
            f"Patch of size {size} at [x={x}, y={y}] exceeds image {width}x{height}. "
            "Set attack.position or reduce attack.patch_size."
        )
    return x, y


def build_target(image, config, categories, detections=None):
    label = config.target_label
    if isinstance(label, str):
        try:
            label = [name.casefold() for name in categories].index(label.casefold())
        except ValueError as error:
            raise ValueError(f"Unknown target label: {config.target_label!r}") from error
    if label <= 0 or label >= len(categories) or categories[label] == "N/A":
        raise ValueError("Target label must be a valid foreground class")
    if config.target_boxes is not None:
        boxes = torch.tensor(config.target_boxes, device=image.device, dtype=image.dtype)
    else:
        if detections is None:
            raise ValueError("Provide clean detections or explicit attack.target_boxes")
        keep = detections["scores"] >= config.target_score_threshold
        boxes = detections["boxes"][keep].detach().clone().to(image)
        if config.target_selection == "highest_score" and len(boxes):
            best = detections["scores"][keep].argmax().item()
            boxes = boxes[best:best + 1]
    if boxes.numel() == 0:
        raise ValueError(
            "No clean detections meet attack.target_score_threshold. "
            "Lower it or supply explicit attack.target_boxes."
        )
    height, width = image.shape[-2:]
    if (boxes.ndim != 2 or boxes.shape[1] != 4 or not torch.isfinite(boxes).all()
            or (boxes < 0).any() or (boxes[:, 2] > width).any()
            or (boxes[:, 3] > height).any() or (boxes[:, 2:] <= boxes[:, :2]).any()):
        raise ValueError("Target boxes must be valid xyxy boxes inside the input image")
    # The notebook supplied N boxes but only one label. Every box needs a label.
    labels = torch.full((len(boxes),), label, device=image.device, dtype=torch.int64)
    return {"boxes": boxes, "labels": labels}


def generate_attack(image, detector, config, categories, detections=None):
    validate_image(image)
    x, y = patch_position(image, config)
    target = build_target(image, config, categories, detections)
    size = config.patch_size
    patch = nn.Parameter(torch.rand((1, 3, size, size), device=image.device, dtype=image.dtype))
    optimizer = torch.optim.Adam([patch], lr=config.learning_rate)
    modes = [(module, module.training) for module in detector.modules()]
    flags = [(parameter, parameter.requires_grad) for parameter in detector.parameters()]
    losses = []
    try:
        detector.train()
        detector.requires_grad_(False)
        # Randomly initialized torchvision detectors have regular BatchNorm; do
        # not update detector statistics while optimizing only the patch.
        for module in detector.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        for step in range(config.steps):
            optimizer.zero_grad(set_to_none=True)
            attacked = image.detach().clone()
            attacked[:, :, y:y + size, x:x + size] = patch
            components = detector(list(attacked), [target])
            missing = set(config.loss_weights) - components.keys()
            if missing:
                raise ValueError(f"Detector did not return configured losses: {sorted(missing)}")
            loss = sum(components[name] * weight for name, weight in config.loss_weights.items())
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite attack loss at step {step + 1}")
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                patch.clamp_(0, 1)
            losses.append({"step": step + 1, "total": loss.item(), **{
                name: value.item() for name, value in components.items()
            }})
            if step == 0 or (step + 1) % 10 == 0 or step + 1 == config.steps:
                print(f"Attack {step + 1}/{config.steps}: loss={loss.item():.6f}", flush=True)
    finally:
        for parameter, flag in flags:
            parameter.requires_grad_(flag)
        for module, mode in modes:
            module.training = mode
    attacked = image.detach().clone()
    attacked[:, :, y:y + size, x:x + size] = patch.detach()
    return AttackResult(attacked, patch.detach(), target, (x, y), losses)
