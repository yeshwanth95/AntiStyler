"""Differentiable placement of one universal patch on every target object."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass
class Placement:
    x: int  # top-left, may lie partly outside the image
    y: int
    side: int


def placements(boxes, config, augment=None, generator=None):
    """One square placement per xyxy box, centered on it; None for objects too small to patch."""
    result = []
    for x1, y1, x2, y2 in boxes.tolist():
        width, height = x2 - x1, y2 - y1
        length = (width * height) ** 0.5 if config.reference == "area" else (width ** 2 + height ** 2) ** 0.5
        side = min(config.scale * length, config.max_box_fraction * min(width, height))
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if augment is not None:
            jitter = (torch.rand(3, generator=generator) * 2 - 1).tolist()
            side *= 1 + augment.scale_jitter * jitter[0]
            cx += augment.position_jitter * side * jitter[1]
            cy += augment.position_jitter * side * jitter[2]
        side = round(side)
        result.append(Placement(round(cx - side / 2), round(cy - side / 2), side)
                      if side >= config.min_side else None)
    return result


def photometric(patch, augment, generator):
    contrast = 1 + augment.contrast * (torch.rand(1, generator=generator).item() * 2 - 1)
    brightness = augment.brightness * (torch.rand(1, generator=generator).item() * 2 - 1)
    noise = (torch.rand(patch.shape, generator=generator) * 2 - 1).to(patch) * augment.noise
    return (patch * contrast + brightness + noise).clamp(0, 1)


def apply_patch(image, patch, places, augment=None, generator=None):
    """Paste `patch` [3, S, S] onto a copy of `image` [3, H, W]; differentiable in both."""
    output = image.clone()
    height, width = image.shape[-2:]
    resized = {}
    for place in filter(None, places):
        if place.side not in resized:
            resized[place.side] = F.interpolate(
                patch[None], size=(place.side, place.side), mode="bilinear",
                align_corners=False, antialias=True,
            )[0]
        pasted = resized[place.side]
        if augment is not None:
            pasted = photometric(pasted, augment, generator)
        x1, y1 = max(place.x, 0), max(place.y, 0)
        x2, y2 = min(place.x + place.side, width), min(place.y + place.side, height)
        if x2 > x1 and y2 > y1:
            output[:, y1:y2, x1:x2] = pasted[:, y1 - place.y:y2 - place.y, x1 - place.x:x2 - place.x]
    return output


def total_variation(patch):
    return ((patch[:, 1:, :] - patch[:, :-1, :]).abs().mean()
            + (patch[:, :, 1:] - patch[:, :, :-1]).abs().mean())
