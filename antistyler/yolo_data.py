"""YOLO-format detection splits: <root>/<split>/images and matching labels/*.txt."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class Sample:
    image: Path
    label: Path
    classes: torch.Tensor  # [N] int64
    boxes: torch.Tensor  # [N, 4] normalized center-x, center-y, width, height


def read_labels(path):
    rows = []
    if path.is_file():
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if line.strip():
                values = line.split()
                if len(values) != 5:
                    raise ValueError(f"{path}:{number}: expected 'class cx cy w h'")
                rows.append([float(value) for value in values])
    rows = torch.tensor(rows, dtype=torch.float64).reshape(-1, 5)
    return rows[:, 0].long(), rows[:, 1:]


def load_split(root, split):
    directory = Path(root) / split
    images = sorted(p for p in (directory / "images").iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"No images found in {directory / 'images'}")
    samples = []
    for image in images:
        label = directory / "labels" / f"{image.stem}.txt"
        samples.append(Sample(image, label, *read_labels(label)))
    return samples


def read_image(path):
    """Decode exactly as Ultralytics does (cv2, BGR), returning RGB uint8 [3, H, W]."""
    array = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if array is None:
        raise ValueError(f"Could not read image: {path}")
    return torch.from_numpy(np.ascontiguousarray(array[:, :, ::-1])).permute(2, 0, 1)


def write_image(path, image):
    """Write a [3, H, W] RGB float [0,1] or uint8 tensor."""
    if image.dtype != torch.uint8:
        image = image.detach().clamp(0, 1).mul(255).round().to(torch.uint8)
    array = np.ascontiguousarray(image.cpu().permute(1, 2, 0).numpy()[:, :, ::-1])
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if Path(path).suffix.lower() in (".jpg", ".jpeg") else []
    if not cv2.imwrite(str(path), array, params):
        raise OSError(f"Could not write image: {path}")


def pixel_boxes(sample, height, width, class_ids=None):
    """Return labeled boxes as xyxy pixels, optionally only for the given class IDs."""
    keep = torch.ones(len(sample.classes), dtype=torch.bool)
    if class_ids is not None:
        keep = torch.isin(sample.classes, torch.tensor(sorted(class_ids)))
    cx, cy, w, h = (sample.boxes[keep] * torch.tensor([width, height, width, height])).unbind(1)
    boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, width)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, height)
    return boxes.float(), sample.classes[keep]
