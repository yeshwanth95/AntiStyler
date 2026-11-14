"""Image/tensor input and headless output, without notebook display globals."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch


def validate_image(image):
    if image.ndim != 4 or image.shape[:2] != (1, 3) or min(image.shape[2:]) < 1:
        raise ValueError("Expected one RGB image with shape [1, 3, height, width]")
    if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("Image values must be finite and in [0, 1]")


def load_image(path, device):
    path = Path(path)
    if path.suffix.lower() == ".pt":
        image = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(image, torch.Tensor):
            raise ValueError("A .pt image must contain a tensor, not a dictionary")
        image = image.to(dtype=torch.float32)
        if image.ndim == 3:
            image = image.unsqueeze(0)
    else:
        if path.suffix.lower() == ".npy":
            array = np.load(path, allow_pickle=False)
            if array.dtype == np.uint8:
                array = array.astype(np.float32) / 255
            elif np.issubdtype(array.dtype, np.floating):
                array = array.astype(np.float32)
            else:
                raise ValueError("NumPy images must be uint8 [0,255] or float [0,1]")
            if array.ndim == 2:
                array = np.repeat(array[:, :, None], 3, axis=2)
            if array.ndim != 3 or array.shape[2] not in (1, 3):
                raise ValueError("NumPy images must have shape HxW, HxWx1, or HxWx3")
            if array.shape[2] == 1:
                array = np.repeat(array, 3, axis=2)
        else:
            with Image.open(path) as source:
                array = np.asarray(source.convert("RGB"), dtype=np.float32) / 255
        image = torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0)
    validate_image(image)
    return image.to(device)


def to_pil(image):
    array = image.detach().cpu().squeeze(0).clamp(0, 1)
    if array.ndim == 3:
        array = array.permute(1, 2, 0)
        if array.shape[-1] == 1:
            array = array.squeeze(-1)
    return Image.fromarray((array.numpy() * 255).round().astype(np.uint8))


def save_tensor_image(directory, name, image):
    tensor = image.detach().cpu()
    torch.save(tensor, directory / f"{name}.pt")
    to_pil(tensor).save(directory / f"{name}.png")


def save_detections(directory, name, image, detections, categories, threshold):
    from .runtime import write_json

    write_json(directory / f"{name}_detections.json", {
        key: value.detach().cpu().tolist() for key, value in detections.items()
    })
    preview = to_pil(image)
    draw = ImageDraw.Draw(preview)
    for box, label, score in zip(detections["boxes"], detections["labels"], detections["scores"]):
        if score.item() < threshold:
            continue
        x1, y1, x2, y2 = box.detach().cpu().tolist()
        index = label.item()
        label_name = categories[index] if index < len(categories) else str(index)
        draw.rectangle((x1, y1, x2, y2), outline="red", width=2)
        draw.text((x1, max(0, y1 - 12)), f"{label_name}: {score.item():.2f}", fill="red")
    preview.save(directory / f"{name}_detections.png")
