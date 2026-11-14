"""Ultralytics YOLO victim: differentiable class scores and the official inference path."""

import torch
from torch.nn import functional as F
from torchvision.ops import box_convert, box_iou


def load_victim(weights, device):
    """Return (YOLO wrapper for official prediction, raw network for gradients)."""
    from ultralytics import YOLO

    # Separate instances: prediction fuses Conv+BN in place, the gradient path stays untouched.
    network = YOLO(weights).model.float().eval().requires_grad_(False).to(device)
    return YOLO(weights), network


def class_ids(names, wanted):
    lookup = {name.casefold(): index for index, name in names.items()}
    missing = [name for name in wanted if name.casefold() not in lookup]
    if missing:
        raise ValueError(f"Unknown target classes {missing}; victim classes are {list(names.values())}")
    return sorted(lookup[name.casefold()] for name in wanted)


def letterbox_geometry(height, width, imgsz):
    """(resized height, resized width, top, left) of Ultralytics' square letterbox."""
    ratio = min(imgsz / height, imgsz / width)
    new_height, new_width = max(1, round(height * ratio)), max(1, round(width * ratio))
    return new_height, new_width, round((imgsz - new_height) / 2 - 0.1), round((imgsz - new_width) / 2 - 0.1)


def letterbox(image, imgsz):
    """Differentiable square letterbox matching Ultralytics' cv2 INTER_LINEAR resize and 114 fill."""
    new_height, new_width, top, left = letterbox_geometry(*image.shape[-2:], imgsz)
    resized = F.interpolate(image[None], size=(new_height, new_width), mode="bilinear",
                            align_corners=False, antialias=False)[0]
    output = image.new_full((3, imgsz, imgsz), 114 / 255)
    output[:, top:top + new_height, left:left + new_width] = resized
    return output


def letterbox_boxes(boxes, height, width, imgsz):
    """Map xyxy pixel boxes of a height x width image into letterbox coordinates."""
    new_height, new_width, top, left = letterbox_geometry(height, width, imgsz)
    return boxes * boxes.new_tensor([new_width / width, new_height / height] * 2) + boxes.new_tensor([left, top] * 2)


def target_scores(network, batch, ids, objects=None, object_iou=0.3):
    """Per-anchor score to suppress, [B, anchors]: the maximum sigmoid score over the target classes.

    With objects (one [N, 4] xyxy letterbox tensor per image), anchors covering an object (anchor
    point inside it, or predicted box IoU >= object_iou) use the maximum over all classes instead,
    so the object has to vanish rather than be relabelled as another class.
    """
    output = network(batch)
    output = output[0] if isinstance(output, (list, tuple)) else output
    scores = output[:, 4:][:, ids].amax(1)
    if objects is None:
        return scores
    head = network.model[-1]
    points = (head.anchors * head.strides).T  # [anchors, 2] pixel centers, set by the forward pass
    predicted = box_convert(output[:, :4].detach().transpose(1, 2), "cxcywh", "xyxy")
    covered = torch.zeros_like(scores, dtype=torch.bool)
    for index, boxes in enumerate(objects):
        if len(boxes):
            inside = ((points[:, None] > boxes[:, :2]) & (points[:, None] < boxes[:, 2:])).all(2).any(1)
            covered[index] = inside | (box_iou(predicted[index], boxes).amax(1) >= object_iou)
    return torch.where(covered, output[:, 4:].amax(1), scores)


def predict(model, source, imgsz, config, device):
    """Official Ultralytics preprocessing, inference and NMS for one image path or BGR array."""
    result = model.predict(source, imgsz=imgsz, conf=config.conf, iou=config.iou,
                           max_det=config.max_det, device=str(device), verbose=False)[0]
    boxes = result.boxes
    return result, {
        "boxes": boxes.xyxy.detach().cpu().float(), "scores": boxes.conf.detach().cpu().float(),
        "classes": boxes.cls.detach().cpu().long(),
    }
