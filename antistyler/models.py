"""Explicit torchvision model/weight selection and local checkpoint support."""

import torch
from torchvision.models import VGG19_Weights, vgg19
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn


def _weights(enum, name):
    if name is None:
        return None
    if name == "DEFAULT":
        raise ValueError("Use an explicit weight version instead of the mutable DEFAULT alias")
    try:
        return enum[name]
    except KeyError as error:
        raise ValueError(f"Unknown weights {name!r}; choose from {[w.name for w in enum]}") from error


def _checkpoint(model, path):
    if path:
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)


def load_detector(config, device):
    model = fasterrcnn_resnet50_fpn(
        weights=_weights(FasterRCNN_ResNet50_FPN_Weights, config.weights),
        weights_backbone=None, min_size=config.min_size, max_size=config.max_size,
    )
    _checkpoint(model, config.checkpoint)
    model.eval().requires_grad_(False).to(device)
    return model


def load_backbone(config, device):
    model = vgg19(weights=_weights(VGG19_Weights, config.weights))
    _checkpoint(model, config.checkpoint)
    return model.features.eval().requires_grad_(False).to(device)


def class_names():
    return FasterRCNN_ResNet50_FPN_Weights.COCO_V1.meta["categories"]


@torch.no_grad()
def detect(model, image):
    model.eval()
    return model(list(image))[0]
