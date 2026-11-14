"""AntiStyle optimization followed by difference filtering and mask enhancement.

Preserves the notebook's two Adam updates per iteration, unnormalized VGG
inputs, kth-value threshold, and masking of the AntiStyled image.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms.functional import resize

from .images import validate_image


def gram_matrix(features):
    batch, channels, height, width = features.shape
    flattened = features.reshape(batch * channels, height * width)
    return flattened @ flattened.t() / (batch * channels * height * width)


class ContentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("target", None)
        self.loss = None

    def forward(self, features):
        if self.target is not None:
            self.loss = F.mse_loss(features, self.target)
        return features


class StyleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("target", None)
        self.loss = None

    def forward(self, features):
        if self.target is not None:
            self.loss = F.mse_loss(gram_matrix(features), self.target)
        return features


class AntiStyle:
    def __init__(self, backbone, config):
        self.config = config
        self.model = nn.Sequential()
        self.content_losses = []
        self.style_losses = []
        selected = set(config.content_layers + config.style_layers)
        seen = set()
        convolution = 0
        for layer in backbone.children():
            if isinstance(layer, nn.Conv2d):
                convolution += 1
                name = f"conv_{convolution}"
            elif isinstance(layer, nn.ReLU):
                name = f"relu_{convolution}"
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f"pool_{convolution}"
            elif isinstance(layer, nn.BatchNorm2d):
                name = f"bn_{convolution}"
            else:
                raise ValueError(f"Unsupported backbone layer: {type(layer).__name__}")
            self.model.add_module(name, layer)
            if name in config.content_layers:
                loss = ContentLoss()
                self.model.add_module(f"content_loss_{name}", loss)
                self.content_losses.append(loss)
            if name in config.style_layers:
                loss = StyleLoss()
                self.model.add_module(f"style_loss_{name}", loss)
                self.style_losses.append(loss)
            seen.add(name)
            if selected <= seen:
                break
        if selected - seen:
            raise ValueError(f"Unknown backbone loss layers: {sorted(selected - seen)}")
        self.model.eval().requires_grad_(False)

    @torch.no_grad()
    def update_targets(self, content, style):
        for layer in self.model.children():
            if isinstance(layer, ContentLoss):
                layer.target = content.detach()
            elif isinstance(layer, StyleLoss):
                layer.target = gram_matrix(style).detach()
            else:
                content, style = layer(content), layer(style)

    def apply(self, content, style, initial):
        self.update_targets(content.detach(), style.detach())
        image = initial.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([image], lr=self.config.learning_rate)
        history = []
        for step in range(self.config.steps):
            optimizer.zero_grad(set_to_none=True)
            self.model(image)
            anti_style_loss = -sum(item.loss for item in self.style_losses)
            if not torch.isfinite(anti_style_loss):
                raise RuntimeError(f"Non-finite style loss at step {step + 1}")
            anti_style_loss.backward()
            optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            self.model(image)
            content_loss = sum(item.loss for item in self.content_losses)
            style_loss = -sum(item.loss for item in self.style_losses)
            total = content_loss * self.config.content_weight + style_loss * self.config.style_weight
            if not torch.isfinite(total):
                raise RuntimeError(f"Non-finite defense loss at step {step + 1}")
            total.backward()
            optimizer.step()
            with torch.no_grad():
                image.clamp_(0, 1)
            history.append({
                "step": step + 1, "anti_style": anti_style_loss.item(),
                "content": content_loss.item(), "negative_style": style_loss.item(),
                "combined": total.item(),
            })
        # Do not retain the final computation graph between applications.
        for loss in self.content_losses + self.style_losses:
            loss.loss = None
        return image.detach(), history


def dilate(mask, kernel):
    return F.max_pool2d(mask, kernel, stride=1, padding=kernel // 2)


def erode(mask, kernel):
    return 1 - dilate(1 - mask, kernel)


@dataclass
class DefenseResult:
    image: torch.Tensor
    intermediates: dict[str, torch.Tensor]
    threshold: float
    losses: list[dict[str, float]]


class AntiStyler:
    def __init__(self, backbone, config):
        self.config = config
        self.anti_style = AntiStyle(backbone, config)

    def apply(self, image):
        validate_image(image)
        cfg = self.config
        size, padding = cfg.image_size, cfg.padding
        content = image.new_zeros((1, 3, size, size))
        if padding:
            # Draw on CPU as in the notebook, then transfer to the image device.
            content[:, :, :padding, :] = torch.rand((1, 3, padding, size)).to(image)
            content[:, :, -padding:, :] = torch.rand((1, 3, padding, size)).to(image)
            content[:, :, :, :padding] = torch.rand((1, 3, size, padding)).to(image)
            content[:, :, :, -padding:] = torch.rand((1, 3, size, padding)).to(image)
        interior = slice(padding, size - padding)
        content[:, :, interior, interior] = resize(
            image, [size - 2 * padding] * 2, antialias=True,
        )
        style = torch.randn_like(content)
        styled, history = self.anti_style.apply(content, style, content)
        difference = (styled - content).abs().mean(dim=1, keepdim=True)
        # kthvalue is 1-based; tau may otherwise round down to zero on tiny inputs.
        rank = max(1, int(cfg.tau * difference.numel()))
        threshold = torch.kthvalue(difference.flatten(), rank).values
        raw_mask = (difference >= threshold).to(image.dtype)[:, :, interior, interior]
        mask = erode(dilate(raw_mask, cfg.dilation_kernel), cfg.erosion_kernel)
        kernel = mask.new_ones((1, 1, cfg.smoothing_kernel, cfg.smoothing_kernel))
        kernel /= cfg.smoothing_kernel**2
        mask = F.conv2d(mask, kernel, padding=cfg.smoothing_kernel // 2)
        mask = dilate((mask >= cfg.mask_threshold).to(image.dtype), cfg.final_dilation_kernel)
        if mask.any():
            defended = resize(styled[:, :, interior, interior] * (1 - mask),
                              list(image.shape[-2:]), antialias=True)
        else:
            defended = image.detach().clone()
        return DefenseResult(defended, {
            "padded": content, "antistyled": styled, "difference": difference,
            "raw_mask": raw_mask, "mask": mask,
        }, threshold.item(), history)
