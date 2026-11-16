"""CPU tests using small models; no pretrained weights or network required."""

import copy
from dataclasses import replace
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
import torchvision

from antistyler.attack import build_target, generate_attack, patch_position
from antistyler.cli import main
from antistyler.config import AttackConfig, Config, DefenseConfig, RuntimeConfig, load_config, validate
from antistyler.defense import AntiStyler
from antistyler.images import load_image
from antistyler.runtime import seed_everything, setup_runtime


ROOT = Path(__file__).resolve().parents[1]


class TinyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, images, targets=None):
        image = images[0]
        if self.training:
            assert len(targets[0]["boxes"]) == len(targets[0]["labels"])
            loss = ((image * self.scale - 0.25) ** 2).mean()
            # Also exercise the detector's stochastic training operations.
            loss = loss * (0.5 + torch.rand((), device=image.device))
            return {name: loss * weight for name, weight in (
                ("loss_classifier", 1), ("loss_box_reg", 2),
                ("loss_objectness", 3), ("loss_rpn_box_reg", 4),
            )}
        return [{
            "boxes": image.new_tensor([[1, 1, 8, 8], [3, 3, 10, 12]]),
            "labels": torch.tensor([1, 2], device=image.device),
            "scores": image.new_tensor([0.9, 0.8]),
        }]


def tiny_backbone():
    layers = []
    for index in range(5):
        layers.extend([nn.Conv2d(3 if index == 0 else 4, 4, 3, padding=1), nn.ReLU(inplace=True)])
    return nn.Sequential(*layers)


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup_runtime(RuntimeConfig(device="cpu"))

    def setUp(self):
        seed_everything(42)
        self.image = torch.rand(1, 3, 24, 32)
        self.attack_config = AttackConfig(patch_size=4, position=[5, 6], steps=2,
                                         target_boxes=[[1, 1, 8, 9], [3, 3, 10, 12]])
        self.defense_config = DefenseConfig(image_size=32, padding=3, steps=2,
                                           dilation_kernel=3, erosion_kernel=3,
                                           smoothing_kernel=3, final_dilation_kernel=3)

    def test_seed_replays_all_rngs_and_random_seed_is_recordable(self):
        def draws():
            return random.random(), np.random.rand(), torch.rand(3)
        seed_everything(12)
        a = draws()
        seed_everything(12)
        b = draws()
        self.assertEqual(a[:2], b[:2])
        self.assertTrue(torch.equal(a[2], b[2]))
        runtime = RuntimeConfig(seed=None, device="cpu")
        setup_runtime(runtime)
        self.assertIsInstance(runtime.seed, int)
        a = draws()
        seed_everything(runtime.seed)
        b = draws()
        self.assertEqual(a[:2], b[:2])
        self.assertTrue(torch.equal(a[2], b[2]))

    def test_attack_only_changes_patch_and_replays(self):
        detector = TinyDetector().eval()
        initial = copy.deepcopy(detector.state_dict())
        seed_everything(20)
        first = generate_attack(self.image, detector, self.attack_config, ["background", "person"])
        seed_everything(20)
        second = generate_attack(self.image, detector, self.attack_config, ["background", "person"])
        self.assertTrue(torch.equal(first.image, second.image))
        self.assertEqual(first.losses, second.losses)
        seed_everything(21)
        different = generate_attack(self.image, detector, self.attack_config, ["background", "person"])
        self.assertFalse(torch.equal(first.patch, different.patch))
        unchanged = torch.ones_like(self.image, dtype=torch.bool)
        unchanged[:, :, 6:10, 5:9] = False
        self.assertTrue(torch.equal(first.image[unchanged], self.image[unchanged]))
        self.assertEqual(len(first.target["labels"]), 2)
        self.assertTrue(first.image.min() >= 0 and first.image.max() <= 1)
        self.assertFalse(detector.training)
        self.assertTrue(detector.scale.requires_grad)
        self.assertIsNone(detector.scale.grad)
        for key, value in detector.state_dict().items():
            self.assertTrue(torch.equal(value, initial[key]))

    def test_attack_rejects_out_of_bounds_and_empty_targets(self):
        with self.assertRaisesRegex(ValueError, "exceeds image"):
            patch_position(self.image, replace(self.attack_config, position=[30, 22]))
        detector = TinyDetector().eval()
        detections = detector(list(self.image))[0]
        config = replace(self.attack_config, target_boxes=None, target_score_threshold=0.95)
        with self.assertRaisesRegex(ValueError, "No clean detections"):
            build_target(self.image, config, ["background", "person"], detections)
        config.target_score_threshold = 0
        config.target_selection = "highest_score"
        target = build_target(self.image, config, ["background", "person"], detections)
        self.assertTrue(torch.equal(target["boxes"], detections["boxes"][:1]))

    def test_attack_restores_detector_on_failure(self):
        detector = TinyDetector().eval()
        with patch.object(detector, "forward", side_effect=RuntimeError("failure")):
            with self.assertRaisesRegex(RuntimeError, "failure"):
                generate_attack(self.image, detector, self.attack_config, ["background", "person"])
        self.assertFalse(detector.training)
        self.assertTrue(detector.scale.requires_grad)

    def test_defense_replays_and_zero_padding_works(self):
        defense = AntiStyler(tiny_backbone(), self.defense_config)
        seed_everything(10)
        first = defense.apply(self.image)
        seed_everything(10)
        second = defense.apply(self.image)
        self.assertTrue(torch.equal(first.image, second.image))
        self.assertEqual(first.losses, second.losses)
        for key in first.intermediates:
            self.assertTrue(torch.equal(first.intermediates[key], second.intermediates[key]))
        seed_everything(11)
        third = defense.apply(self.image)
        self.assertFalse(torch.equal(first.intermediates["padded"], third.intermediates["padded"]))
        self.assertEqual(first.image.shape, self.image.shape)
        self.assertTrue(first.image.min() >= 0 and first.image.max() <= 1)
        defense = AntiStyler(tiny_backbone(), replace(self.defense_config, padding=0))
        result = defense.apply(self.image)
        self.assertEqual(result.intermediates["raw_mask"].shape, (1, 1, 32, 32))
        self.assertEqual(result.image.shape, self.image.shape)

    def test_defense_matches_original_notebook(self):
        # Compare against the actual notebook source, including intermediates,
        # to catch accidental changes to optimization order or mask processing.
        notebook = json.loads((ROOT / "AntiStyler_Demo.ipynb").read_text())
        source = "".join(next(cell["source"] for cell in notebook["cells"]
                              if "class ContentLoss" in "".join(cell["source"])))
        captured = {}
        namespace = {
            "torch": torch, "nn": nn, "F": F, "optim": torch.optim,
            "torchvision": torchvision, "transforms": torchvision.transforms,
            "imshow": lambda image, title: captured.update({title: image.detach().clone()}),
        }
        exec(compile(source, "AntiStyler_Demo.ipynb", "exec"), namespace)
        backbone = tiny_backbone()
        cfg = DefenseConfig(image_size=32, padding=3)
        original = namespace["AntiStyler"](
            copy.deepcopy(backbone), cfg.content_layers, cfg.style_layers,
            cfg.content_weight, cfg.style_weight, cfg.steps,
        )
        original.image_size, original.padding = cfg.image_size, cfg.padding
        converted = AntiStyler(backbone, cfg)
        seed_everything(77)
        # Keep the notebook's auto-device selection on CPU even on GPU hosts.
        original.anti_style.model.cpu()
        expected = original.apply(self.image, torch.device("cpu"))
        seed_everything(77)
        actual = converted.apply(self.image)
        torch.testing.assert_close(actual.image, expected, rtol=0, atol=0)
        for name, title in (("padded", "Padded Input Image"), ("antistyled", "AntiStyled Image")):
            torch.testing.assert_close(actual.intermediates[name], captured[title], rtol=0, atol=0)
        for name, title in (("raw_mask", "Raw Mask"), ("mask", "Final Mask")):
            torch.testing.assert_close(actual.intermediates[name].expand(-1, 3, -1, -1),
                                       captured[title], rtol=0, atol=0)

    def test_empty_mask_preserves_input(self):
        defense = AntiStyler(tiny_backbone(), self.defense_config)
        # A smoothing window larger than the whole mask cannot reach threshold 1.
        defense.config = replace(self.defense_config, smoothing_kernel=101, mask_threshold=1)
        result = defense.apply(self.image)
        self.assertFalse(result.intermediates["mask"].any())
        self.assertTrue(torch.equal(result.image, self.image))

    def test_masked_input_output_only_zeroes_masked_pixels(self):
        config = replace(self.defense_config, tau=0.7)  # a mask that survives the tiny kernels
        torch.manual_seed(0)
        backbone = tiny_backbone()
        seed_everything(5)
        notebook = AntiStyler(copy.deepcopy(backbone), config).apply(self.image)
        seed_everything(5)
        masked = AntiStyler(backbone, replace(config, output="masked_input"))
        result = masked.apply(self.image)
        self.assertTrue(result.intermediates["mask"].any())
        self.assertTrue(torch.equal(result.intermediates["mask"], notebook.intermediates["mask"]))
        self.assertEqual(result.image.shape, self.image.shape)
        kept = (result.image != 0).any(1)
        self.assertTrue(torch.equal(result.image[:, :, kept[0]], self.image[:, :, kept[0]]))
        self.assertTrue(0 < (~kept).sum() < kept.numel())
        masked.config = replace(masked.config, smoothing_kernel=101, mask_threshold=1)
        self.assertTrue(torch.equal(masked.apply(self.image).image, self.image))
        with self.assertRaisesRegex(ValueError, "defense.output"):
            validate(replace(Config(), defense=replace(self.defense_config, output="input")))

    def test_invalid_layers_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown backbone loss layers"):
            AntiStyler(tiny_backbone(), replace(self.defense_config, style_layers=["conv_99"]))

    def test_config_paths_defaults_and_typos(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"input_image": "sample.png", "runtime": {"seed": None}}))
            config = load_config(path)
            self.assertEqual(config.input_image, str(path.parent / "sample.png"))
            self.assertIsNone(config.runtime.seed)
            self.assertEqual(config.defense.style_weight, 1000)
            self.assertEqual(load_config(path, override_seed=True, seed_override=100).runtime.seed, 100)
            path.write_text('{"attack": {"stepz": 1}}')
            with self.assertRaisesRegex(ValueError, "Unknown configuration key"):
                load_config(path)
        for invalid in (replace(Config(), runtime=RuntimeConfig(seed=-1)),
                        replace(Config(), defense=DefenseConfig(padding=300)),
                        replace(Config(), defense=DefenseConfig(smoothing_kernel=2))):
            with self.assertRaises(ValueError):
                validate(invalid)

    def test_image_formats_and_exact_tensor_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            torch.save(self.image, directory / "image.pt")
            self.assertTrue(torch.equal(load_image(directory / "image.pt", "cpu"), self.image))
            Image.new("L", (8, 6), 128).save(directory / "gray.png")
            self.assertEqual(load_image(directory / "gray.png", "cpu").shape, (1, 3, 6, 8))
            np.save(directory / "float.npy", np.full((6, 8, 3), 0.5, dtype=np.float32))
            self.assertTrue((load_image(directory / "float.npy", "cpu") == 0.5).all())
            np.save(directory / "bad.npy", np.full((6, 8, 3), 128.0, dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "in \\[0, 1\\]"):
                load_image(directory / "bad.npy", "cpu")

    def test_all_entrypoints_and_standalone_pipeline_equivalence(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            image_path = directory / "image.pt"
            torch.save(self.image, image_path)
            config = Config(input_image=str(image_path), attack=self.attack_config,
                            defense=self.defense_config, runtime=RuntimeConfig(device="cpu"))
            config_path = directory / "config.json"
            config_path.write_text(json.dumps(config.to_dict()))
            with patch("antistyler.models.load_detector", side_effect=lambda *_: TinyDetector().eval()), \
                 patch("antistyler.models.load_backbone", side_effect=lambda *_: tiny_backbone()):
                for command in ("attack", "detect", "experiment"):
                    self.assertEqual(main(command, ["--config", str(config_path), "--output-dir",
                                                    str(directory / command)]), 0)
                config.input_image = str(directory / "attack" / "attacked.pt")
                config_path.write_text(json.dumps(config.to_dict()))
                self.assertEqual(main("defense", ["--config", str(config_path), "--output-dir",
                                                  str(directory / "defense")]), 0)
                for left, right in (("attack/attacked.pt", "experiment/attacked.pt"),
                                    ("defense/defended.pt", "experiment/defended_attacked.pt")):
                    self.assertTrue(torch.equal(torch.load(directory / left, weights_only=True),
                                                torch.load(directory / right, weights_only=True)))
                for command in ("attack", "defense", "detect", "experiment"):
                    output = directory / command
                    metadata = json.loads((output / "metadata.json").read_text())
                    self.assertEqual(metadata["seed"], 42)
                    self.assertEqual(json.loads((output / "status.json").read_text())["status"], "complete")
                    self.assertEqual(json.loads((output / "resolved_config.json").read_text())["runtime"]["seed"], 42)
                # Existing results must never be silently overwritten.
                self.assertEqual(main("defense", ["--config", str(config_path), "--output-dir",
                                                  str(directory / "defense")]), 1)


if __name__ == "__main__":
    unittest.main()
