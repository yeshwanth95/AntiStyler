"""CPU tests for the dataset patch attack; a randomly initialized YOLO, no downloads."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from antistyler.cli import patch_attack_main, prepare_dataset_main
from antistyler.config import RuntimeConfig
from antistyler.evaluation import (attack_success, coco_ground_truth, coco_metrics, coco_results, detected,
                                   occupied)
from antistyler.patch_config import AugmentConfig, PatchConfig, load_patch_config
from antistyler.patch_attack import check_dataset_names
from antistyler.patching import Placement, apply_patch, placements
from antistyler.runtime import setup_runtime
from antistyler.victim import letterbox, letterbox_boxes, load_victim, target_scores
from antistyler.yolo_data import Sample, load_split, read_image, write_image

ROOT = Path(__file__).resolve().parents[1]


def make_dataset(root, splits=("train", "val"), count=2):
    generator = torch.Generator().manual_seed(0)
    for split in splits:
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for index in range(count):
            write_image(root / split / "images" / f"{index}.png",
                        torch.rand((3, 96, 128), generator=generator))
            # Class IDs of the test victim: 2 car, 5 bus, 0 an untargeted class.
            (root / split / "labels" / f"{index}.txt").write_text(
                "2 0.3 0.4 0.25 0.3\n5 0.7 0.6 0.2 0.2\n0 0.5 0.5 0.01 0.01\n")


class PatchingTests(unittest.TestCase):
    def test_placements_center_cap_and_skip(self):
        config = PatchConfig(scale=0.5, max_box_fraction=0.5, min_side=3)
        places = placements(torch.tensor([[10.0, 10, 50, 30], [0, 0, 4, 4]]), config)
        # side = min(0.5 * sqrt(40 * 20), 0.5 * 20) = 10, centered on (30, 20).
        self.assertEqual(places[0], Placement(25, 15, 10))
        self.assertIsNone(places[1])

    def test_diagonal_reference(self):
        box = torch.tensor([[0.0, 0, 30, 40]])
        self.assertEqual(placements(box, PatchConfig(scale=0.2, reference="diagonal"))[0].side, 10)
        self.assertEqual(placements(box, PatchConfig(scale=2.0, reference="diagonal"))[0].side, 30)  # capped

    def test_apply_patch_is_local_clipped_and_differentiable(self):
        image = torch.zeros(3, 20, 20)
        patch = torch.ones(3, 8, 8, requires_grad=True)
        output = apply_patch(image, patch, [Placement(-2, 16, 6), None])
        self.assertTrue(torch.equal(output[:, 16:20, 0:4], torch.ones(3, 4, 4)))
        self.assertEqual(output.sum().item(), 3 * 16)
        self.assertEqual(image.sum().item(), 0)
        output.sum().backward()
        self.assertGreater(patch.grad.abs().sum().item(), 0)

    def test_augmented_application_replays_with_seed(self):
        image, patch = torch.rand(3, 40, 40), torch.rand(3, 8, 8)
        boxes = torch.tensor([[5.0, 5, 25, 20], [20, 20, 38, 39]])

        def draw(seed):
            generator = torch.Generator().manual_seed(seed)
            places = placements(boxes, PatchConfig(), AugmentConfig(), generator)
            return apply_patch(image, patch, places, AugmentConfig(), generator)
        self.assertTrue(torch.equal(draw(1), draw(1)))
        self.assertFalse(torch.equal(draw(1), draw(2)))

    def test_letterbox_matches_ultralytics(self):
        from ultralytics.data.augment import LetterBox

        image = (torch.rand(3, 75, 130) * 255).round().to(torch.uint8)
        bgr = image.permute(1, 2, 0).numpy()[:, :, ::-1].copy()
        expected = LetterBox((64, 64), auto=False)(image=bgr)[:, :, ::-1].astype(np.float32) / 255
        actual = letterbox(image.float() / 255, 64).permute(1, 2, 0).numpy()
        self.assertLessEqual(np.abs(expected - actual).max() * 255, 1.0)

    def test_letterbox_boxes_follow_the_image(self):
        image = torch.zeros(3, 75, 130)
        image[:, 20:50, 40:100] = 1
        boxed = letterbox(image, 64)
        x1, y1, x2, y2 = letterbox_boxes(torch.tensor([[40.0, 20, 100, 50]]), 75, 130, 64)[0].tolist()
        rows, columns = (boxed[0] > 0.5).nonzero().unbind(1)
        self.assertLessEqual(abs(columns.min().item() - x1), 1)
        self.assertLessEqual(abs(columns.max().item() + 1 - x2), 1)
        self.assertLessEqual(abs(rows.min().item() - y1), 1)
        self.assertLessEqual(abs(rows.max().item() + 1 - y2), 1)


class EvaluationTests(unittest.TestCase):
    def test_perfect_predictions_score_one_and_asr_counts(self):
        sample = Sample(Path("a.png"), Path("a.txt"), torch.tensor([0, 1]),
                        torch.tensor([[0.25, 0.25, 0.2, 0.2], [0.75, 0.75, 0.2, 0.2]], dtype=torch.float64))
        names = {0: "car", 1: "bus"}
        ground_truth = coco_ground_truth([sample], [(100, 100)], names)
        boxes = torch.tensor([[15.0, 15, 35, 35], [65, 65, 85, 85]])
        prediction = {"boxes": boxes, "scores": torch.tensor([0.9, 0.8]), "classes": torch.tensor([0, 1])}
        metrics = coco_metrics(ground_truth, coco_results([prediction]), [1, 10, 100])
        self.assertAlmostEqual(metrics["summary"]["AP"], 1.0)
        self.assertEqual(set(metrics["per_class_AP"]), {"car", "bus"})
        for value in metrics["per_class_AP"].values():
            self.assertAlmostEqual(value, 1.0)
        empty = coco_metrics(ground_truth, [], [1, 10, 100], [1])
        self.assertEqual(empty["per_class_AP"], {"bus": 0.0})
        clean = detected(boxes, prediction, [0, 1], 0.25, 0.5)
        hidden = detected(boxes, dict(prediction, scores=torch.tensor([0.9, 0.1])), [0, 1], 0.25, 0.5)
        self.assertEqual(clean.tolist(), [True, True])
        relabelled = dict(prediction, classes=torch.tensor([0, 7]))
        self.assertEqual(detected(boxes, relabelled, [0, 1], 0.25, 0.5).tolist(), [True, False])
        self.assertEqual(detected(boxes, relabelled, None, 0.25, 0.5).tolist(), [True, True])
        # A small confident box centered on the second object still occupies it; IoU matching misses it.
        small = {"boxes": torch.tensor([[73.0, 73, 77, 77]]), "scores": torch.tensor([0.9]),
                 "classes": torch.tensor([7])}
        self.assertEqual(detected(boxes, small, None, 0.25, 0.5).tolist(), [False, False])
        self.assertEqual(occupied(boxes, small, 0.25).tolist(), [False, True])
        self.assertEqual(occupied(boxes, dict(small, scores=torch.tensor([0.1])), 0.25).tolist(), [False, False])
        result = attack_success(torch.ones(2, dtype=torch.bool), clean, hidden)
        self.assertEqual((result["hidden"], result["attack_success_rate"]), (1, 0.5))


INRIA_ANNOTATION = """# PASCAL Annotation Version 1.00

Image filename : "Train/pos/crop_1.png"
Image size (X x Y x C) : 40 x 30 x 3
Database : "The INRIA Rh\xf4ne-Alpes Annotated Person Database"
Objects with ground truth : 2 { "PASperson" "PASperson" }
Bounding box for object 1 "PASperson" (Xmin, Ymin) - (Xmax, Ymax) : (0, 0) - (19, 29)
Bounding box for object 2 "PASperson" (Xmin, Ymin) - (Xmax, Ymax) : (20, 10) - (39, 19)
"""

VOC_ANNOTATION = """<annotation><filename>{name}.jpg</filename><size><width>40</width><height>30</height></size>
<object><name>person</name><difficult>0</difficult><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>20</xmax><ymax>30</ymax></bndbox></object>
<object><name>tvmonitor</name><difficult>0</difficult><bndbox><xmin>21</xmin><ymin>11</ymin><xmax>40</xmax><ymax>20</ymax></bndbox></object>
<object><name>person</name><difficult>1</difficult><bndbox><xmin>5</xmin><ymin>5</ymin><xmax>9</xmax><ymax>9</ymax></bndbox></object>
</annotation>"""


class DatasetConversionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.image = torch.rand((3, 30, 40), generator=torch.Generator().manual_seed(0))

    def tearDown(self):
        self.directory.cleanup()

    def test_inria_conversion(self):
        source = self.root / "INRIAPerson"
        for split in ("Train", "Test"):
            (source / split / "pos").mkdir(parents=True)
            (source / split / "annotations").mkdir()
            write_image(source / split / "pos/crop_1.png", self.image)
            (source / split / "annotations/crop_1.txt").write_bytes(
                INRIA_ANNOTATION.replace("Train/", f"{split}/").encode("latin-1"))
        self.assertEqual(prepare_dataset_main(["--dataset", "inria", "--source", str(source),
                                               "--output", str(self.root / "out")]), 0)
        for split in ("train", "test"):
            self.assertTrue(torch.equal(read_image(self.root / f"out/{split}/images/crop_1.png"),
                                        read_image(source / "Train/pos/crop_1.png")))
            self.assertEqual((self.root / f"out/{split}/labels/crop_1.txt").read_text(),
                             "0 0.250000 0.500000 0.500000 1.000000\n0 0.750000 0.500000 0.500000 0.333333\n")
        self.assertEqual(prepare_dataset_main(["--dataset", "inria", "--source", str(source),
                                               "--output", str(self.root / "out")]), 1)  # nonempty

    def test_voc_conversion_maps_classes_and_skips_difficult(self):
        source = self.root / "VOC2007"
        for directory in ("ImageSets/Main", "Annotations", "JPEGImages",
                          "Detection_splits/test_500/images", "Detection_splits/test_500/annotations"):
            (source / directory).mkdir(parents=True)
        for name, images, annotations in (("000001", "JPEGImages", "Annotations"),
                                          ("000002", "JPEGImages", "Annotations"),
                                          ("000003", "Detection_splits/test_500/images",
                                           "Detection_splits/test_500/annotations")):
            write_image(source / images / f"{name}.jpg", self.image)
            (source / annotations / f"{name}.xml").write_text(VOC_ANNOTATION.format(name=name))
        (source / "ImageSets/Main/train.txt").write_text("000001\n")
        (source / "ImageSets/Main/val.txt").write_text("000002\n")
        self.assertEqual(prepare_dataset_main(["--dataset", "voc2007", "--source", str(source),
                                               "--output", str(self.root / "out")]), 0)
        expected = "0 0.250000 0.500000 0.500000 1.000000\n62 0.750000 0.500000 0.500000 0.333333\n"
        for split, name in (("train", "000001"), ("val", "000002"), ("test_500", "000003")):
            self.assertEqual((self.root / f"out/{split}/labels/{name}.txt").read_text(), expected)
        summary = json.loads((self.root / "out/conversion.json").read_text())
        self.assertEqual(summary["splits"]["train"]["difficult_skipped"], 1)
        self.assertEqual(summary["splits"]["test_500"]["objects"], {"person": 1, "tv": 1})
        check_dataset_names(self.root / "out", __import__("antistyler.datasets").datasets.coco_names())
        with self.assertRaises(ValueError):
            check_dataset_names(self.root / "out", {0: "person"})


class PatchAttackWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup_runtime(RuntimeConfig(device="cpu"))
        from ultralytics import YOLO

        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        make_dataset(cls.root / "data")
        torch.manual_seed(0)
        model = YOLO("yolo11n.yaml")
        model.model.names = {i: {2: "car", 5: "bus"}.get(i, f"class{i}") for i in range(80)}
        model.save(cls.root / "victim.pt")

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def write_config(self, name, **changes):
        config = {
            "output_dir": f"runs/{name}", "victim": {"weights": "victim.pt", "imgsz": 64},
            "dataset": {"root": "data", "target_classes": ["car", "bus"]},
            "patch": {"size": 8, "scale": 0.5},
            "train": {"images": 2, "epochs": 1, "batch_size": 2},
            "output_dataset": {"dir": f"attacked/{name}"},
            "evaluation": {"previews": 1},
        }
        config.update(changes)
        path = self.root / f"{name}.json"
        path.write_text(json.dumps(config))
        return path

    def test_attack_writes_dataset_metrics_and_replays(self):
        for name in ("first", "replay"):
            self.assertEqual(patch_attack_main(["--config", str(self.write_config(name))]), 0)
        run, data = self.root / "runs/first", self.root / "attacked/first"
        metrics = json.loads((run / "metrics.json").read_text())
        self.assertEqual(set(metrics) >= {"clean", "control", "attacked"}, True)
        self.assertEqual(metrics["target_objects"], 4)
        self.assertIn("asr_patched_objects", metrics["attacked"])
        self.assertEqual(sorted(p.name for p in (data / "val/images").iterdir()), ["0.png", "1.png"])
        self.assertEqual((data / "val/labels/0.txt").read_text(),
                         (self.root / "data/val/labels/0.txt").read_text())
        self.assertTrue(torch.equal(torch.load(run / "patch.pt"),
                                    torch.load(self.root / "runs/replay/patch.pt")))
        # Unpatched pixels are the original image; patched pixels changed.
        original, attacked = read_image(self.root / "data/val/images/0.png"), read_image(data / "val/images/0.png")
        self.assertTrue(torch.equal(original[:, :10, :10], attacked[:, :10, :10]))
        self.assertFalse(torch.equal(original, attacked))
        self.assertEqual(len(load_split(data, "val")), 2)

    def test_vanish_scores_use_all_classes_on_objects(self):
        _, network = load_victim(self.root / "victim.pt", "cpu")
        batch = torch.rand((2, 3, 64, 64), generator=torch.Generator().manual_seed(1))
        objects = [torch.tensor([[8.0, 8, 40, 40]]), torch.zeros(0, 4)]
        output = network(batch)[0]
        target = output[:, 4:][:, [2, 5]].amax(1)
        scores = target_scores(network, batch, [2, 5], objects, object_iou=1.0)
        points = (network.model[-1].anchors * network.model[-1].strides).T
        inside = ((points > 8) & (points < 40)).all(1)
        self.assertTrue(inside.any())
        self.assertTrue(torch.equal(scores[0, inside], output[0, 4:, inside].amax(0)))
        self.assertTrue(torch.equal(scores[0, ~inside], target[0, ~inside]))
        self.assertTrue(torch.equal(scores[1], target[1]))
        self.assertTrue(torch.equal(target_scores(network, batch, [2, 5]), target))

    def test_vanish_attack_writes_variant_layout(self):
        config = self.write_config("vanish", train={"images": 2, "epochs": 1, "batch_size": 2,
                                                    "suppress": "all_classes_on_objects"},
                                   output_dataset={"dir": "attacked/vanish", "layout": "variants"})
        self.assertEqual(patch_attack_main(["--config", str(config)]), 0)
        split = self.root / "attacked/vanish/val"
        self.assertEqual(sorted(p.name for p in split.iterdir()), ["all_instance", "clean", "labels"])
        for variant in ("all_instance", "clean"):
            self.assertEqual(sorted(p.name for p in (split / variant).iterdir()), ["0.png", "1.png"])
        self.assertEqual((split / "labels/0.txt").read_text(), (self.root / "data/val/labels/0.txt").read_text())
        original = read_image(self.root / "data/val/images/0.png")
        self.assertTrue(torch.equal(read_image(split / "clean/0.png"), original))
        self.assertFalse(torch.equal(read_image(split / "all_instance/0.png"), original))
        self.assertFalse((self.root / "attacked/vanish/data.yaml").exists())
        metrics = json.loads((self.root / "runs/vanish/metrics.json").read_text())
        self.assertIn("vanish_patched_objects", metrics["attacked"])
        self.assertIn("vanish_strict_patched_objects", metrics["attacked"])

    def test_seed_override_and_saved_patch_reuse(self):
        self.assertEqual(patch_attack_main(["--config", str(self.write_config("seed42"))]), 0)
        changed = self.write_config("seed")
        self.assertEqual(patch_attack_main(["--config", str(changed), "--seed", "7"]), 0)
        self.assertEqual(json.loads((self.root / "runs/seed/metadata.json").read_text())["seed"], 7)
        self.assertFalse(torch.equal(torch.load(self.root / "runs/seed/patch.pt"),
                                     torch.load(self.root / "runs/seed42/patch.pt")))
        reuse = self.write_config("reuse", patch={"size": 8, "scale": 0.5, "init": "runs/seed/patch.pt"},
                                  train={"epochs": 0})
        self.assertEqual(patch_attack_main(["--config", str(reuse)]), 0)
        self.assertTrue(torch.equal(torch.load(self.root / "runs/reuse/patch.pt"),
                                    torch.load(self.root / "runs/seed/patch.pt")))
        self.assertEqual(patch_attack_main(["--config", str(reuse)]), 1)  # nonempty output

    def test_config_validation(self):
        config = load_patch_config(ROOT / "configs/attack_visdrone.json")
        self.assertTrue(Path(config.victim.weights).is_absolute())
        bad = self.write_config("bad", patch={"scale": 0})
        self.assertEqual(patch_attack_main(["--config", str(bad), "--validate-config"]), 1)
        for name, changes in (("bad_reference", {"patch": {"reference": "perimeter"}}),
                              ("bad_suppress", {"train": {"suppress": "everything"}}),
                              ("bad_layout", {"output_dataset": {"dir": "x", "layout": "flat"}})):
            self.assertEqual(patch_attack_main(["--config", str(self.write_config(name, **changes)),
                                                "--validate-config"]), 1)
        unknown = self.write_config("unknown", train={"steps": 3})
        self.assertEqual(patch_attack_main(["--config", str(unknown), "--validate-config"]), 1)


if __name__ == "__main__":
    unittest.main()
