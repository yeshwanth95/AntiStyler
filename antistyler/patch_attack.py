"""Universal patch attack on a YOLO-format dataset: train, apply, evaluate.

A single patch is optimized on the training split to suppress the victim's
target-class scores (or, with train.suppress = all_classes_on_objects, every
class score on the patched objects) when pasted on every target object, then pasted (without
augmentation) on the evaluation split to create the attacked dataset. Clean,
control (uniform noise, same placements) and attacked images are evaluated with
Ultralytics' own inference pipeline.
"""

import gzip
import json
import math
from pathlib import Path
import shutil

import torch

from .evaluation import coco_ground_truth, coco_results, format_summary, summarize
from .patching import apply_patch, placements, total_variation
from .runtime import file_hash, seed_everything, write_json, write_provenance
from .victim import class_ids, letterbox, letterbox_boxes, load_victim, predict, target_scores
from .yolo_data import load_split, pixel_boxes, read_image, write_image


def initial_patch(config, generator):
    if config.init == "random":
        return torch.rand((3, config.size, config.size), generator=generator)
    if config.init == "gray":
        return torch.full((3, config.size, config.size), 0.5)
    patch = torch.load(config.init, map_location="cpu", weights_only=True).float()
    if patch.shape != (3, config.size, config.size) or patch.min() < 0 or patch.max() > 1:
        raise ValueError(f"patch.init must hold a [3, {config.size}, {config.size}] tensor in [0, 1]")
    return patch


def score_loss(scores, counts, config):
    if config.loss == "max":
        return scores.amax(1).mean()
    # Suppress every anchor that is still confidently a target, normalized per patched object.
    above = scores * (scores.detach() > config.score_threshold)
    return (above.sum(1) / torch.tensor(counts, device=scores.device).clamp(min=1)).mean()


def train_patch(network, samples, target_ids, config, device, output, generator):
    t = config.train
    patch = initial_patch(config.patch, generator)
    if t.epochs == 0:
        return patch
    candidates = [s for s in samples if torch.isin(s.classes, torch.tensor(target_ids)).any()]
    if t.images is not None:
        candidates = [candidates[i] for i in torch.randperm(len(candidates), generator=generator)[:t.images]]
    write_json(output / "train_images.json", [s.image.name for s in candidates])
    patch = patch.to(device).requires_grad_(True)
    optimizer = torch.optim.Adam([patch], lr=t.learning_rate)
    steps = t.epochs * math.ceil(len(candidates) / t.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps) if t.lr_schedule == "cosine" else None
    epochs_dir = output / "patch_epochs"
    epochs_dir.mkdir()
    step = 0
    with open(output / "train_log.jsonl", "w") as log:
        for epoch in range(1, t.epochs + 1):
            order = torch.randperm(len(candidates), generator=generator).tolist()
            totals = []
            for start in range(0, len(order), t.batch_size):
                images, counts, objects = [], [], []
                for index in order[start:start + t.batch_size]:
                    sample = candidates[index]
                    image = read_image(sample.image).to(device, torch.float32) / 255
                    boxes, _ = pixel_boxes(sample, *image.shape[-2:], target_ids)
                    places = placements(boxes, config.patch, t.augment, generator)
                    images.append(letterbox(apply_patch(image, patch, places, t.augment, generator),
                                            config.victim.imgsz))
                    counts.append(sum(place is not None for place in places))
                    patched = torch.tensor([place is not None for place in places], dtype=torch.bool)
                    objects.append(letterbox_boxes(boxes[patched], *image.shape[-2:], config.victim.imgsz)
                                   .to(device))
                vanish = t.suppress == "all_classes_on_objects"
                scores = target_scores(network, torch.stack(images), target_ids,
                                       objects if vanish else None, t.object_iou)
                detection = score_loss(scores, counts, t)
                smoothness = total_variation(patch)
                loss = detection + t.tv_weight * smoothness
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite patch loss at step {step + 1}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    patch.clamp_(0, 1)
                step += 1
                record = {"epoch": epoch, "step": step, "lr": optimizer.param_groups[0]["lr"],
                          "loss": loss.item(), "detection": detection.item(), "tv": smoothness.item(),
                          "max_score": scores.detach().amax(1).mean().item(), "objects": sum(counts)}
                if scheduler is not None:
                    scheduler.step()
                log.write(json.dumps(record) + "\n")
                log.flush()
                totals.append(detection.item())
                if step == 1 or step % 10 == 0 or step == steps:
                    print(f"Patch step {step}/{steps} (epoch {epoch}): detection={detection.item():.4f} "
                          f"tv={smoothness.item():.4f} max_score={record['max_score']:.3f}", flush=True)
            print(f"Epoch {epoch}: mean detection loss {sum(totals) / len(totals):.4f}", flush=True)
            torch.save(patch.detach().cpu(), epochs_dir / f"epoch_{epoch:03d}.pt")
    return patch.detach().cpu()


def save_patch(directory, name, patch):
    torch.save(patch, directory / f"{name}.pt")
    write_image(directory / f"{name}.png", patch)


def to_bgr(image):
    return image.clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).numpy()[:, :, ::-1].copy()


def attack_and_evaluate(predictor, samples, patch, control, target_ids, config, device, output, names):
    """Write the attacked split and run clean/control/attacked inference image by image."""
    split = config.dataset.eval_split
    variants = config.output_dataset.layout == "variants"
    images_dir = Path(config.output_dataset.dir) / split / ("all_instance" if variants else "images")
    clean_dir = Path(config.output_dataset.dir) / split / "clean" if variants else None
    labels_dir = Path(config.output_dataset.dir) / split / "labels"
    for directory in filter(None, (images_dir, clean_dir, labels_dir)):
        directory.mkdir(parents=True)
    previews_dir = output / "previews"
    previews_dir.mkdir()
    previews = set(torch.linspace(0, len(samples) - 1, config.evaluation.previews).round().long().tolist()
                   if config.evaluation.previews else [])
    conditions = {"clean": [], **({"control": []} if control is not None else {}), "attacked": []}
    sizes, gt_boxes, patched = [], [], []
    for index, sample in enumerate(samples):
        image = read_image(sample.image).float() / 255
        height, width = image.shape[-2:]
        boxes, _ = pixel_boxes(sample, height, width, target_ids)
        places = placements(boxes, config.patch)
        destination = images_dir / f"{sample.image.stem}.{config.output_dataset.image_format}"
        write_image(destination, apply_patch(image, patch, places))
        if clean_dir is not None:
            write_image(clean_dir / destination.name, image)
        if sample.label.is_file():
            shutil.copyfile(sample.label, labels_dir / sample.label.name)
        else:
            (labels_dir / f"{sample.image.stem}.txt").write_text("")
        sources = {"clean": str(sample.image), "attacked": str(destination)}
        if control is not None:
            sources["control"] = to_bgr(apply_patch(image, control, places))
        for name in conditions:
            result, prediction = predict(predictor, sources[name], config.victim.imgsz, config.evaluation, device)
            conditions[name].append(prediction)
            if index in previews:
                confident = result[result.boxes.conf >= config.evaluation.asr_conf]
                write_image(previews_dir / f"{sample.image.stem}_{name}.jpg",
                            torch.from_numpy(confident.plot()[:, :, ::-1].copy()).permute(2, 0, 1))
        sizes.append((height, width))
        gt_boxes.append(boxes)
        patched.append(torch.tensor([place is not None for place in places], dtype=torch.bool))
        if (index + 1) % 50 == 0 or index + 1 == len(samples):
            print(f"Attacked and evaluated {index + 1}/{len(samples)} images", flush=True)
    if not variants:  # Ultralytics cannot pair variant folders with the shared labels folder
        (Path(config.output_dataset.dir) / "data.yaml").write_text(
            f"path: {Path(config.output_dataset.dir)}\n{split}: {split}/images\n"
            f"nc: {len(names)}\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in sorted(names.items()))
        )
    for name, predictions in conditions.items():
        with gzip.open(output / f"predictions_{name}.json.gz", "wt") as stream:
            json.dump(coco_results(predictions), stream)
    return conditions, sizes, gt_boxes, patched


def check_dataset_names(root, names):
    """Fail early if the dataset's data.yaml numbers its classes differently from the victim."""
    path = Path(root) / "data.yaml"
    if not path.is_file():
        return
    from ultralytics.utils import YAML

    listed = YAML.load(path).get("names")
    if listed is None:
        return
    listed = dict(enumerate(listed)) if isinstance(listed, list) else {int(k): v for k, v in listed.items()}
    if listed != dict(names):
        raise ValueError(f"Class names in {path} do not match the victim's class names")


def run_patch_attack(config):
    from .runtime import setup_runtime

    output, dataset_dir = Path(config.output_dir), Path(config.output_dataset.dir)
    for directory in (output, dataset_dir):
        if directory.exists() and any(directory.iterdir()):
            raise ValueError(f"Directory is not empty: {directory}. Choose a new directory.")
    if not Path(config.victim.weights).is_file():
        raise ValueError(f"Victim weights do not exist: {config.victim.weights}")
    device = setup_runtime(config.runtime)
    train = load_split(config.dataset.root, config.dataset.train_split) if config.train.epochs else []
    evaluation = load_split(config.dataset.root, config.dataset.eval_split)
    if config.dataset.eval_images is not None and config.dataset.eval_images < len(evaluation):
        indices = torch.linspace(0, len(evaluation) - 1, config.dataset.eval_images).round().long()
        evaluation = [evaluation[i] for i in indices.tolist()]
    output.mkdir(parents=True, exist_ok=True)
    write_provenance(output, config, "patch_attack", device,
                     {"victim": file_hash(config.victim.weights)},
                     {"rng_policy": "One torch.Generator seeded with the run seed drives training-image "
                                    "sampling, shuffling, initialization and augmentation; the control "
                                    "patch uses a fresh generator with the same seed."})
    write_json(output / "status.json", {"status": "running"})
    print(f"Running patch attack on {device}; seed={config.runtime.seed}; output={output}", flush=True)
    try:
        seed_everything(config.runtime.seed)
        predictor, network = load_victim(config.victim.weights, device)
        names = predictor.names
        check_dataset_names(config.dataset.root, names)
        target_ids = class_ids(names, config.dataset.target_classes)
        write_json(output / "categories.json", {str(k): v for k, v in names.items()})
        print(f"Targets: {[names[i] for i in target_ids]}; {len(train)} train / "
              f"{len(evaluation)} evaluation images", flush=True)
        generator = torch.Generator().manual_seed(config.runtime.seed)
        patch = train_patch(network, train, target_ids, config, device, output, generator)
        save_patch(output, "patch", patch)
        control = None
        if config.evaluation.control:
            control = torch.rand(patch.shape, generator=torch.Generator().manual_seed(config.runtime.seed))
            save_patch(output, "control_patch", control)
        seed_everything(config.runtime.seed)
        conditions, sizes, gt_boxes, patched = attack_and_evaluate(
            predictor, evaluation, patch, control, target_ids, config, device, output, names)
        save_patch(dataset_dir, "patch", patch)
        write_json(dataset_dir / "attack_config.json", config.to_dict())
        ground_truth = coco_ground_truth(evaluation, sizes, names)
        metrics = summarize(conditions, ground_truth, target_ids, gt_boxes, patched, config.evaluation, names)
        write_json(output / "metrics.json", metrics)
        table = format_summary(metrics, config.evaluation.max_dets)
        (output / "summary.txt").write_text(table + "\n")
        print(table, flush=True)
        write_json(output / "status.json", {"status": "complete"})
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "error": str(error)})
        raise
    print(f"Completed. Results in {output}; attacked dataset in {dataset_dir}", flush=True)
    return output
