"""COCO AP/AR (pycocotools) and object-level attack success rate."""

from contextlib import redirect_stdout
import io

import numpy as np
import torch
from torchvision.ops import box_iou

from .yolo_data import pixel_boxes


def stat_names(max_dets):
    return ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
            *(f"AR_{count}" for count in max_dets), "AR_small", "AR_medium", "AR_large"]


def coco_ground_truth(samples, sizes, names):
    from pycocotools.coco import COCO

    images, annotations = [], []
    for image_id, (sample, (height, width)) in enumerate(zip(samples, sizes), 1):
        images.append({"id": image_id, "file_name": sample.image.name, "height": height, "width": width})
        boxes, classes = pixel_boxes(sample, height, width)
        for (x1, y1, x2, y2), label in zip(boxes.tolist(), classes.tolist()):
            annotations.append({
                "id": len(annotations) + 1, "image_id": image_id, "category_id": label + 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0,
            })
    coco = COCO()
    coco.dataset = {"images": images, "annotations": annotations, "categories": [
        {"id": index + 1, "name": name} for index, name in sorted(names.items())
    ]}
    with redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco


def coco_results(predictions):
    """Per-image prediction dictionaries to the COCO results format (1-based IDs)."""
    results = []
    for image_id, prediction in enumerate(predictions, 1):
        for (x1, y1, x2, y2), score, label in zip(prediction["boxes"].tolist(),
                                                  prediction["scores"].tolist(),
                                                  prediction["classes"].tolist()):
            results.append({"image_id": image_id, "category_id": label + 1,
                            "bbox": [round(v, 2) for v in (x1, y1, x2 - x1, y2 - y1)],
                            "score": round(score, 5)})
    return results


def coco_metrics(ground_truth, results, max_dets, class_ids=None):
    """COCO summary statistics plus per-class AP; class_ids restricts evaluation (0-based)."""
    from pycocotools.cocoeval import COCOeval

    names = stat_names(max_dets)
    categories = [i + 1 for i in class_ids] if class_ids is not None else ground_truth.getCatIds()
    if not results:
        return {"summary": dict.fromkeys(names, 0.0), "per_class_AP": {
            ground_truth.cats[category]["name"]: 0.0 for category in categories
        }}
    with redirect_stdout(io.StringIO()):
        evaluator = COCOeval(ground_truth, ground_truth.loadRes(results), "bbox")
        evaluator.params.maxDets = list(max_dets)
        evaluator.params.catIds = categories
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    per_class = {}
    for index, category in enumerate(evaluator.params.catIds):
        precision = evaluator.eval["precision"][:, :, index, 0, -1]
        valid = precision[precision > -1]
        name = ground_truth.cats[category]["name"]
        per_class[name] = float(valid.mean()) if valid.size else -1.0
    return {"summary": dict(zip(names, map(float, evaluator.stats))), "per_class_AP": per_class}


def detected(boxes, prediction, class_ids, confidence, iou):
    """Which ground-truth boxes overlap a confident prediction of the given classes (None: any class)."""
    keep = prediction["scores"] >= confidence
    if class_ids is not None:
        keep &= torch.isin(prediction["classes"], torch.tensor(class_ids))
    if not len(boxes) or not keep.any():
        return torch.zeros(len(boxes), dtype=torch.bool)
    return box_iou(boxes, prediction["boxes"][keep]).amax(1) >= iou


def occupied(boxes, prediction, confidence):
    """Which ground-truth boxes contain the center of a confident prediction of any class."""
    keep = prediction["scores"] >= confidence
    if not len(boxes) or not keep.any():
        return torch.zeros(len(boxes), dtype=torch.bool)
    predicted = prediction["boxes"][keep]
    centers = (predicted[:, :2] + predicted[:, 2:]) / 2
    return ((centers > boxes[:, None, :2]) & (centers < boxes[:, None, 2:])).all(2).any(1)


def attack_success(objects, clean, attacked):
    """ASR = clean-detected target objects that the attack hides / clean-detected objects."""
    hidden = clean & ~attacked
    result = {"objects": int(objects.numel()), "clean_detected": int(clean.sum()),
              "hidden": int(hidden.sum()), "newly_detected": int((~clean & attacked).sum())}
    # null (not NaN) when nothing was detected on clean images: strict JSON output.
    result["attack_success_rate"] = float(hidden.sum() / clean.sum()) if clean.any() else None
    return result


def summarize(conditions, ground_truth, target_ids, gt_boxes, patched, config, names):
    """Metrics for every condition; ASR is relative to the clean condition."""
    metrics = {}
    clean_hits = torch.cat([detected(boxes, prediction, target_ids, config.asr_conf, config.asr_iou)
                            for boxes, prediction in zip(gt_boxes, conditions["clean"])])
    patched = torch.cat(patched)
    for name, predictions in conditions.items():
        results = coco_results(predictions)
        hits = torch.cat([detected(boxes, prediction, target_ids, config.asr_conf, config.asr_iou)
                          for boxes, prediction in zip(gt_boxes, predictions)])
        confident = sum(int(((p["scores"] >= config.asr_conf)
                             & torch.isin(p["classes"], torch.tensor(target_ids))).sum())
                        for p in predictions)
        metrics[name] = {
            "coco_all_classes": coco_metrics(ground_truth, results, config.max_dets),
            "coco_target_classes": coco_metrics(ground_truth, results, config.max_dets, target_ids),
            "target_recall_at_asr_conf": float(hits.float().mean()) if hits.numel() else None,
            "target_predictions_at_asr_conf": confident,
        }
        if name != "clean":
            metrics[name]["asr_all_target_objects"] = attack_success(patched, clean_hits, hits)
            metrics[name]["asr_patched_objects"] = attack_success(
                patched[patched], clean_hits[patched], hits[patched])
            # Vanished: no confident prediction of any class matches the object (not just relabelled).
            seen = torch.cat([detected(boxes, prediction, None, config.asr_conf, config.asr_iou)
                              for boxes, prediction in zip(gt_boxes, predictions)])
            metrics[name]["vanish_patched_objects"] = attack_success(
                patched[patched], clean_hits[patched], seen[patched])
            # Strict: not even a small box of another class (e.g. a "pedestrian" on the patch) remains.
            present = torch.cat([occupied(boxes, prediction, config.asr_conf)
                                 for boxes, prediction in zip(gt_boxes, predictions)])
            metrics[name]["vanish_strict_patched_objects"] = attack_success(
                patched[patched], clean_hits[patched], present[patched])
    metrics["target_classes"] = [names[i] for i in target_ids]
    metrics["target_objects"] = int(patched.numel())
    metrics["patched_objects"] = int(patched.sum())
    return metrics


def format_summary(metrics, max_dets):
    rows = [c for c in metrics if isinstance(metrics[c], dict) and "coco_all_classes" in metrics[c]]
    last = f"AR_{max_dets[-1]}"
    lines = [f"{'condition':<10} {'AP':>6} {'AP50':>6} {last:>7} | {'tgt AP':>6} {'tgt AP50':>8} "
             f"{'tgt ' + last:>10} {'recall':>7} {'ASR':>6} {'ASR(p)':>7} {'vanish(p)':>9} {'strict(p)':>9}"]
    for name in rows:
        m = metrics[name]
        a, t = m["coco_all_classes"]["summary"], m["coco_target_classes"]["summary"]
        asr = m.get("asr_all_target_objects", {}).get("attack_success_rate")
        asr_p = m.get("asr_patched_objects", {}).get("attack_success_rate")
        vanish = m.get("vanish_patched_objects", {}).get("attack_success_rate")
        strict = m.get("vanish_strict_patched_objects", {}).get("attack_success_rate")
        recall = m["target_recall_at_asr_conf"]
        lines.append(f"{name:<10} {a['AP']:6.4f} {a['AP50']:6.4f} {a[last]:7.4f} | {t['AP']:6.4f} "
                     f"{t['AP50']:8.4f} {t[last]:10.4f} {np.nan if recall is None else recall:7.4f} "
                     f"{np.nan if asr is None else asr:6.4f} {np.nan if asr_p is None else asr_p:7.4f} "
                     f"{np.nan if vanish is None else vanish:9.4f} {np.nan if strict is None else strict:9.4f}")
    return "\n".join(lines)
