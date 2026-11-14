"""Shared entry points; --help and config validation need only the standard library."""

import argparse
from pathlib import Path
import sys

from .config import load_config


DEFAULT_CONFIGS = {
    "attack": "attack.json", "defense": "defense.json",
    "detect": "detection.json", "experiment": "experiment.json",
}


def parse_seed(value):
    if value.lower() in ("random", "none", "null"):
        return None
    try:
        seed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Seed must be an integer or 'random'") from error
    if not 0 <= seed < 2**32:
        raise argparse.ArgumentTypeError("Seed must be in [0, 2**32)")
    return seed


def run(command, config):
    from .runtime import setup_runtime, seed_everything, write_json, write_manifest

    output = Path(config.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}. Choose a new --output-dir.")
    if not Path(config.input_image).is_file():
        raise ValueError(f"Input image does not exist: {config.input_image}")
    device = setup_runtime(config.runtime)

    from .attack import generate_attack, patch_position
    from .defense import AntiStyler
    from .images import load_image, save_detections, save_tensor_image
    from .models import class_names, detect, load_backbone, load_detector

    image = load_image(config.input_image, device)
    if command in ("attack", "experiment"):
        patch_position(image, config.attack)
    output.mkdir(parents=True, exist_ok=True)
    write_manifest(output, config, command, device)
    write_json(output / "status.json", {"status": "running"})
    print(f"Running {command} on {device}; seed={config.runtime.seed}; output={output}", flush=True)

    try:
        categories = class_names()
        write_json(output / "categories.json", categories)
        detector = None
        if command != "defense":
            print("Loading Faster R-CNN", flush=True)
            seed_everything(config.runtime.seed)
            detector = load_detector(config.detector, device)

        def evaluate(name, tensor):
            seed_everything(config.runtime.seed)
            result = detect(detector, tensor)
            save_detections(output, name, tensor, result, categories, config.detector.score_threshold)
            return result

        save_tensor_image(output, "input", image)
        if command == "detect":
            evaluate("input", image)
        if command in ("attack", "experiment"):
            benign = evaluate("benign", image)
            seed_everything(config.runtime.seed)
            attack = generate_attack(image, detector, config.attack, categories, benign)
            save_tensor_image(output, "attacked", attack.image)
            save_tensor_image(output, "patch", attack.patch)
            write_json(output / "attack.json", {
                "position": list(attack.position), "losses": attack.losses,
                "target": {key: value.cpu().tolist() for key, value in attack.target.items()},
            })
            evaluate("attacked", attack.image)
        if command in ("defense", "experiment"):
            print("Loading AntiStyler VGG19 backbone", flush=True)
            seed_everything(config.runtime.seed)
            defense = AntiStyler(load_backbone(config.defense, device), config.defense)
            inputs = [("defended", image)] if command == "defense" else [
                ("defended_benign", image), ("defended_attacked", attack.image),
            ]
            for name, source in inputs:
                print(f"Applying defense: {name}", flush=True)
                # Reset for each input so standalone defense and the full pipeline
                # use the same random padding/style, independent of prior stages.
                seed_everything(config.runtime.seed)
                result = defense.apply(source)
                save_tensor_image(output, name, result.image)
                stage_output = output / f"{name}_stages"
                stage_output.mkdir()
                for stage, tensor in result.intermediates.items():
                    save_tensor_image(stage_output, stage, tensor)
                write_json(stage_output / "optimization.json", {
                    "threshold": result.threshold, "losses": result.losses,
                })
                if detector is not None:
                    evaluate(name, result.image)
        write_json(output / "status.json", {"status": "complete"})
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "error": str(error)})
        raise
    print(f"Completed. Results saved to {output}", flush=True)
    return output


def patch_attack_main(argv=None):
    from .patch_config import load_patch_config

    parser = argparse.ArgumentParser(description="Train a universal patch on a YOLO-format dataset, "
                                     "write the attacked evaluation split, and evaluate it.")
    parser.add_argument("--config", type=Path, default=(
        Path(__file__).resolve().parents[1] / "configs" / "attack_visdrone.json"
    ), help="JSON configuration; file paths inside it are relative to the config file")
    parser.add_argument("--seed", type=parse_seed, default=argparse.SUPPRESS,
                        help="Override the configured seed with an integer or 'random'")
    parser.add_argument("--output-dir", type=Path, help="Override run directory (relative to cwd)")
    parser.add_argument("--dataset-dir", type=Path, help="Override attacked dataset directory (relative to cwd)")
    parser.add_argument("--validate-config", action="store_true", help="Validate config without loading models")
    args = parser.parse_args(argv)
    try:
        config = load_patch_config(args.config, seed_override=getattr(args, "seed", None),
                                   override_seed=hasattr(args, "seed"), output_dir=args.output_dir,
                                   dataset_dir=args.dataset_dir)
        if args.validate_config:
            print(f"Valid configuration: {args.config.resolve()}")
            return 0
        from .patch_attack import run_patch_attack

        run_patch_attack(config)
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


def prepare_dataset_main(argv=None):
    parser = argparse.ArgumentParser(description="Convert INRIAPerson or PASCAL VOC2007 to YOLO-format "
                                     "splits with COCO class indices.")
    parser.add_argument("--dataset", choices=("inria", "voc2007"), required=True)
    parser.add_argument("--source", type=Path, required=True,
                        help="INRIAPerson root, or the VOCdevkit/VOC2007 directory")
    parser.add_argument("--output", type=Path, required=True, help="New or empty output directory")
    args = parser.parse_args(argv)
    try:
        from .datasets import prepare_dataset

        prepare_dataset(args.dataset, args.source, args.output)
    except (ValueError, OSError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


def main(command, argv=None):
    parser = argparse.ArgumentParser(description=f"Run the AntiStyler {command} workflow.")
    parser.add_argument("--config", type=Path, default=(
        Path(__file__).resolve().parents[1] / "configs" / DEFAULT_CONFIGS[command]
    ), help="JSON configuration; file paths inside it are relative to the config file")
    parser.add_argument("--seed", type=parse_seed, default=argparse.SUPPRESS,
                        help="Override the configured seed with an integer or 'random'")
    parser.add_argument("--output-dir", type=Path, help="Override output directory (relative to cwd)")
    parser.add_argument("--validate-config", action="store_true", help="Validate config without loading models")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, seed_override=getattr(args, "seed", None),
                             override_seed=hasattr(args, "seed"), output_dir=args.output_dir)
        if args.validate_config:
            print(f"Valid configuration: {args.config.resolve()}")
            return 0
        run(command, config)
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0
