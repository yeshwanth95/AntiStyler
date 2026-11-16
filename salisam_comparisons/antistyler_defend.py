"""
AntiStyler on a SaliSAM attacked set, written as a SaliSAM run directory.

Defends every frame of --patch_imgdir (or of --imgdir with --clean) and writes
what salisam_fastsam.py writes, so SaliSAM's evaluators and
scripts/evaluate_attack_defense.sh score the run unmodified:

  <output_dir>/args.json            SaliSAM's schema; the victim settings are copied
                                    from the matching SaliSAM run (--victim_args)
  <output_dir>/recovered/<stem>.png the defended frame at its own HxW, PNG as
                                    salisam_fastsam.py's save_png rounds it

plus AntiStyler's own <output_dir>/masks/<stem>.png (the final mask at the
defense's working resolution) and per_image.json (mask fired, mask share,
threshold, seconds).

AntiStyler never runs the victim: it is detector-agnostic, and the victim only
enters when the run is scored. The RNGs are reset to the configured seed before
every frame, so a clean frame and its attacked twin get the same random border
and style, whatever --lim or the order.

Usage (from the repo root, in this repo's venv):
    python salisam_comparisons/antistyler_defend.py \\
        --imgdir ../SaliSAM/data/UCYHPC_inria_yolo11n_patch_vanish/clean \\
        --patch_imgdir ../SaliSAM/data/UCYHPC_inria_yolo11n_patch_vanish/all_instance \\
        --output_dir salisam_comparisons/outputs/AntiStyler_UCYHPC_inria_yolo11n_patch_vanish_all_instance_yolo11
"""

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import os
import shlex
import sys
import time

from common import REPO_ROOT, SALISAM_ROOT, VICTIM_KEYS, list_images, salisam_path

sys.path.insert(0, REPO_ROOT)

from antistyler.config import Config, DefenseConfig, RuntimeConfig, validate  # noqa: E402
from antistyler.runtime import seed_everything, setup_runtime  # noqa: E402

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'antistyler_inria.json')


def load_settings(path, output=None):
    """Runtime and defense sections of a config, paths relative to the file."""
    path = os.path.abspath(path)
    with open(path) as f:
        data = json.load(f)
    unknown = set(data) - {'runtime', 'defense'}
    if unknown:
        sys.exit(f"ERROR: unknown config sections {sorted(unknown)} in {path}")
    runtime = RuntimeConfig(**data.get('runtime', {}))
    if runtime.cache_dir is not None:
        runtime.cache_dir = os.path.normpath(os.path.join(os.path.dirname(path), runtime.cache_dir))
    defense = DefenseConfig(**{**data.get('defense', {}), **({'output': output} if output else {})})
    validate(Config(runtime=runtime, defense=defense))
    return runtime, defense


def save_png(image, path):
    """salisam_fastsam.py's save_png: (1, 3, H, W) in [0, 1] -> 8-bit RGB PNG."""
    import numpy as np
    from PIL import Image
    import torch

    arr = image[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8)
    Image.fromarray(np.ascontiguousarray(arr.numpy())).save(path, format='PNG')


def victim_args(path, dataset_root, variant):
    """The victim settings of the SaliSAM run this one is compared with."""
    path = path or os.path.join(SALISAM_ROOT, 'outputs',
                                f'SaliSAM_FastSAM_{os.path.basename(dataset_root)}_{variant}_yolo11',
                                'args.json')
    if not os.path.exists(path):
        sys.exit(f"ERROR: no SaliSAM run args at {path}; pass --victim_args")
    with open(path) as f:
        reference = json.load(f)['args']
    victim = {k: reference[k] for k in VICTIM_KEYS if k in reference}
    victim['yolo11_weights'] = salisam_path(victim['yolo11_weights'])
    return victim, os.path.abspath(path)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--imgdir', required=True, help='clean frames')
    p.add_argument('--patch_imgdir', required=True, help='attacked frames (a variant folder)')
    p.add_argument('--clean', action='store_true', help='defend --imgdir instead (for LPR)')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--config', default=DEFAULT_CONFIG, help='runtime and defense sections')
    p.add_argument('--defense_output', choices=('antistyled', 'masked_input'), default=None,
                   help="override the config's defense.output")
    p.add_argument('--victim_args', default=None,
                   help='args.json of the SaliSAM run to compare with (default: the '
                        'SaliSAM_FastSAM run of this dataset and variant)')
    p.add_argument('--lim', type=int, default=None, help='defend the first N frames only')
    args = p.parse_args()

    clean_dir = os.path.abspath(os.path.expanduser(args.imgdir))
    attacked_dir = os.path.abspath(os.path.expanduser(args.patch_imgdir))
    source = clean_dir if args.clean else attacked_dir
    output = os.path.abspath(os.path.expanduser(args.output_dir))
    if os.path.isdir(output) and os.listdir(output):
        sys.exit(f"ERROR: output directory is not empty: {output}")
    runtime, defense_config = load_settings(args.config, args.defense_output)
    variant = os.path.basename(attacked_dir)
    victim, reference = victim_args(args.victim_args, os.path.dirname(attacked_dir), variant)

    def stems(directory):
        return [os.path.splitext(f)[0] for f in list_images(directory)]

    if stems(clean_dir) != stems(attacked_dir):
        sys.exit(f"ERROR: {clean_dir} and {attacked_dir} hold different frames")
    files = list_images(source)
    files = files[:args.lim] if args.lim else files

    device = setup_runtime(runtime)
    import torch
    import torchvision
    from antistyler.defense import AntiStyler
    from antistyler.images import load_image
    from antistyler.models import load_backbone

    recovered_dir, masks_dir = os.path.join(output, 'recovered'), os.path.join(output, 'masks')
    os.makedirs(recovered_dir)
    os.makedirs(masks_dir)
    method = 'AntiStyler' if defense_config.output == 'antistyled' else 'AntiStyler-masked'
    with open(os.path.join(output, 'args.json'), 'w') as f:
        json.dump({
            'args': {**victim, 'imgdir': clean_dir, 'patch_imgdir': attacked_dir,
                     'n_patches': variant, 'clean': args.clean, 'output_dir': output,
                     'lim': args.lim, 'save_recovered': True},
            'method': method,
            'antistyler': {'runtime': asdict(runtime), 'defense': asdict(defense_config)},
            'victim_args_from': reference,
            'command': shlex.join([sys.executable] + sys.argv),
            'device': str(device),
            'versions': {'torch': torch.__version__, 'torchvision': torchvision.__version__},
            'detection_output': 'recovered_image_predictions',
            'geometry': 'native',
            'timestamp': datetime.now().astimezone().isoformat(timespec='seconds'),
        }, f, indent=2, sort_keys=True)

    seed_everything(runtime.seed)
    defense = AntiStyler(load_backbone(defense_config, device), defense_config)
    print(f"{method} on {len(files)} frames of {source}\n  -> {recovered_dir}", flush=True)

    records, start = {}, time.time()
    for i, fname in enumerate(files, 1):
        stem = os.path.splitext(fname)[0]
        tick = time.time()
        image = load_image(os.path.join(source, fname), device)
        seed_everything(runtime.seed)
        result = defense.apply(image)
        if result.image.shape != image.shape:
            sys.exit(f"ERROR: {fname}: defended shape {tuple(result.image.shape)} != input "
                     f"{tuple(image.shape)}")
        save_png(result.image, os.path.join(recovered_dir, stem + '.png'))
        mask = result.intermediates['mask']
        save_png(mask.expand(-1, 3, -1, -1), os.path.join(masks_dir, stem + '.png'))
        records[stem] = {'file': fname, 'size': list(image.shape[-2:][::-1]),
                         'fired': bool(mask.any()), 'mask_share': mask.mean().item(),
                         'threshold': result.threshold, 'seconds': round(time.time() - tick, 3)}
        if i % 10 == 0 or i == len(files):
            fired = sum(r['fired'] for r in records.values())
            print(f"  {i}/{len(files)}  fired {fired}  {(time.time() - start) / i:.2f} s/frame",
                  flush=True)

    with open(os.path.join(output, 'per_image.json'), 'w') as f:
        json.dump(records, f, indent=1)
    fired = sum(r['fired'] for r in records.values())
    print(f"Done: {len(records)} frames, mask fired on {fired}, "
          f"{time.time() - start:.0f} s. Saved to {output}", flush=True)


if __name__ == '__main__':
    main()
