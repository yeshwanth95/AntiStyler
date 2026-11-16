"""
Score a defense run (recovered/ + args.json) with SaliSAM's evaluators.

Run with SaliSAM's venv, so the evaluators see the packages SaliSAM pins. With
the victim settings recorded in <run>/args.json (as SaliSAM's
scripts/evaluate_attack_defense.sh reads them), this calls, unmodified:

  core/nets/yolo_tools/coco_eval.py        -> <out>/<run>/coco_eval_{clean,atk,rec}
                                              <out>/<run>_clean/coco_eval_rec  (D(clean))
  core/nets/yolo_tools/saliuitl_metrics.py -> <out>/metrics/<run>/saliuitl
  core/nets/yolo_tools/attack_success.py   -> <out>/metrics/<run>/per_target

The same script scores an antistyler_defend.py run (<out> defaults to the run's
parent directory) and rescores a salisam_fastsam.py run inside the SaliSAM
checkout (<out> defaults to salisam_comparisons/outputs/salisam_rescored), so
both methods go through identical evaluator settings.

Every evaluator runs with --batch 1. Ultralytics' predict letterboxes a batch
whose images all share one size onto a rectangle and any other batch onto a
square, so at SaliSAM's default batch 8 a frame's canvas -- and its detections --
depend on its neighbours (168 of the 288 UCYHPC INRIA frames get the rectangle,
the other 120 the square). At batch 1 every frame gets the rectangle. This is
the setting of the Saliuitl comparison (Saliuitl/salisam_comparisons), whose
SaliSAM rescore this one therefore reproduces.

--visualize also draws the victim's operating-point boxes on the attacked and
recovered frames into <run>/visualizations/{attacked,cleaned}/, named and
coloured as salisam_fastsam.py names and colours them.

Usage (from the repo root):
    ../SaliSAM/.venv/bin/python salisam_comparisons/evaluate_antistyler.py \\
        --run salisam_comparisons/outputs/AntiStyler_UCYHPC_inria_yolo11n_patch_vanish_all_instance_yolo11
    ../SaliSAM/.venv/bin/python salisam_comparisons/evaluate_antistyler.py \\
        --run ../SaliSAM/outputs/SaliSAM_FastSAM_UCYHPC_inria_yolo11n_patch_vanish_all_instance_yolo11
"""

import argparse
import json
import os
import sys

from common import HERE, SALISAM_ROOT, list_images, salisam_path, use_salisam

use_salisam()

import attack_success  # noqa: E402  (SaliSAM's, via use_salisam)
import coco_eval  # noqa: E402
import saliuitl_metrics  # noqa: E402


def call(main, prog, argv, **kw):
    argv = [str(a) for a in argv]
    print(f"\n$ {prog} {' '.join(argv)}", flush=True)
    saved = sys.argv
    sys.argv = [prog] + argv
    try:
        main(**kw)
    finally:
        sys.argv = saved


def visualize(run, attacked_dir, weights, imgsz, conf, nms):
    """salisam_fastsam.py's visualizations/: attacked boxes red, recovered boxes green."""
    import numpy as np
    from PIL import Image
    import torch
    from core.visualization import draw_bboxes_on_image

    recovered = os.path.join(run, 'recovered')
    files = list_images(recovered)
    attacked = {os.path.splitext(f)[0]: f for f in list_images(attacked_dir)}
    model, _ = coco_eval.load_model(weights, None)
    for kind, directory, names, color in (
            ('attacked', attacked_dir, [attacked[os.path.splitext(f)[0]] for f in files], (255, 0, 0)),
            ('cleaned', recovered, files, (0, 255, 0))):
        out = os.path.join(run, 'visualizations', kind)
        os.makedirs(out, exist_ok=True)
        for fname, (w, h), dets in coco_eval.run_inference(model, directory, names, imgsz, conf,
                                                           nms, 300, 1, None):
            boxes = [[(x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h, s, s, c]
                     for c, s, (x, y, bw, bh) in dets]
            with Image.open(os.path.join(directory, fname)) as img:
                arr = np.asarray(img.convert('RGB'), dtype=np.float32) / 255
            tensor = torch.from_numpy(arr).permute(2, 0, 1)
            draw_bboxes_on_image(tensor, boxes, color=color, thickness=2, conf_threshold=conf).save(
                os.path.join(out, f"{os.path.splitext(fname)[0]}_{kind}_boxes.png"))
        print(f"Saved: {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--run', required=True, help='a run directory with args.json and recovered/')
    p.add_argument('--clean_run', default=None, help='the --clean run (default: <run>_clean)')
    p.add_argument('--labels', default=None, help='YOLO labels (default: <dataset root>/labels)')
    p.add_argument('--patch_geometry', default=None,
                   help='default: <dataset root>/patch_geometry.json')
    p.add_argument('--resolve_classes', type=int, nargs='*', default=None,
                   help='attack_success.py --resolve_classes (default: 0 for inria, any class otherwise)')
    p.add_argument('--target_class', type=int, default=3,
                   help="attack_success.py --target_class (evaluate_attack_defense.sh's 3; "
                        "ignored for derived geometry)")
    p.add_argument('--out_root', default=None,
                   help='holds <run>/coco_eval_* and metrics/<run>/ (see the module docstring)')
    p.add_argument('--skip', nargs='*', default=[], choices=('coco', 'saliuitl', 'per_target'),
                   help='metric families to skip')
    p.add_argument('--visualize', action='store_true',
                   help='draw operating-point boxes into <run>/visualizations/')
    args = p.parse_args()

    run = os.path.normpath(os.path.abspath(os.path.expanduser(args.run)))
    clean_run = os.path.normpath(os.path.abspath(os.path.expanduser(args.clean_run or run + '_clean')))
    name = os.path.basename(run)
    with open(os.path.join(run, 'args.json')) as f:
        a = json.load(f)['args']
    if a.get('clean'):
        sys.exit("ERROR: --run is a --clean run; pass the attacked run (its _clean twin is found)")
    v = a['victim']
    if v not in ('yolo11', 'yolo26'):
        sys.exit(f"ERROR: victim '{v}' is not an Ultralytics model")
    if not os.path.isdir(os.path.join(run, 'recovered')):
        sys.exit(f"ERROR: {run} has no recovered/")

    clean_dir, attacked_dir = salisam_path(a['imgdir']), salisam_path(a['patch_imgdir'])
    weights = salisam_path(a[f'{v}_weights'])
    imgsz, conf, nms = a['input_size'] or 416, a[f'{v}_conf'], a.get(f'{v}_iou', 0.7)
    root, variant = os.path.dirname(attacked_dir), os.path.basename(attacked_dir)
    labels = os.path.expanduser(args.labels or os.path.join(root, 'labels'))
    geometry = os.path.expanduser(args.patch_geometry or os.path.join(root, 'patch_geometry.json'))
    resolve = args.resolve_classes
    if resolve is None:
        resolve = [0] if a['dataset'] == 'inria' else []
    salisam_run = os.path.commonpath([os.path.realpath(run), SALISAM_ROOT]) == SALISAM_ROOT
    out_root = os.path.expanduser(args.out_root or (
        os.path.join(HERE, 'outputs', 'salisam_rescored') if salisam_run else os.path.dirname(run)))
    coco_dir = os.path.join(out_root, name)
    coco_clean_dir = os.path.join(out_root, os.path.basename(clean_run))
    metrics_out = os.path.join(out_root, 'metrics', name)

    defended = os.path.join(run, 'recovered')
    defended_clean = os.path.join(clean_run, 'recovered')
    if not os.path.isdir(defended_clean):
        print(f"NOTE: no defended clean run at {clean_run}; LPR and D(clean) AP will be n/a")
        defended_clean = None
    n_images = len(list_images(defended))

    print(f"Run:      {run}")
    print(f"Victim:   {weights} @ {imgsz}, conf {conf}, NMS {nms}, batch 1")
    print(f"Data:     {root}  variant {variant}  ({n_images} defended frames)")
    print(f"Output:   {coco_dir}/coco_eval_*  {metrics_out}/")
    common_det = ['--weights', weights, '--imgsz', imgsz, '--batch', 1]

    if 'coco' not in args.skip:
        limit = ['--limit', n_images] if n_images < len(list_images(clean_dir)) else []
        jobs = [(clean_dir, os.path.join(coco_dir, 'coco_eval_clean')),
                (attacked_dir, os.path.join(coco_dir, 'coco_eval_atk')),
                (defended, os.path.join(coco_dir, 'coco_eval_rec'))]
        if defended_clean:
            jobs.append((defended_clean, os.path.join(coco_clean_dir, 'coco_eval_rec')))
        for images, output in jobs:
            call(coco_eval.main, 'evaluate_coco.py',
                 ['--images', images, '--labels', labels, *common_det, '--output', output, *limit],
                 default_weights=weights)

    if 'saliuitl' not in args.skip:
        call(saliuitl_metrics.main, 'saliuitl_metrics.py',
             ['--clean_dir', clean_dir, '--attacked_dir', attacked_dir, '--defended_dir', defended,
              *(['--defended_clean_dir', defended_clean] if defended_clean else []),
              *common_det, '--conf', conf, '--iou', nms,
              '--output', os.path.join(metrics_out, 'saliuitl')],
             default_weights=weights)

    if 'per_target' not in args.skip:
        call(attack_success.main, 'attack_success.py',
             ['--dataset', root, '--variant', variant, '--patch_geometry', geometry,
              '--score_dir', defended,
              *(['--defended_clean_dir', defended_clean] if defended_clean else []),
              *(['--resolve_classes', *resolve] if resolve else []),
              '--target_class', args.target_class,
              *common_det, '--tau', conf, '--iou', nms,
              '--output', os.path.join(metrics_out, 'per_target')],
             default_weights=weights)

    if args.visualize:
        visualize(run, attacked_dir, weights, imgsz, conf, nms)


if __name__ == '__main__':
    main()
