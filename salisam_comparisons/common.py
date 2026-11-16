"""
Paths shared by the AntiStyler <-> SaliSAM comparison.

The defense (antistyler_defend.py) runs in this repo's venv; the scoring
(evaluate_antistyler.py) runs in SaliSAM's, so the evaluators see exactly the
packages SaliSAM pins. This module therefore imports nothing beyond the
standard library. The SaliSAM checkout defaults to the sibling directory
../SaliSAM; set SALISAM_ROOT to point elsewhere.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
SALISAM_ROOT = os.path.realpath(os.environ.get(
    'SALISAM_ROOT', os.path.join(REPO_ROOT, '..', 'SaliSAM')))
YOLO_TOOLS = os.path.join(SALISAM_ROOT, 'core', 'nets', 'yolo_tools')

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
DEFAULT_DATASET = os.path.join(SALISAM_ROOT, 'data/UCYHPC_inria_yolo11n_patch_vanish')

# The victim settings a run is scored with; antistyler_defend.py copies them from
# the matching SaliSAM run's args.json so both defenses share one operating point.
VICTIM_KEYS = ('dataset', 'victim', 'input_size', 'yolo11_weights', 'yolo11_conf',
               'yolo11_adv_conf', 'yolo11_iou')


def salisam_path(path):
    """A SaliSAM-relative path made absolute (absolute paths pass through)."""
    return path if os.path.isabs(path) else os.path.join(SALISAM_ROOT, path)


def list_images(directory):
    return sorted(f for f in os.listdir(directory) if f.lower().endswith(IMAGE_EXTS))


def use_salisam():
    """Put SaliSAM's checkout and yolo_tools on sys.path (for its evaluators)."""
    if not os.path.isdir(YOLO_TOOLS):
        sys.exit(f"ERROR: no SaliSAM checkout at {SALISAM_ROOT} (looked for {YOLO_TOOLS}); "
                 f"set SALISAM_ROOT")
    for p in (YOLO_TOOLS, SALISAM_ROOT):
        if p not in sys.path:
            sys.path.insert(0, p)
