"""
One table from runs scored by evaluate_antistyler.py (or Saliuitl's evaluate_saliuitl.py).

Reads, per run, the files those scripts write (SaliSAM's formats):
<out_root>/<run>/coco_eval_{clean,atk,rec}/coco_metrics.json,
<out_root>/<run>_clean/coco_eval_rec/coco_metrics.json and
<out_root>/metrics/<run>/{saliuitl/saliuitl_metrics.json, per_target/attack_success.json},
where <out_root> is the run's parent directory, and prints / saves one row per
run. Missing pieces are left blank. The columns are the Saliuitl comparison's,
so its rows can be put in the same table.

Usage (from the repo root):
    python salisam_comparisons/compare.py \\
        AntiStyler=salisam_comparisons/outputs/AntiStyler_UCYHPC_inria_yolo11n_patch_vanish_all_instance_yolo11 \\
        SaliSAM-FastSAM=salisam_comparisons/outputs/salisam_rescored/SaliSAM_FastSAM_UCYHPC_inria_yolo11n_patch_vanish_all_instance_yolo11 \\
        --csv salisam_comparisons/outputs/comparison.csv
"""

import argparse
import csv
import json
import os

COLUMNS = [
    ('AP clean', 'AP of the undefended clean images'),
    ('AP atk', 'AP of the undefended attacked images'),
    ('AP D(atk)', 'AP of the defense output on attacked images'),
    ('AP D(clean)', 'AP of the defense output on clean images'),
    ('AP50 D(atk)', ''),
    ('AP50 D(clean)', ''),
    ('img ASR', "Saliuitl image-level attack success"),
    ('img RR', ''), ('img LPR', ''), ('img IAR', ''),
    ('tgt ASR', 'per-target attack success, undefended'),
    ('tgt ASR D', 'per-target attack success after the defense'),
    ('tgt RR', ''), ('tgt new fail', ''), ('tgt LPR', ''),
]


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def row_for(run):
    run = os.path.normpath(run)
    out_root, name = os.path.dirname(run), os.path.basename(run)
    r = {}

    def coco(d, key, col, stat='AP'):
        m = load(os.path.join(d, key, 'coco_metrics.json'))
        if m:
            r[col] = m['overall'][stat]

    coco(run, 'coco_eval_clean', 'AP clean')
    coco(run, 'coco_eval_atk', 'AP atk')
    coco(run, 'coco_eval_rec', 'AP D(atk)')
    coco(run, 'coco_eval_rec', 'AP50 D(atk)', 'AP50')
    coco(run + '_clean', 'coco_eval_rec', 'AP D(clean)')
    coco(run + '_clean', 'coco_eval_rec', 'AP50 D(clean)', 'AP50')

    s = load(os.path.join(out_root, 'metrics', name, 'saliuitl', 'saliuitl_metrics.json'))
    if s:
        m = s['metrics']
        r.update({'img ASR': m.get('asr'), 'img RR': m.get('rr'), 'img LPR': m.get('lpr'),
                  'img IAR': m.get('iar')})
    a = load(os.path.join(out_root, 'metrics', name, 'per_target', 'attack_success.json'))
    if a:
        summ = {x['label']: x for x in a['summaries']}
        variant = a['config']['variant']
        r['tgt ASR'] = summ.get(variant, {}).get('asr')
        rec = a['recovery'].get('recovered') or next(iter(a['recovery'].values()), {})
        r['tgt ASR D'] = summ.get('recovered', {}).get('asr')
        r.update({'tgt RR': rec.get('rr'), 'tgt new fail': rec.get('new_failure_rate'),
                  'tgt LPR': rec.get('lpr')})
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('runs', nargs='+', help='[label=]run dir under its out_root')
    p.add_argument('--csv', default=None)
    args = p.parse_args()

    rows = []
    for spec in args.runs:
        label, _, path = spec.rpartition('=')
        rows.append({'method': label or os.path.basename(os.path.normpath(path)),
                     **row_for(os.path.expanduser(path))})

    cols = [c for c, _ in COLUMNS]
    width = max(len(r['method']) for r in rows) + 2
    print(f"{'method':<{width}}" + ''.join(f"{c:>14}" for c in cols))
    for r in rows:
        print(f"{r['method']:<{width}}" + ''.join(
            f"{r[c]:>14.3f}" if isinstance(r.get(c), (int, float)) else f"{'':>14}" for c in cols))
    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['method'] + cols)
            w.writeheader()
            w.writerows(rows)
        print(f"Saved: {args.csv}")


if __name__ == '__main__':
    main()
