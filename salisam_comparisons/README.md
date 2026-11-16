# AntiStyler vs SaliSAM on YOLO11n

This runs AntiStyler on SaliSAM's `UCYHPC_inria_yolo11n_patch_vanish` set: 288 native-resolution INRIA
test frames, with every person patched. It uses the same YOLO11n COCO victim as `salisam_fastsam.py`
(`SaliSAM/core/nets/yolo11/weights/COCO/yolo11n.pt`, identical to `models/yolo11/yolo11n.pt`) and scores
the result with SaliSAM's own evaluators, writing SaliSAM's output files. It mirrors the Saliuitl
comparison in `../Saliuitl/salisam_comparisons`, and its table includes that run when it has been scored.

```sh
bash salisam_comparisons/run_antistyler_inria_yolo11.sh                  # everything
LIM=5 OUT=/tmp/smoke bash salisam_comparisons/run_antistyler_inria_yolo11.sh   # smoke test
```

The stages are: defend attacked → defend clean (both for each output mode) → evaluate (these runs, plus
a rescore of the matching SaliSAM run) → comparison table. The defense runs with this repo's `.venv`. The
evaluators run with SaliSAM's `.venv` (`EVAL_PY`), so they see the package versions SaliSAM pins
(ultralytics 8.4.9, opencv 4.11; this repo has newer ones). The script needs the SaliSAM checkout at
`../SaliSAM`; set `SALISAM_ROOT` if it lives elsewhere.

## Files

| file | what |
|---|---|
| `antistyler_defend.py` | AntiStyler on one image folder, written as a SaliSAM run directory |
| `antistyler_inria.json` | its runtime and defense settings: AntiStyler's released defaults, seed 42 |
| `evaluate_antistyler.py` | SaliSAM's `coco_eval` / `saliuitl_metrics` / `attack_success` on a run's `recovered/` |
| `compare.py` | one row per run, from the files above (same columns as the Saliuitl comparison) |
| `common.py` | SaliSAM checkout paths, standard library only (it is imported from both venvs) |

## What was adapted, and why

- **Output.** AntiStyler is image-in, image-out, so unlike Saliuitl its run directory has exactly the
  shape of a `salisam_fastsam.py` run: `args.json` in SaliSAM's schema plus `recovered/<stem>.png` at
  each frame's own H×W. The PNGs are rounded the same way as SaliSAM's `save_png`. SaliSAM's evaluators
  score it unmodified, and so does `SaliSAM/scripts/evaluate_attack_defense.sh`. AntiStyler also writes
  `masks/` (the final mask at its working resolution) and `per_image.json` (mask fired, mask share,
  threshold, seconds).
- **Two output modes.** The released notebook, which this repo reproduces, does not apply the mask to the
  input. When the mask fires, its output is the AntiStyled 580×580 working interior with the mask zeroed,
  resized back to H×W. The whole frame is therefore resampled through a square and slightly restyled. The
  paper and README describe masking the input image instead. Both are run:
  - `AntiStyler` (`defense.output: antistyled`) is the notebook's output.
  - `AntiStyler-masked` (`masked_input`) zero-fills the mask, upsampled with nearest neighbour, on the
    native frame, which is also how SaliSAM fills its own mask (`--inpaint_value 0 0 0`). Every pixel
    outside the mask is left unchanged.
- **Victim.** AntiStyler is detector-agnostic and never runs the victim. The victim enters only when a run
  is scored. Its settings are copied from the SaliSAM run's `args.json`: imgsz 416, conf 0.4 and NMS 0.4
  (Saliuitl's operating point), and `yolo11_adv_conf` 0.01.
- **Settings.** These are AntiStyler's released defaults, untuned: tau 0.99, a 600 px working square with a
  10 px random border, one AntiStyle iteration, and kernels 11/11/51/11. They come from the COCO /
  Faster R-CNN notebook. SaliSAM's INRIA configuration, by contrast, was grid-searched on a dev half of
  these 288 frames.
- **Seeding.** Before every frame the RNGs are reset to seed 42. A clean frame and its attacked twin
  therefore get the same random border and style, regardless of `LIM` or order.

## Evaluator settings

Every evaluator runs with `--batch 1`, as in the Saliuitl comparison. At SaliSAM's default `--batch 8`,
Ultralytics letterboxes a batch onto a rectangle when all its images share one size and onto a square
otherwise. A frame's canvas, and so its detections, then depend on its neighbours: 168 of the 288 frames
get the rectangle. The SaliSAM run is rescored the same way into `outputs/salisam_rescored/`, which must
reproduce the Saliuitl comparison's rescore. COCO AP uses imgsz 416, conf 0.001 and NMS 0.7, the settings
of `evaluate_coco.py`. The image-level and per-target metrics use the 0.4 / 0.4 operating point.

## Things to keep in mind when reading the numbers

- The UCYHPC set was made on the cluster with `configs/attack_inria_yolo11n.json` but
  `train.epochs: 100` (10 here). Its patch therefore differs from `outputs/attack_inria_yolo11n/patch.pt`.
  Its `clean/` and `labels/` are byte-identical to `data/inria_yolo11n_patch_vanish`.
- The patch was optimised at imgsz 640 and is scored here at 416, SaliSAM's victim size. That is the same
  for every method.

## Cost

The defense takes about 7 s per frame on CPU (6 threads), whether or not the mask fires. Most of that
is VGG19 forward and backward passes on the 600×600 working image. Four passes over 288 frames
(attacked and clean, for each output mode) take about 2.3 h. Scoring with the evaluators at batch 1
takes a few minutes per run. VGG19 (`IMAGENET1K_V1`, 548 MB) is downloaded once into `.cache/torch/`.

## Outputs (`salisam_comparisons/outputs/`, gitignored)

```
AntiStyler_UCYHPC_inria_yolo11n_patch_vanish_all_instance_yolo11/
    args.json  recovered/  masks/  per_image.json
    visualizations/{attacked,cleaned}/<stem>_{attacked,cleaned}_boxes.png
    coco_eval_{clean,atk,rec}/     coco_metrics.json  per_class.csv  coco_summary.txt
AntiStyler_..._yolo11_clean/       the same for the --clean run; coco_eval_rec = AP of D(clean)
AntiStyler-masked_..._yolo11{,_clean}/
metrics/<run>/
    saliuitl/                      saliuitl_metrics.json  .txt  saliuitl_per_image.csv
    per_target/                    attack_success.json  .txt  patch_effectiveness.csv ...
salisam_rescored/                  the same tree for the SaliSAM run
logs/
comparison.csv
```
