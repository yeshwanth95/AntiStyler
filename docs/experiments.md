# Standalone experiments

## What was extracted

The notebook contains one targeted patch optimizer, Faster R-CNN inference,
VGG19-based AntiStyle optimization, and the AntiStyler mask pipeline. It does
not contain implementations of all the attack families listed in the paper's
README, dataset evaluation, mAP computation, or training a feed-forward style
removal network. These scripts reproduce the demo workflows, not the paper's
complete benchmark suite.

| File | Responsibility |
| --- | --- |
| `run_attack.py` | Clean detection, patch optimization, attacked detection |
| `run_defense.py` | Detector-independent defense and intermediate masks |
| `run_detection.py` | Detection JSON and an annotated image |
| `run_experiment.py` | Clean/attacked detection, defense of both images, defended detection |
| `antistyler/attack.py` | Patch placement, target construction, detector-loss optimization |
| `antistyler/defense.py` | Gram losses, two-step AntiStyle optimization, mask enhancement |
| `antistyler/models.py` | Explicit pretrained weights or local checkpoints |
| `antistyler/images.py` | Image/tensor loading and headless output |
| `antistyler/config.py` | Defaults, strict config loading and validation |
| `antistyler/runtime.py` | RNG setup and run provenance |
| `antistyler/cli.py` | Workflow orchestration and common CLI |

The notebook remains unchanged. There are no imports from IPython, display
calls, global class-name dependencies, or package-install commands in the new
runtime code. Visualization uses Pillow and works without a display server.

## Configuration

Each entry point accepts `--config`, `--seed`, `--output-dir`, and
`--validate-config`. Its default file is the corresponding JSON under `configs/`.
These complete examples include all parameters; partial JSON configs may omit
fields to use the defaults in `antistyler/config.py`. Unknown keys are errors.
`--validate-config` checks configuration structure and numeric ranges; actual
image bounds, weight enum names, and backbone layer existence are checked when
the corresponding stage is loaded.

All paths in JSON are relative to the JSON file's directory. CLI
`--output-dir` is relative to the current working directory. Resolved configs
contain absolute paths and can be supplied directly to another run with a new
`--output-dir`. An existing nonempty output directory is rejected.

| Section/key | Meaning and default |
| --- | --- |
| `input_image` | RGB/grayscale image, `.npy`, or exact `.pt` tensor; examples use the supplied `image.jpg` |
| `output_dir` | Run artifact directory |
| `runtime.seed` | `42`; `null` requests a fresh seed that is then recorded |
| `runtime.deterministic` | `true`; requires deterministic PyTorch algorithms and configures cuDNN/cuBLAS |
| `runtime.device` | `auto`, `cpu`, `cuda`, or `cuda:N` |
| `runtime.num_threads` | CPU PyTorch thread count, `1` |
| `runtime.cache_dir` | Torch model cache; examples use `../.cache/torch`; `null` uses Torch's environment/default |
| `detector.weights` | Faster R-CNN ResNet50 FPN `COCO_V1`; explicit version, no `DEFAULT` alias |
| `detector.checkpoint` | Optional full detector state dictionary; set `weights` to `null` |
| `detector.min_size`, `max_size` | Internal detector resize, `800` / `1333` |
| `detector.score_threshold` | Drawing threshold, `0.7`; detection JSON keeps all returned detections |
| `attack.patch_size` | Square patch side in original-image pixels, `100` |
| `attack.position` | Top-left `[x, y]`; `null` uses the notebook's `floor((dimension + patch_size) / 2)` formula for each axis |
| `attack.steps`, `learning_rate` | Adam updates, `200` / `0.1` |
| `attack.target_label` | Case-insensitive COCO name or foreground class ID, `person` |
| `attack.target_boxes` | Optional nonempty list of `[x1, y1, x2, y2]` boxes in original-image coordinates |
| `attack.target_score_threshold` | Minimum clean detection confidence for automatic target boxes, `0.7` |
| `attack.target_selection` | `all` qualifying boxes or `highest_score`; default `all` |
| `attack.loss_weights` | Weights for classifier, box, objectness and RPN box losses; all `1.0` |
| `defense.weights` | VGG19 `IMAGENET1K_V1` |
| `defense.checkpoint` | Optional full torchvision VGG19 state dictionary; set `weights` to `null` |
| `defense.content_layers` | VGG layer names, `["conv_4"]` |
| `defense.style_layers` | `["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]` |
| `defense.content_weight`, `style_weight` | `1.0` / `1000.0` |
| `defense.steps`, `learning_rate` | Two Adam updates per iteration, `1` / `0.001` |
| `defense.image_size`, `padding` | Square working image and random border widths, `600` / `10` |
| `defense.tau` | Difference order statistic fraction, `0.99`; must be in `(0,1]` |
| `defense.mask_threshold` | Smoothed-mask cutoff, `0.5` |
| `defense.dilation_kernel`, `erosion_kernel` | Initial mask closing kernels, `11` / `11` |
| `defense.smoothing_kernel` | Mean filter width, `51` |
| `defense.final_dilation_kernel` | Final mask expansion, `11` |

Morphology kernels must be positive odd integers. Padding may be zero, but must
leave a positive interior. The working size must also be large enough for the
pooling layers preceding your selected VGG loss layers.

Image files are converted to RGB. NumPy inputs support H×W, H×W×1, or H×W×3;
`uint8` arrays are divided by 255, floating arrays must already be in `[0,1]`.
Tensor inputs must contain an RGB C×H×W or 1×C×H×W tensor in `[0,1]`.
Checkpoints are plain state dictionaries and are loaded strictly with
`weights_only=True`. Setting both weights and checkpoint to `null` creates a
random model for plumbing tests, not meaningful attack/defense evaluation.

### Attack target semantics

As in the demo, the attack **minimizes** the detector's supervised training
losses toward the configured target boxes and class. It is not an untargeted
loss-maximizing attack. The model's weights are frozen; gradients still flow
through it into the learnable patch. One target label is created per box.

When `target_boxes` is null, clean detections supply the boxes, regardless of
their original classes. The selected boxes are all relabeled to `target_label`.
The new default threshold `0.7` omits low-confidence boxes; set
`target_score_threshold` to `0.0` with `target_selection: "all"` to use every
returned box as the notebook attempted. No qualifying boxes is an actionable
error: lower the threshold or provide explicit boxes. Explicit boxes bypass
automatic score filtering and selection. Actual targets are saved in
`attack.json`.

The notebook's placement formula is preserved by default but is not centered
placement. If it extends beyond the image, the run fails before model loading;
choose a valid explicit position or a smaller patch.

### Defense semantics

The implementation preserves these notebook details:

1. Resize into a square interior and draw four uniform random borders, with
   the vertical borders overwriting the corners. Draw the style image from a
   normal distribution.
2. Use VGG features on `[0,1]` image values without adding ImageNet normalization.
3. For each iteration, make an unweighted negative-style-loss Adam update,
   then a content-weighted/negative-style-weighted Adam update using the same
   optimizer. Clamp to `[0,1]` after both updates.
4. Threshold the channel-mean absolute difference using the 1-based
   `floor(tau * pixel_count)` order statistic, including the padded border.
   The rank is clamped to at least one for tiny inputs.
5. Crop the border, dilate, erode, apply a zero-padded mean filter, threshold,
   and dilate again.
6. If the mask is nonempty, mask the **AntiStyled** interior and resize it to
   the input resolution. If the mask is empty, return the original image exactly.

The last behavior follows the notebook even though the README overview can
be read as masking the original input. Saved masks use the defense's working
interior resolution, not the input resolution. PNG previews are rounded;
`.pt` tensors preserve the numerical outputs.

## Reproducibility

Python `random`, NumPy's global RNG, PyTorch CPU, and all CUDA RNGs are seeded.
The run resets these RNGs before each model construction and workflow stage.
The two defense inputs use the same seed, allowing their masks to be compared
under the same random padding and style. A standalone defense run on the
attack's `.pt` output therefore agrees with the full pipeline when configs,
models, hardware and software match.

With `deterministic: true`, deterministic algorithms are required, cuDNN
benchmarking is disabled, TF32 is disabled, and cuBLAS is configured before
CUDA initialization. Unsupported deterministic GPU operations raise an error
instead of silently changing behavior. Bitwise replay is only expected within
the same software/hardware environment; PyTorch does not guarantee equality
across platforms, releases, or CPU/GPU devices.

Change `runtime.seed` or pass `--seed 123` to explore another repeatable run.
Use `runtime.seed: null` or `--seed random` for a newly sampled seed on each
invocation; the sampled integer replaces null in `resolved_config.json`.
To permit nondeterministic backend kernels as well, set
`runtime.deterministic` to `false`. That setting alone does not guarantee
different outputs, and an explicit seed alone cannot make those kernels
replay exactly.

```bash
# Several independent, reproducible runs.
for seed in 42 43 44; do
    python run_experiment.py --seed "$seed" --output-dir "outputs/seed-$seed"
done

# Replay a completed run with its exact resolved settings.
python run_experiment.py --config outputs/seed-42/resolved_config.json \
    --output-dir outputs/seed-42-replay
```

The scripts process one image per run, so there are no DataLoader workers or
worker RNGs. No hash-based ordering is used in numerical computations. Model
downloads are managed by torchvision. Versioned pretrained weight enums are
stored in the config; local checkpoint files are also SHA-256 fingerprinted.

## Outputs

Every run saves `resolved_config.json`, `metadata.json`, `environment.txt`,
`categories.json`, `status.json`, and the input as `.pt` / `.png`.
Metadata includes the actual seed/device, Python and package versions,
CUDA/cuDNN information, Git revision/dirty state, hashes of the implementation,
input hash, and any local checkpoint hashes. `status.json` records completion
or a stage failure; partial results are retained after failure.

Attack runs add `patch`, `attacked`, targets, per-step loss components, and clean
and attacked detections. Defense runs add `defended` and a `defended_stages/`
directory containing padded/styled images, differences, raw/final masks, and
optimization losses. Full runs use `defended_benign` and `defended_attacked`
names and evaluate both results. Detection outputs include the full boxes,
scores and labels JSON plus a thresholded annotated PNG. Detection coordinates
refer to the image supplied to that stage.

For exact chaining, use `attacked.pt` as defense input, and `defended.pt` as
detection input. PNG is convenient for viewing but quantizes pixel values.

## Verification and fixes

Run `python -m unittest discover -s tests -v`. Tests need the project
dependencies but require no network or pretrained weights. They cover RNG
replay and changed seeds, patch-only edits and detector state restoration,
box/label shape consistency, invalid targets/placement, image input handling,
empty masks, zero padding, configuration validation, all four workflows, and
equivalence between standalone stages and the full pipeline. A regression
test executes the original notebook's defense definitions with a tiny CPU
backbone and checks exact equality of outputs and intermediate masks.

The conversion also fixes CPU-created patches being used with CUDA images,
out-of-bounds writes, uninitialized loss targets, and zero-padding slicing.
It removes unnecessary retained graphs, detector parameter gradients, and
per-iteration CUDA cache clearing. BatchNorm statistics are held fixed during
patch optimization, and detector training/gradient flags are restored even
if optimization fails. Pretrained Faster R-CNN already uses frozen BatchNorm.

The default 200-step pretrained experiment is compute-heavy, especially on
CPU. Smoke tests with random weights validate execution, not the paper's
accuracy, performance, or adversarial effectiveness.

## Dataset patch attack

`run_patch_attack.py` (default config `configs/attack_visdrone.json`; the
vanishing variant is `configs/attack_visdrone_vanish.json`) is not
part of the notebook. It generates an adversarially patched copy of a
YOLO-format dataset against an Ultralytics YOLO detector, for evaluating
defenses such as AntiStyler at dataset scale.

| File | Responsibility |
| --- | --- |
| `antistyler/patch_config.py` | Strict config, validation, path resolution |
| `antistyler/yolo_data.py` | YOLO split loading, cv2 image I/O identical to Ultralytics |
| `antistyler/patching.py` | Per-object placement, differentiable pasting, augmentation, TV |
| `antistyler/victim.py` | YOLO loading, differentiable letterbox and class scores, official prediction |
| `antistyler/evaluation.py` | COCO AP/AR via pycocotools, object-level attack success rate |
| `antistyler/patch_attack.py` | Training, attacked split writing, evaluation |

### Threat model

A single universal square patch is pasted on the center of every ground-truth
object of the target classes, with side `patch.scale * sqrt(box_w * box_h)`
(capped at `patch.max_box_fraction` of the shorter box side; objects needing a
side below `patch.min_side` pixels are left unpatched). This is the per-object
universal setting of TSEA and naturalistic patch attacks: physically, one
printed sticker on every vehicle roof. Ground-truth labels determine where the
patch goes; the attacked dataset keeps the original labels.

The patch is optimized on the **training** split and applied to the
**evaluation** split, so evaluation images are never seen in training. Each
step loads original-resolution images, pastes the patch with position, scale
and photometric jitter, applies a differentiable letterbox that matches
Ultralytics' cv2 `INTER_LINEAR` resize and gray padding, and minimizes

    sum(target-class scores of anchors above train.score_threshold) / patched objects
    + train.tv_weight * total_variation(patch)

over all anchors, where an anchor's score is its maximum sigmoid score over the
target classes. `train.loss: "max"` minimizes only the per-image maximum instead.

With `train.suppress: "target_classes"` (the default) the loss is indifferent
to what else the detector sees, so hidden vehicles may be relabelled as other
classes. `train.suppress: "all_classes_on_objects"` makes them vanish instead:
an anchor that covers a patched object (its anchor point lies inside the
object's letterboxed box, or its predicted box has IoU at least
`train.object_iou` with it) is scored by its maximum over **all** classes;
every other anchor keeps the target-class score, so vehicles cannot reappear
elsewhere and unrelated objects are not suppressed.
The attacked split is written without augmentation as lossless PNG by default.

### Metrics

All conditions use Ultralytics' own inference pipeline (`YOLO.predict`, one
image at a time) on the saved files, so evaluation includes the real
preprocessing and NMS. Three conditions are compared:

- `clean`: original images.
- `control`: a uniform-noise patch with identical placements (the random
  initialization of the patch when `patch.init` is `random`). It measures how
  much the occlusion alone hurts detection.
- `attacked`: the saved attacked dataset.

COCO AP/AR come from pycocotools at `evaluation.conf` (0.001) for all classes
and for the target classes alone, with per-class AP. A target object counts as
detected if a prediction of any target class with confidence at least
`evaluation.asr_conf` overlaps it with IoU at least `evaluation.asr_iou`.
The attack success rate is

    ASR = |detected on clean and not detected on attacked| / |detected on clean|

reported over all target objects and over patched objects only. Target recall
and the number of confident target predictions are also reported. ASR is null
if nothing is detected on clean images.

ASR alone does not distinguish vanishing from misclassification, so two
vanish rates are reported for patched objects detected on clean images:
`vanish_patched_objects` (`vanish(p)` in `summary.txt`) counts objects with
no confident prediction of **any** class at IoU at least `asr_iou`, and
`vanish_strict_patched_objects` (`strict(p)`) counts objects that do not even
contain the center of a confident prediction of any class. The strict rate also
catches small spurious boxes, such as a "pedestrian" on the patch, whose IoU with
the vehicle box is below 0.5.

### Outputs

The run directory contains the resolved config, provenance metadata (including
the victim checkpoint hash), `train_images.json`, `train_log.jsonl`, the
`patch` and `control_patch` (`.pt` and `.png`), the patch after each epoch,
`metrics.json`, `summary.txt`, gzip COCO-format predictions for each condition
and annotated `previews/`. The attacked dataset directory contains the patch,
`attack_config.json` and, with `output_dataset.layout: "yolo"`,
`<eval_split>/images`, `<eval_split>/labels` and a `data.yaml` usable by
Ultralytics. With `layout: "variants"` (the SaliSAM layout used by both VisDrone
configs) it contains `<eval_split>/all_instance/` (attacked images),
`<eval_split>/clean/` (original pixels in the same format and with the same
names) and one shared `<eval_split>/labels/`; Ultralytics cannot pair these
folders with labels, so no `data.yaml` is written.

Reproducibility follows the rest of the project: one `torch.Generator` seeded
with `runtime.seed` drives training-image sampling, shuffling, patch
initialization and augmentation; `--seed` and `--seed random` work as for the
other entry points. `--dataset-dir` overrides the attacked dataset directory.
Set `patch.init` to a saved `patch.pt` and `train.epochs` to `0` to apply an
existing patch (for example to another victim) without retraining. CPU results
also depend on `runtime.num_threads`, which is recorded in the config.

### Results: VisDrone val, YOLO11m

`configs/attack_visdrone.json` (seed 42, CPU, 6 threads; 1024 training images
x 4 epochs at batch size 4, patch scale 0.5). Targets: car, van, truck, bus;
17,040 target objects in 548 images, 17,034 patched. Attacked dataset:
`data/visdrone_yolo_patch_attacked` (PNG, original labels; written as
`data/visdrone_yolo_patch` in the YOLO layout and later rearranged into the
variants layout with a `clean` folder). AP is COCO AP@[.5:.95].

| Condition | AP (all) | AP50 (all) | AR@100 (all) | AP (targets) | AP50 (targets) | AR@100 (targets) | Target recall @0.25 | ASR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.257 | 0.417 | 0.342 | 0.400 | 0.549 | 0.464 | 82.3% | - |
| control (noise patch) | 0.184 | 0.322 | 0.294 | 0.244 | 0.355 | 0.342 | 63.9% | 23.6% |
| attacked | 0.075 | 0.146 | 0.168 | 0.030 | 0.056 | 0.061 | 15.7% | 81.3% |

Per-class AP, clean -> attacked: car 0.575 -> 0.090, van 0.293 -> 0.002,
truck 0.277 -> 0.011, bus 0.453 -> 0.019. Confident (>= 0.25) target-class
predictions fall from 16,690 to 4,200. Of the clean-detected patched vehicles,
61.7% vanish at IoU 0.5 but only 22.6% vanish strictly (control: 18.6% and 14.2%).

The loss only suppresses target-class scores, and the optimizer partly does so
by making the patch resemble other classes. Of the 11,406 hidden vehicles,
28.5% have no confident detection inside their box; the rest are covered by a
confident non-target detection, mostly pedestrian (3,038), people (2,061),
tricycle (1,973) and motor (943). Those spurious boxes also lower non-target AP
(people 0.138 -> 0.060, pedestrian 0.237 -> 0.181, motor 0.241 -> 0.175). The
attack therefore removes vehicle predictions but is largely a misclassification
rather than a pure vanishing attack.

The scale and training settings were chosen on 150 test-dev images, never on
val. Random-patch occlusion alone at scale 0.2/0.3/0.4 hid 4%/8%/14% of
detected vehicles there.

### Results: vanishing variant, VisDrone val, YOLO11m

`configs/attack_visdrone_vanish.json` is identical to `attack_visdrone.json`
(same seed, patch size and scale, training images and budget, placements)
except for `train.suppress: "all_classes_on_objects"` with `train.object_iou: 0.3`.
Attacked dataset: `data/visdrone_yolo_patch_vanish` (variants layout; the
`clean` images and `labels` are byte-identical to those of
`data/visdrone_yolo_patch_attacked`). Clean and control rows are unchanged from
the table above.

| Condition | AP (all) | AP50 (all) | AR@100 (all) | AP (targets) | AP50 (targets) | AR@100 (targets) | Target recall @0.25 | ASR | Vanish @IoU 0.5 | Vanish (strict) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.257 | 0.417 | 0.342 | 0.400 | 0.549 | 0.464 | 82.3% | - | - | - |
| control (noise patch) | 0.184 | 0.322 | 0.294 | 0.244 | 0.355 | 0.342 | 63.9% | 23.6% | 18.6% | 14.2% |
| misclassification patch | 0.075 | 0.146 | 0.168 | 0.030 | 0.056 | 0.061 | 15.7% | 81.3% | 61.7% | 22.6% |
| vanishing patch | 0.113 | 0.220 | 0.209 | 0.077 | 0.126 | 0.137 | 29.2% | 65.2% | 55.3% | 50.1% |

Vanish rates are over the 14,030 patched vehicles detected on clean images.
Of the 9,149 vehicles the vanishing patch hides, 7,027 (77%) leave no
confident box of any class centered on them, against 28% for the
misclassification patch. Non-target classes are almost unaffected: AP clean ->
vanishing (misclassification) for pedestrian 0.237 -> 0.224 (0.181), people
0.138 -> 0.127 (0.060), motor 0.241 -> 0.228 (0.175). Per-class target AP, clean ->
vanishing: car 0.575 -> 0.217, van 0.293 -> 0.004, truck 0.277 -> 0.029, bus
0.453 -> 0.059; confident target predictions fall from 16,690 to 6,427.

Vanishing is the harder objective, so at the same budget it hides fewer
vehicles than the misclassification patch: ASR 65% against 81%, and target AP
0.077 against 0.030. Its all-class AP drop is smaller because it no longer adds
false pedestrian, people, tricycle and motor boxes. The training loss was still
decreasing after the fourth epoch (1.36, then 1.29). On the 150 tuning
images from test-dev, the same objective at a smaller budget reached a strict
vanish rate of 47% against 39% for the target-class loss.

## Person vanishing patches: INRIAPerson and PASCAL VOC2007

Universal vanishing patches for the `person` class against the default
COCO-pretrained Ultralytics `yolo11n.pt` and `yolo11m.pt` (official v8.3.0
release assets in `models/yolo11/`; SHA-256 recorded in each run's metadata).

### Dataset conversion

`prepare_dataset.py` (`antistyler/datasets.py`) converts the original
datasets into the YOLO layout read by `run_patch_attack.py`. Labels use the
victim's COCO class indices, taken from Ultralytics' bundled `coco.yaml`. Images
are copied unchanged. The output has a `data.yaml`, which `run_patch_attack.py`
checks against the victim's class names before training, and a
`conversion.json` with per-split counts.

```bash
python prepare_dataset.py --dataset inria --source data/INRIAPerson --output <dir>     # -> data/inria_yolo
python prepare_dataset.py --dataset voc2007 --source data/pascalVOC2007 --output <dir> # -> data/voc2007_yolo
```

| Dataset | Split | Source | Images | Objects | Persons |
| --- | --- | --- | --- | --- | --- |
| INRIA | train | `Train/pos` + `Train/annotations` | 614 | 1,237 | 1,237 |
| INRIA | test | `Test/pos` + `Test/annotations` | 288 | 589 | 589 |
| VOC2007 | train | `ImageSets/Main/train.txt` | 2,501 | 6,301 | 2,358 |
| VOC2007 | val | `ImageSets/Main/val.txt` (smoke checks only) | 2,510 | 6,307 | 2,332 |
| VOC2007 | test_500 | `Detection_splits/test_500` | 500 | 1,189 | 434 |

INRIA annotations are 0-based inclusive pixel boxes and VOC annotations are
1-based inclusive; both are converted to continuous coordinates. All 20 VOC
classes map onto COCO names (aeroplane -> airplane, diningtable -> dining
table, motorbike -> motorcycle, pottedplant -> potted plant, sofa -> couch,
tvmonitor -> tv). Objects marked `difficult` are dropped, as in Ultralytics' VOC
conversion. The non-person VOC classes stay in the labels, so collateral AP
can be measured. INRIA labels only some of the people in each image, so
unlabelled people are counted as false positives in AP for every condition.

### Configurations

`configs/attack_{inria,voc2007}_{yolo11n,yolo11m}.json` differ only in victim,
dataset and epochs:

- **Patch:** a 64 x 64 patch with side `0.2 * sqrt(w^2 + h^2)` of each person box
  (`patch.reference: "diagonal"`, the AdvPatch INRIA convention), centered on the box.
- **Training:** the vanishing objective (`train.suppress:
  "all_classes_on_objects"`) with the per-image maximum score loss
  (`train.loss: "max"`), lr 0.03 cosine, TV 0.05, batch size 4, all training
  images containing persons and the VisDrone augmentations. Training runs 10
  epochs on INRIA (614 images) and 6 on VOC (1,025 images), about 6,000 image
  passes in each case, with the same budget for both victims.
- **Evaluation:** INRIA `test` (Test/pos) and VOC `test_500`, written in the
  variants layout (`<split>/all_instance`, `<split>/clean`, `<split>/labels`) to
  `data/{inria,voc2007}_{yolo11n,yolo11m}_patch_vanish`.

The VisDrone optimizer settings (sum-above-threshold loss, lr 0.1, TV 0.5)
were far too weak here. With yolo11n on INRIA test, person AP50 only fell from
0.959 to 0.742, with ASR 13.8% and strict vanish 5.0%; that run is kept in
`outputs/superseded/`. The loss stayed nearly flat. Summing over the roughly 20
confident anchors per large person spreads the gradient over anchors the small
patch barely influences. The settings were therefore re-chosen with short
yolo11n runs (4 epochs) that train on 512 INRIA train images and evaluate on
the other 102 (every sixth image).

| Loss | lr | TV | Person AP50 (clean 0.734) | ASR | Vanish @IoU 0.5 | Strict vanish |
| --- | --- | --- | --- | --- | --- | --- |
| sum | 0.1 | 0.5 | 0.571 | 4.0% | 4.0% | 1.0% |
| sum | 0.03 | 0.5 | 0.548 | 5.6% | 5.6% | 1.5% |
| sum | 0.01 | 0.5 | 0.640 | 4.0% | 4.0% | 1.5% |
| max | 0.01 | 0.5 | 0.404 | 9.1% | 7.6% | 1.5% |
| max | 0.01 | 0.05 | 0.567 | 2.5% | 2.5% | 1.0% |
| max | 0.1 | 0.05 | 0.344 | 12.1% | 12.1% | 1.5% |
| **max** | **0.03** | **0.05** | **0.256** | **22.2%** | **22.2%** | **2.5%** |

The same settings are used for all four experiments. No hyperparameter was
tuned on the test splits. A clean-only smoke run on 50
test images (no training) gave a yolo11n person AP50 of 0.955 on INRIA and 0.81
on VOC, which confirms that the labels and class indices are correct. A
random-noise patch of this size hid only about 1% of detected persons.

### Results

Person metrics on INRIA `test` (288 images, 589 persons) and VOC `test_500`
(500 images, 434 persons, all patched). ASR is the share of clean-detected
persons that are no longer detected as person at confidence 0.25 / IoU 0.5.
Vanish requires that no class is detected there; strict vanish requires that no
confident prediction center falls inside the box. The control is a uniform
noise patch of the same size and placement.

| Dataset | Victim | Condition | Person AP | AP50 | AR@100 | Recall | ASR | Vanish | Strict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| INRIA | yolo11n | clean | 0.657 | 0.959 | 0.747 | 98.5% | – | – | – |
| | | control | 0.651 | 0.951 | 0.750 | 97.1% | 1.7% | 1.7% | 1.0% |
| | | **attacked** | **0.147** | **0.363** | 0.637 | 70.5% | **28.6%** | 28.6% | 4.3% |
| INRIA | yolo11m | clean | 0.654 | 0.948 | 0.751 | 98.3% | – | – | – |
| | | control | 0.636 | 0.929 | 0.750 | 97.6% | 0.9% | 0.9% | 0.2% |
| | | **attacked** | **0.296** | **0.491** | 0.723 | 83.4% | **15.5%** | 15.5% | 1.4% |
| VOC2007 | yolo11n | clean | 0.617 | 0.883 | 0.755 | 92.6% | – | – | – |
| | | control | 0.590 | 0.848 | 0.746 | 89.4% | 6.5% | 6.2% | 3.2% |
| | | **attacked** | **0.375** | **0.621** | 0.682 | 77.7% | **18.7%** | 17.7% | 4.5% |
| VOC2007 | yolo11m | clean | 0.687 | 0.905 | 0.797 | 97.0% | – | – | – |
| | | control | 0.625 | 0.834 | 0.785 | 91.9% | 5.9% | 5.9% | 1.7% |
| | | **attacked** | **0.446** | **0.638** | 0.741 | 77.7% | **20.4%** | 19.7% | 3.8% |

All-class VOC AP (20 classes) barely moves: yolo11n 0.629 -> 0.612 and yolo11m
0.709 -> 0.683, so the damage is confined to the patched persons.

The patch clearly beats the noise control in every experiment, but at
0.2 x diagonal it mainly degrades person detections (lower confidence, worse
localisation) rather than erasing them. Only 1-5% of persons leave no confident
prediction at all. On INRIA the larger yolo11m is roughly half as susceptible as
yolo11n. On VOC the two victims end up with similar ASR (18.7% and 20.4%).

Each attacked dataset was checked against its source split. Every image is
present, `clean` is pixel-identical to the decoded original, labels are
identical to the source, and attacked pixels differ only inside person boxes.
The datasets keep the source split names (`test`, `test_500`) rather than `val`.
