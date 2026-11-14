# 🛡️ AntiStyler  
**Defending Object Detection Models Against Adversarial Patch Attacks Using Style Removal**

> Official implementation of the CVPR 2026 paper:  
> *AntiStyler: Defending Object Detection Models Against Adversarial Patch Attacks Using Style Removal*

---

## 🚀 Overview

Adversarial patch attacks can severely degrade object detection (OD) performance by inserting localized, malicious patterns into images.

**AntiStyler** is a **fast, training-free, and fully agnostic defense** that:

- 🧠 Detects adversarial patches via “style inconsistency”
- 🎭 Removes adversarial “random style” using a modified style transfer process
- 🎯 Masks suspicious regions with minimal impact on benign images
- ⚡ Runs in real-time (~10–12 FPS)

Unlike many prior defenses, AntiStyler:
- ✅ Does **not require retraining**
- ✅ Is **model-, patch-, and attack-agnostic**
- ✅ Preserves **benign performance**
- ✅ Works on **both digital and physical attacks**

---

## 🖼️ AntiStyler Pipeline

![Pipeline](AntiStyler%20pipeline.png)

AntiStyler consists of **four sequential phases**:

### (A) Style Removal
- Apply **random padding** to ensure the presence of “random style”
- Use a modified style transfer model (**AntiStyle**) to remove style
- Output: *AntiStyled image*

### (B) Filter Phase
- Compute **absolute difference** between original and AntiStyled image  
- Extract pixels with the largest changes → **raw mask**

### (C) Enhancement Phase
Apply morphological operations:
- Dilation → connect regions  
- Erosion → remove noise  
- Smoothing + thresholding → refine mask  

### (D) Mask Phase
- Apply mask to input image  
- Output: **defended image** → fed to detector

---

## 📊 Results

![Results](AntiStyler%20results%20COCO.png)

![Results](AntiStyler%20results%20INRIA.png)

### Key Findings

- 📈 **+8–15 mAP improvement** under attacks  
- 🟰 **No degradation on benign images**  
- ⚡ **~80–90 ms per image (~10–12 FPS)**  
- 🥇 Best **speed–robustness tradeoff** among SOTA defenses  

---

## 🔬 Method Intuition

AntiStyler is based on a key observation:

> Adversarial patches introduce **high-frequency, random “style” patterns** that differ from natural image statistics.

Instead of reconstructing images, AntiStyler:
1. Removes style (not content) using a modified loss:
   - Minimize content loss  
   - **Maximize style loss**
2. Regions that change the most → likely adversarial
3. These regions are **masked instead of reconstructed**

This avoids:
- Object misalignment  
- Localization errors  
- Heavy computation  

---

## 🧪 Supported Attacks

AntiStyler was evaluated against:

- Google Adversarial Patch  
- M-PGD  
- DPatch  
- TSEA  
- Printable patches  
- Naturalistic patches  
- Physical attacks (APRICOT, Superstore)

---

## ▶️ Quick Start

### Standalone Python experiments

Requires Python 3.11 or newer. Create an environment and install the pinned
dependencies (choose matching PyTorch wheels for your CUDA installation if needed):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the complete notebook workflow without Jupyter:

```bash
python run_experiment.py --config configs/experiment.json
```

Or run each component independently:

```bash
python run_attack.py --config configs/attack.json
python run_defense.py --config configs/defense.json
python run_detection.py --config configs/detection.json
```

The attack reads `image.jpg`; the defense example reads the attack's
`outputs/attack/attacked.pt`; the detection example reads `image.jpg`.
Change `input_image` in a configuration to process another image or saved tensor.
Paths inside JSON files are relative to that file. Pretrained weights download
on first use and are cached in `.cache/torch/`.

Every run saves its resolved configuration, seed, environment information,
images, exact tensors, and applicable losses/detections. Output directories
must be empty so previous experiments are preserved.

```bash
# Repeat with another fixed seed and a separate output directory.
python run_experiment.py --seed 123 --output-dir outputs/seed-123

# Choose a fresh random seed, recorded in the output for later replay.
python run_experiment.py --seed random --output-dir outputs/random-run

# Check configuration without importing PyTorch or downloading models.
python run_experiment.py --validate-config

# Run offline CPU tests; no pretrained weights are needed.
python -m unittest discover -s tests -v
```

### Dataset-level patch attack (VisDrone, YOLO11)

`run_patch_attack.py` trains one universal patch against an Ultralytics YOLO
detector on a YOLO-format training split, pastes it on every target object of
the evaluation split to write an attacked dataset, and reports COCO AP/AR and
attack success rate for clean, random-noise-control and attacked images:

```bash
python run_patch_attack.py --config configs/attack_visdrone.json
# Vehicles vanish instead of being relabelled as other classes:
python run_patch_attack.py --config configs/attack_visdrone_vanish.json
```

Person-vanishing patches for INRIAPerson and PASCAL VOC2007 against the default
COCO YOLO11n/YOLO11m first need the datasets converted to YOLO format with COCO
class indices:

```bash
python prepare_dataset.py --dataset inria --source data/INRIAPerson --output data/inria_yolo
python prepare_dataset.py --dataset voc2007 --source data/pascalVOC2007 --output data/voc2007_yolo
python run_patch_attack.py --config configs/attack_inria_yolo11n.json  # also _yolo11m, attack_voc2007_*
```

See [the patch attack documentation](docs/experiments.md#dataset-patch-attack)
for the threat model, metrics, data layout and results.

See [experiment documentation](docs/experiments.md) for the code structure,
parameter reference, reproducibility guarantees, and notebook fixes. The original
`AntiStyler_Demo.ipynb` remains available as an interactive reference.

 ---

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@InProceedings{Yankelev_2026_CVPR,
    author    = {Yankelev, Idan and Grolman, Edita and Levi, Yarin Yerushalmi and Giloni, Amit and Hofman, Omer and Shimizu, Toshiya and Elovici, Yuval and Shabtai, Asaf},
    title     = {{AntiStyler}: Defending Object Detection Models Against Adversarial Patch Attacks Using Style Removal},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {27936--27945}
}
```

## 🙌 Acknowledgements

- Ben-Gurion University  
- Fujitsu Research  

---

## ⭐ If You Find This Useful

Please ⭐ the repo and cite the paper!
