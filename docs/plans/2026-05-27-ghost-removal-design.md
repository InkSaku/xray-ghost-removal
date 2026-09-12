# X-Ray Ghost Artifact Removal - Design Document


> **Handover note added 2026-09-12.** This is the original design record, kept as written.
> Some performance figures in it, in particular the 97% reduction on image 4, and the
> description of `run_ghost_removal.py` as the confidence-gated linear remover, were not
> reproduced when the code was re-run on 2026-09-12. See `RESULTS.md` in the repository root
> for the re-measured numbers before quoting anything here.

> **Date**: 2026-05-27
> **Status**: Approved
> **Goal**: Remove ghost (residual) artifacts from sequential CR X-ray images

---

## Problem

Computed Radiography (CR) systems use phosphor plates that retain a residual image from previous exposures when not fully erased. The result is that image N contains faint "ghost" shapes from images N-1, N-2, etc. These ghosts become visible under adjusted window/level settings and compromise image quality.

**Data**: 30 sequential DICOM images (3048x2548, 16-bit, MONOCHROME2 CR) in `data/raw/残影图像/`. More paired data can be acquired from the CR system.

**Objective**: Given a ghosted image and its preceding image(s), produce a cleaned version containing only the current examination.

---

## Approach

Two-phase strategy. Phase 1 (physics-based subtraction) serves as a baseline and informs synthetic data generation. Phase 2 (U-Net) is the primary model.

### Phase 1: Physics-Based Subtraction

Model the observed image as a linear mixture:

```
I_obs(n) = I_true(n) + α₁·I_obs(n-1) + α₂·I_obs(n-2) + noise
```

**Coefficient estimation**: Identify background regions in image N (regions with minimal true signal) and regress pixel values against corresponding pixels in image N-1. The slope gives α₁. Repeat for N-2 to get α₂.

**Ghost removal**: Subtract the estimated ghost contribution and clamp to valid range:

```
I_clean(n) = clamp(I_obs(n) - α₁·I_obs(n-1) - α₂·I_obs(n-2))
```

### Phase 2: U-Net with Paired Training Data

**Architecture**: U-Net with ResNet34 encoder (ImageNet pre-trained, adapted to single-channel). Input is a 3-channel tensor [current_image, previous_image_1, previous_image_2]. Output is a single-channel cleaned image.

**Training data** (three tiers):

| Tier | Source | Volume | Purpose |
|------|--------|--------|---------|
| Synthetic | Clean images blended with ghost fractions from Phase 1 | ~5,000 pairs | Pre-training |
| Semi-real | Phase 1 cleaned images as pseudo ground truth | ~28 pairs | Bootstrapping |
| Real paired | Same object on fresh vs. ghosted plate (user acquires) | 50-100 pairs | Fine-tuning |

**Loss**: L1 + SSIM + perceptual (VGG feature matching).

**Training recipe**: Pre-train on synthetic → fine-tune on real paired data.

---

## Data Acquisition Protocol (for real paired data)

1. Fully erase the CR plate
2. Image object A on erased plate → `clean.dcm` (ground truth)
3. Image object B on same plate without full erase (creates ghost of B on plate)
4. Image object A again → `ghosted.dcm` (contains ghost of B)
5. Save `previous.dcm` (image of B, the ghost source)
6. Repeat with varied objects, exposures, and positions. Target: 50-100 pairs.

---

## Directory Structure

```
data/
├── raw/
│   ├── 残影图像/               # Existing 30 sequential images
│   └── paired/                 # New paired acquisitions
│       ├── pair_001/
│       │   ├── clean.dcm       # Ground truth (no ghost)
│       │   ├── ghosted.dcm     # Image with ghost artifact
│       │   └── previous.dcm    # Image that caused the ghost
│       └── ...
├── processed/
│   ├── physics_cleaned/        # Phase 1 outputs
│   └── synthetic_pairs/        # Generated synthetic training data

src/
├── data/
│   ├── dataset.py              # DICOM loading, normalization, patching
│   └── synthetic.py            # Synthetic ghost pair generation
├── models/
│   ├── physics_model.py        # Coefficient estimation + subtraction
│   └── unet_ghost.py           # U-Net architecture
├── training/
│   └── trainer.py              # Training loop
├── evaluation/
│   └── metrics.py              # PSNR, SSIM evaluation
└── utils/
    └── dicom_utils.py          # DICOM I/O preserving headers

configs/
├── data/
│   └── default.yaml            # Normalization, patch size, augmentation
├── model/
│   └── unet.yaml               # Architecture hyperparameters
└── train/
    └── default.yaml            # LR, epochs, batch size, loss weights

scripts/
├── run_physics_baseline.py     # Run Phase 1 on all images
├── generate_synthetic.py       # Generate synthetic training pairs
├── train.py                    # Train U-Net
└── evaluate.py                 # Evaluate on test pairs
```

---

## Implementation Plan

### Step 1: DICOM Utilities + Data Loading (~2 days)
- `src/utils/dicom_utils.py`: Load/save DICOM preserving headers, normalize to float32
- `src/data/dataset.py`: PyTorch Dataset for sequential images and paired data
- Config files for data parameters

### Step 2: Physics-Based Baseline (~3 days)
- `src/models/physics_model.py`: Background detection, coefficient estimation, ghost subtraction
- `scripts/run_physics_baseline.py`: Process all 30 images, save to `data/processed/physics_cleaned/`
- Visual comparison of before/after

### Step 3: Synthetic Data Generation (~2 days)
- `src/data/synthetic.py`: Generate ghosted images by blending clean images with ghost fractions
- `scripts/generate_synthetic.py`: Produce ~5,000 synthetic pairs
- Use Phase 1 estimated coefficients to set realistic ghost intensities

### Step 4: U-Net Architecture + Training (~5 days)
- `src/models/unet_ghost.py`: U-Net with 3-channel input, 1-channel output
- `src/training/trainer.py`: Training with L1 + SSIM + perceptual loss
- `scripts/train.py`: Main training script with config-driven hyperparameters
- Train on synthetic data first

### Step 5: Evaluation (~3 days)
- `src/evaluation/metrics.py`: PSNR, SSIM computation
- `scripts/evaluate.py`: Evaluate physics baseline and U-Net on held-out pairs
- Generate comparison figures

### Step 6: Fine-tuning with Real Paired Data (after acquisition)
- Fine-tune U-Net on real paired data
- Compare against synthetic-only model

---

## Key Decisions

- **Input channels**: Provide previous images explicitly rather than asking the model to learn ghost patterns blindly. This is more data-efficient.
- **Patch-based training**: Full 3048x2548 images are too large for GPU memory. Train on 512x512 patches with overlap, stitch at inference.
- **16-bit preservation**: Normalize to [0,1] float for training but output back to original 16-bit range in DICOM.
- **No ghost detection/scoring**: Scope is removal only. Detection can be added later.
