# CR X-Ray Ghost Artifact Removal

Removal of ghost (residual/afterglow) artifacts from sequential computed radiography images.
Computed radiography plates retain a faint latent image from the previous exposure when the
erase cycle is incomplete, so image `t` carries a scaled copy of image `t-1`. This package
contains the code, the design documents, and the current findings for that problem.

**Status as of 2026-09-12.** The linear physical model is confirmed on real data. A remover
runs end to end and clearly helps on frames with a strong ghost, but its effect on weak-ghost
frames ranges from negligible to unclear, and there is no ground truth to score it against.
Full-coverage removal is blocked on one missing ingredient, namely real paired data. Read
`RESULTS.md` before trusting any performance number, including the ones in the design
documents, then read `docs/plans/2026-05-28-real-data-acquisition-protocol.md` for how to
unblock the project.

---

## The model

The observed image is a linear superposition of the true exposure and scaled previous images.

```
I_t(x) = S_t(x) + sum_k alpha_k * (I_{t-k}(x) - bg_{t-k})
```

`alpha` is the ghost coupling coefficient. Measured on the 30-frame preliminary sequence, the
median `alpha` over the 11 frame pairs that carry a measurable ghost is 0.0040, rising to
0.0095 on the strongest frame. The ghost therefore carries well under 1% of the previous
image's signal, which puts it near the per-pixel noise floor and makes separability the central
difficulty of this project. After spatial averaging the ghost is unmistakable, with block R
squared reaching 0.584. Pixel by pixel it is not, with R squared near 0.013. See `RESULTS.md`
for the full table.

---

## What is implemented

| Module | Method | State |
|---|---|---|
| `src/models/physics_model.py` | Diffusion inpainting of ghost-contaminated air regions | Works, avoids halos, does not recover structure under the object |
| `src/models/linear_ghost.py` | Least-squares `alpha` estimation with a confidence gate | Safest logic, skips rather than over-corrects, but no script currently runs it with the gate enabled |
| `src/models/seamless_ghost.py` | Low-frequency swap that preserves noise texture | What `run_ghost_removal.py` actually calls, least visible correction, effect varies a lot by frame |
| `src/models/unet.py` | U-Net trained on synthetic ghost pairs | Trains but does not transfer to real ghosts, see `RESULTS.md` |
| `src/data/synthetic.py` | Synthetic (ghosted, clean, previous, ghost) pair generation | Works, but the synthetic ghost model does not match reality |
| `src/utils/dicom_utils.py` | DICOM load and save preserving headers | Works |

---

## Layout

```
.
├── README.md      # this file
├── DATA.md        # where the DICOM data goes, read this before running anything
├── RESULTS.md     # what has been established, with numbers
├── requirements.txt
├── docs/plans/    # design document and the paired-data acquisition protocol
├── src/           # library code, no scripts
├── scripts/       # entry points, each runnable standalone
└── tests/         # smoke test, verifies the install without DICOM data
```

---

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

**Use Python 3.11 or 3.12, not 3.13 or 3.14.** PyTorch publishes no wheels for 3.14, so
`pip install torch` fails outright there and the U-Net path becomes unavailable. Everything
else, namely the three classical removers, the analysis, and the figures, needs only NumPy,
SciPy, pydicom, and matplotlib, and those run fine on any recent version. For GPU training on a
remote machine, `scripts/setup_remote.sh` builds a conda environment on Python 3.12 with CUDA
12.4 wheels.

Verified on 2026-09-12. A clean Python 3.14 virtual environment installs the core dependencies
and passes the test suite, with the single torch test skipped.

Verify the install before touching data.

```bash
python -m pytest tests/ -v
```

---

## Running

Put the data in place first, following `DATA.md`. Then run from the repository root, since
every script resolves paths relative to the working directory.

```bash
# Main remover. Despite the naming this runs the SEAMLESS method, not the gated
# linear one, see RESULTS.md section 2.
# Writes cleaned DICOMs, comparison figures, and a per-image JSON log.
python scripts/run_ghost_removal.py
python scripts/run_ghost_removal.py --indices 4,6,18

# Quantify the linear model on the real sequence.
# Prints per-pair alpha, R^2, and suppression, writes figures/final/fig_alpha_fit.png.
python scripts/analyze_linear_fit.py

# Diffusion-inpainting baseline over the whole sequence.
python scripts/run_physics_baseline.py --n-previous 5

# Window center/width sweep, shows where the ghost appears and whether it was removed.
python scripts/generate_comparisons.py --indices 4,6,9,19,30

# U-Net training. Requires a GPU for anything beyond a smoke run.
python scripts/train_unet.py --epochs 100 --batch-size 16 --device cuda
python scripts/train_unet.py --epochs 10 --samples-per-epoch 500   # quick check

# Diagnostic. Separates a training failure from a synthetic-to-real transfer failure.
python scripts/diagnose_model.py --device cuda
```

`scripts/generate_proposal_figures.py` renders six figures for a funding proposal. Figures 1
and 2 come from the real data and the removal code. Figures 3 to 6 are schematic flowcharts
of planned work, not results.

---

## Start here if you are taking this over

1. Read `DATA.md` and put the DICOM files where it says.
2. Run `python -m pytest tests/ -v`, then `python scripts/run_ghost_removal.py --indices 4`.
   Image 4 has the strongest clean ghost and is the reference case. Expect a visible
   improvement, measured at about 40% coherent-ghost suppression. Older documents quote 97%
   for this frame under a different metric, see `RESULTS.md` section 2 before quoting either.
3. Read `RESULTS.md` for what is settled and what is not.
4. Read `docs/plans/2026-05-28-real-data-acquisition-protocol.md`. Acquiring 10 to 30 paired
   images is the single highest-value next step and it needs scanner access, not code.

---

## Known gaps

- There is no ground truth for most images, so removal quality cannot be scored with PSNR or
  SSIM. Validation today is visual plus the coherent-ghost suppression metric.
- `src/evaluation/` and `src/training/` are empty packages. Metrics and the training loop live
  inline in `scripts/train_unet.py`.
- Hyperparameters are command-line arguments rather than YAML configs.
- The dark/light dataset described in `DATA.md` has no processing script yet. It was acquired
  after the current code was written.
- Two naming and labelling mismatches are documented in `RESULTS.md` sections 1 and 2. Neither
  changes any result but both mislead a reader of the code.
- No trained U-Net checkpoint is included. The synthetic-trained model did not transfer, so
  retraining on real paired data is the intended path.
