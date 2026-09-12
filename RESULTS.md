# Results

What is established, what is not, and where the numbers come from. Every figure below was
regenerated on 2026-09-12 from the 30-frame sequence in `data/raw/残影图像/`. Where a number
disagrees with an older document, that is called out rather than quietly reconciled.

---

## 1. The linear superposition model holds, but only after spatial averaging

Reproduce with `python scripts/analyze_linear_fit.py`.

| Quantity | Value |
|---|---|
| Consecutive frame pairs analysed | 29 of 30 frames |
| Pairs with a physically plausible positive coefficient, 0.001 to 0.02 | 11 of 29 |
| Background pixels per regression, median | 6.46 million at full resolution |
| Coupling coefficient alpha, median | 0.0040, interquartile range 0.0026 to 0.0111 |
| Per-pixel R squared, median | 0.013 |
| Block-averaged R squared, median | 0.010, maximum 0.584 |
| Strongest frame, image 4 | alpha 0.0095, block R squared 0.584 |
| Per-pixel noise sigma, image 4 | about 239 at 16-bit |

The reading is straightforward. Pixel by pixel the fit explains almost nothing, with R squared
near 0.013, because a ghost carrying under 1% of the previous image sits below a per-pixel
noise sigma of roughly 239. Average over 16 by 16 blocks and the strongest frame reaches R
squared 0.584, which is what confirms the ghost is a coherent scaled copy of the previous frame
rather than noise. The model is real. The per-pixel signal to noise ratio is the obstacle.

Only 11 of 29 pairs show a usable ghost at all. In the rest the ghost is either absent or
buried, and the fitted coefficient falls outside the plausible range. Any claim about method
performance that averages over all 30 frames is therefore averaging mostly over frames with no
measurable ghost.

**Discrepancy to be aware of.** `docs/plans/2026-05-28-real-data-acquisition-protocol.md`
quotes alpha of about 0.0085 for a clean case. The median over the 11 valid pairs measured here
is 0.0040, and image 4 specifically gives 0.0095. The older figure is consistent with the
strong frames but not with the sequence median, so treat 0.0085 as a best-case value.

**Small bug worth fixing.** `scripts/analyze_linear_fit.py` prints its block-averaged statistic
as `空间分块(32px)` while `fit_pair` uses `blk=16` by default. The label says 32 pixels and the
computation uses 16. The numbers above are for 16-pixel blocks.

---

## 2. What the removers actually do

Reproduce with `python scripts/run_ghost_removal.py --indices 4,6,18`.

Note that despite its name and despite how it is described in the acquisition protocol
document, `run_ghost_removal.py` calls `remove_ghost_iterative` from `seamless_ghost.py`, not
the confidence-gated linear remover in `linear_ghost.py`. The gated linear remover is currently
reached only by `generate_proposal_figures.py`, and there with `r2_threshold=0.0`, which
disables the gate. This matters because the safety argument for the project, namely that a
low-confidence image is left untouched, applies to the linear remover and not to the script
that is normally run.

Measured coherent-ghost suppression in the ghost zone, defined as the reduction in the standard
deviation of 16-pixel block means over the region that is air in the current frame and object in
the previous frame.

| Image | Ghost zone, share of frame | Block std before | Block std after | Suppression |
|---|---|---|---|---|
| 4 | 6.4% | 256.1 | 152.3 | 40.5% |
| 6 | 9.3% | 2925.1 | 2813.9 | 3.8% |
| 18 | 1.6% | 983.3 | 1384.0 | -40.8% |

All three were reported as applied by the script. On image 4, the reference case, the
correction clearly helps. On image 6 it does almost nothing. On image 18 this metric gets worse
after the correction.

Two caveats before reading too much into that table. The ghost zone also contains real anatomy
from the current exposure, so block variance there is not a pure ghost measurement, and the
seamless method deliberately rewrites the low-frequency component, which this metric penalises.
The negative value on image 18 is therefore not proof of damage. What it does show is that
there is no metric available today that cleanly separates a good correction from a bad one.

**Discrepancy to be aware of.** Both design documents state a 97% reduction on image 4. That
number was produced by a different suppression definition than the one above, and the run in
this session did not reproduce it under the block-variance metric, which gives 40.5%. Neither
number is validated against ground truth. Do not quote 97% without re-deriving it.

---

## 3. The U-Net does not transfer

The U-Net in `src/models/unet.py` trains on synthetic pairs from `src/data/synthetic.py`, where
a ghost is planted with a known coefficient, and it learns to predict the planted ghost.
`scripts/diagnose_model.py` exists to separate the two ways this can fail, specifically a
training failure, where the model cannot remove even the synthetic ghosts it was trained on,
against a transfer failure, where it handles synthetic ghosts and fails on real ones.

The recorded outcome is a transfer failure. The synthetic ghost model does not match the real
plate physics closely enough. `data/processed/unet_cleaned/` and `results/figures/unet/` are
empty and no checkpoint was kept, so there is no trained model to inherit.

This is not a reason to rework the architecture. It is the same conclusion the other three
methods reached, namely that the missing ingredient is real supervision rather than a better
estimator.

---

## 4. The one thing that unblocks everything

Every method here estimates alpha from the image itself and then has no way to check the
estimate. Strong clean ghosts can be verified by eye. Weak ghosts cannot, so every estimator
either under-corrects or over-corrects and no tuning resolves it.

Paired acquisition removes the ambiguity. With a clean image, a previous image, and a ghosted
image of the same object, the true ghost is exactly `ghosted - clean` and the true coefficient
follows directly. That makes PSNR and SSIM meaningful, replaces the R squared confidence
heuristic with a measured prior, and gives the U-Net real targets.

Ten pairs varying only the exposure of the ghost source are enough to confirm the linear model
on real data and produce an alpha against dose curve. The full protocol is in
`docs/plans/2026-05-28-real-data-acquisition-protocol.md`. This needs scanner time, not code.

---

## 5. Open items

- The dark/light dataset in `data/raw/AI修残影例图/`, 62 images acquired in July 2026, has no
  processing script. Whether the dark frame can serve as a ghost-free reference for its light
  partner has not been tested, and that test is cheap.
- No metric cleanly separates a good correction from a bad one without ground truth. The
  block-variance suppression used in section 2 is a stopgap.
- `run_ghost_removal.py` is named and documented as the gated remover but runs the seamless one,
  as described in section 2.
- The block size label in `analyze_linear_fit.py` does not match the code, as described in
  section 1.
- Removal parameters, specifically `alpha_cap`, `r2_threshold`, `min_air_fraction`, and the
  diffusion settings, were set by hand and never swept.
