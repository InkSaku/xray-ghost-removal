# Real Paired-Data Acquisition Protocol for Ghost Removal


> **Handover note added 2026-09-12.** This is the original design record, kept as written.
> Some performance figures in it, in particular the 97% reduction on image 4, and the
> description of `run_ghost_removal.py` as the confidence-gated linear remover, were not
> reproduced when the code was re-run on 2026-09-12. See `RESULTS.md` in the repository root
> for the re-measured numbers before quoting anything here.

> **Date**: 2026-05-28
> **Purpose**: Acquire ground-truth (clean, ghosted) pairs so ghost removal can be validated and trained on real physics instead of synthetic assumptions.

---

## Why this is needed

Every method tried so far (physics linear subtraction, diffusion inpainting, U-Net with synthetic data) hit the same wall: **no ground truth**. We can verify removal only when the ghost is strong and clean (e.g. image 4, 97% reduction), because there we can see it. For weak ghosts buried in air noise, we cannot tell whether an estimated coefficient is right, so every estimator either under- or over-corrects, and no amount of tuning fixes this.

What we *did* prove: the ghost is linear, `current = true + sum_k alpha_k * (previous_k - bg_k)`, and for a clean case `alpha ~= 0.0085`. The single unknown blocking reliable removal is **how alpha behaves** across exposure level, position, and time between exposures. Real paired data measures this directly.

---

## What a "pair" is

| File | How it is captured | Role |
|------|-------------------|------|
| `clean.dcm` | Image of object A on a **fully erased** plate | Ground-truth target (no ghost) |
| `previous.dcm` | Image of object B (the ghost source) | Known ghost source |
| `ghosted.dcm` | Image of object A again, on the plate **after** B's exposure (not fully erased) | Input with a real ghost of B |

With `clean` as ground truth, the real ghost is exactly `ghosted - clean`, and the true coefficient is `alpha = mean((ghosted - clean)) / mean((previous - bg))` in the relevant region. No assumptions.

---

## Acquisition procedure (per pair)

1. **Fully erase** the CR plate (run the reader's erase cycle, or expose to bright light per vendor spec, then read-and-discard).
2. Image **object A**. Save as `clean.dcm`. This is the ghost-free reference.
3. **Fully erase** again.
4. Image **object B**. Save as `previous.dcm`. (B is the ghost source.)
5. **Without fully erasing**, immediately image **object A** again. Save as `ghosted.dcm`.
6. Record metadata for the pair (see below).

The critical control: steps 2 and 5 image the **same object A in the same position**, so `ghosted - clean` isolates exactly the ghost of B. Keep A and the tube/plate geometry fixed between steps 2 and 5.

---

## What to vary across pairs (to map alpha)

Capture pairs spanning these axes so the model/estimator learns how alpha changes:

| Variable | Range to cover | Why |
|----------|---------------|-----|
| Exposure of ghost source B | low / medium / high mAs | alpha likely scales with B's dose |
| Time gap between B and ghosted A | immediate, 30 s, 2 min | phosphor residual decays over time |
| Object B type | sharp-edged, large flat, fine structure | tests spatial fidelity of the ghost |
| Position on plate | center, edges, corners | tests spatial variation of alpha |
| Exposure of A | low / normal | tests interaction with the underlying image |

**Target: 50-100 pairs.** Even 20-30 well-spread pairs are enough to measure alpha's behavior and validate the linear model.

---

## Directory layout

```
data/raw/paired/
├── pair_001/
│   ├── clean.dcm        # object A, erased plate (ground truth)
│   ├── previous.dcm     # object B (ghost source)
│   ├── ghosted.dcm      # object A again, after B (real ghost)
│   └── meta.json        # see below
├── pair_002/
└── ...
```

`meta.json` per pair:
```json
{
  "object_A": "aluminum step wedge",
  "object_B": "circular phantom",
  "kvp": 70,
  "mas_B": 5.0,
  "mas_A": 4.0,
  "time_gap_seconds": 30,
  "position": "center",
  "notes": "fully erased before clean and before previous"
}
```

---

## How this data unblocks each method

1. **Validate the linear model.** Compute `alpha_true = (ghosted - clean) / (previous - bg)` per pair. If alpha is stable and the residual `ghosted - clean - alpha*(previous-bg)` is near zero, the linear model is confirmed and we can deploy the simple subtractor with a calibrated alpha (and its dependence on exposure/time).

2. **Calibrate the gated remover.** Replace the R^2 confidence heuristic with a real alpha prior. Set `alpha_cap` and the time/exposure dependence from measured values, removing the guesswork that currently forces us to skip noisy images.

3. **Train the U-Net with real supervision.** Use `(ghosted, previous)` as input and `clean` as target. This removes the synthetic-ghost assumption that was the root weakness. With real targets, the alpha ambiguity disappears because the network sees the true mapping.

---

## Minimum viable first batch

If acquiring 50-100 pairs is a lot up front, start with **10 pairs** varying only exposure of B (5 levels x 2 repeats) at center position, immediate gap. That alone will:
- confirm the linear model on real data,
- give the alpha-vs-dose curve,
- let us replace the current heuristic gate with a calibrated one.

Then expand to position and time-gap variation.

---

## Current deliverable (works today, no new data)

`scripts/run_ghost_removal.py` runs the confidence-gated linear remover:
- cleans strong, clean ghosts (image 4: 97% reduction),
- leaves low-confidence images untouched (safe for QC - never corrupts),
- outputs cleaned DICOMs + comparison figures + a per-image log.

This is the safe interim tool until real paired data enables full-coverage removal.
```
python scripts/run_ghost_removal.py
```
