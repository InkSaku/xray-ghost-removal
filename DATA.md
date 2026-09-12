# Data

No DICOM files are included in this package. The imaging data is about 2.8 GB and is
transferred separately. This file states exactly where each folder goes.

All scripts resolve paths relative to the repository root, so create `data/` as a sibling of
`src/` and `scripts/` and run every command from the repository root.

---

## Expected layout

```
<repo root>/
├── src/
├── scripts/
└── data/
    ├── raw/
    │   ├── 残影图像/                 # 30 sequential CR images, 1.dcm ... 30.dcm
    │   └── AI修残影例图/              # 62 dark/light images plus 拍摄参数记录.xlsx
    └── processed/                    # created by the scripts, do not populate by hand
        ├── linear_cleaned/
        ├── physics_cleaned/
        ├── seamless_cleaned/
        └── unet_cleaned/
```

Create the output folders once before the first run.

```bash
mkdir -p data/raw data/processed/{linear_cleaned,physics_cleaned,seamless_cleaned,unet_cleaned}
mkdir -p results/figures/{linear_cleaned,physics_baseline,seamless_cleaned,unet} figures/final
```

The folder names are Chinese and the scripts hard-code them, so keep them exactly as written.
`残影图像` means ghost images and `AI修残影例图` means AI ghost-repair example images.

---

## Dataset 1, the sequential ghost series

`data/raw/残影图像/` holds 30 files named `1.dcm` through `30.dcm`, about 15.5 MB each.

| Property | Value |
|---|---|
| Modality | CR |
| Size | 3048 x 2548 |
| Depth | 16-bit, MONOCHROME2 |
| Photometric convention | Air is high value, absorbing objects are low value |
| Study date in header | 20260515 |
| Identifiers | No patient name, ID, birth date, institution, or accession number in the headers |

The images were acquired back to back on the same plate, so image `n` carries the ghost of
image `n-1` and, more faintly, of earlier images. There is no ground truth. Every quantitative
claim in `RESULTS.md` comes from this series.

Image 4 is the reference case. It has the strongest clean ghost and is the fastest way to
confirm a working setup.

## Dataset 2, the dark/light pairs

`data/raw/AI修残影例图/` holds 62 DICOM files named `N-dark.dcm` and `N-light.dcm` for N from
1 to 31, plus `拍摄参数记录.xlsx`.

The spreadsheet records the exposure parameters, specifically tube voltage in kV, current in
mA, and time in ms, for the light image of each pair. Its header note states that images were
acquired in order with dark and light alternating, and that within each group the dark image
was acquired first. Recorded settings run at 70 kV with current and time varying from 32 to
100.

**No script processes this dataset yet.** It was acquired in July 2026, after the current code
was written, and integrating it is open work. It is not the paired ground-truth format
described in the acquisition protocol, so read
`docs/plans/2026-05-28-real-data-acquisition-protocol.md` before assuming it can be used as
supervision.

---

## Data that does not exist yet

The acquisition protocol calls for triplets of a clean image, a previous image, and a ghosted
image, laid out as follows.

```
data/raw/paired/
├── pair_001/
│   ├── clean.dcm
│   ├── previous.dcm
│   ├── ghosted.dcm
│   └── meta.json
└── pair_002/
```

Acquiring 10 to 30 of these is the single change that unblocks quantitative validation and
supervised training. The full procedure is in
`docs/plans/2026-05-28-real-data-acquisition-protocol.md`.

---

## Outputs

Scripts write cleaned DICOMs to `data/processed/`, comparison figures to `results/figures/`,
and analysis figures to `figures/final/`. Each removal run also writes a JSON log recording,
per image, whether the correction was applied, the fitted coefficients, the fit quality, and
the reason when it was skipped. Read that log before trusting any output image, since the
gated remover deliberately leaves low-confidence images untouched.
