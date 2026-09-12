"""Run seamless ghost removal on the CR sequence.

Removes the ghost of the nearest previous image so the de-ghosted region
matches the surrounding background (level + noise texture, no halo).
Produces multi-window comparison figures so the ghost and its removal are
visible regardless of window center/width.

Pure NumPy/SciPy - runs on CPU.

Usage:
    python scripts/run_ghost_removal.py
    python scripts/run_ghost_removal.py --indices 4,6,18
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pydicom

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.seamless_ghost import remove_ghost_iterative
from src.models.physics_model import detect_air_mask
from src.utils.dicom_utils import save_dicom


def apply_window(img, wc, ww):
    return np.clip((img - (wc - ww / 2)) / max(ww, 1), 0, 1)


def ghost_windows(img):
    """A few WC/WW settings that reveal the ghost at different levels.

    Handles saturated air (returns a midtone window instead)."""
    p1, p50, p99 = np.percentile(img, [1, 50, 99])
    air = detect_air_mask(img)
    settings = [("full", (p1 + p99) / 2, max(p99 - p1, 1))]

    if air.sum() > 1000 and np.std(img[air]) > 1.0:
        amed = float(np.median(img[air]))
        astd = float(np.std(img[air]))
        settings.append((f"air x4 (WC={amed:.0f})", amed, max(astd * 8, 100)))
        settings.append((f"air x2 (WC={amed:.0f})", amed, max(astd * 4, 80)))
    else:
        # saturated air: probe midtones where the ghost may live
        mid = (p1 + p50) / 2
        settings.append((f"midtone (WC={mid:.0f})", mid, max((p50 - p1) * 0.5, 100)))
        settings.append((f"midtone narrow", mid, max((p50 - p1) * 0.25, 80)))
    return settings


def comparison_figure(previous, original, cleaned, idx, out_path):
    wins = ghost_windows(original)
    n = len(wins)
    fig, axes = plt.subplots(n, 3, figsize=(18, 5.5 * n))
    if n == 1:
        axes = axes[None, :]
    for r, (name, wc, ww) in enumerate(wins):
        axes[r, 0].imshow(apply_window(previous, wc, ww), cmap="gray", vmin=0, vmax=1)
        axes[r, 1].imshow(apply_window(original, wc, ww), cmap="gray", vmin=0, vmax=1)
        axes[r, 2].imshow(apply_window(cleaned, wc, ww), cmap="gray", vmin=0, vmax=1)
        for c in range(3):
            axes[r, c].axis("off")
        axes[r, 0].set_ylabel(name, fontsize=10, rotation=0, ha="right", va="center", labelpad=70)
    axes[0, 0].set_title(f"Previous (img {idx-1})", fontsize=12, fontweight="bold")
    axes[0, 1].set_title(f"Image {idx} original", fontsize=12, fontweight="bold", color="red")
    axes[0, 2].set_title(f"Image {idx} cleaned", fontsize=12, fontweight="bold", color="green")
    fig.suptitle(f"Image {idx}: ghost removal across window settings", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0.06, 0, 1, 0.98])
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight", facecolor="white")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Seamless ghost removal")
    parser.add_argument("--data-dir", type=str, default="data/raw/残影图像")
    parser.add_argument("--output-dir", type=str, default="data/processed/seamless_cleaned")
    parser.add_argument("--figures-dir", type=str, default="results/figures/seamless_cleaned")
    parser.add_argument("--indices", type=str, default=None, help="e.g. '4,6,18'; default all")
    parser.add_argument("--n-previous", type=int, default=3, help="ghost layers to peel off")
    args = parser.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.figures_dir); fig_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*.dcm"), key=lambda p: int(p.stem))
    images = [pydicom.dcmread(str(f)).pixel_array.astype(np.float32) for f in files]
    idx_map = {int(f.stem): k for k, f in enumerate(files)}

    if args.indices:
        targets = [int(x) for x in args.indices.split(",")]
    else:
        targets = [int(f.stem) for f in files]

    print(f"Loaded {len(images)} images. Processing {len(targets)} targets.\n")
    log = []
    n_applied = 0

    for img_id in targets:
        k = idx_map[img_id]
        current = images[k]
        f = files[k]

        if k == 0:
            save_dicom(current, pydicom.dcmread(str(f)), out_dir / f.name)
            print(f"  img {img_id}: first image, no previous - saved as-is")
            continue

        prev_start = max(0, k - args.n_previous)
        previous_list = images[prev_start:k][::-1]  # nearest first
        res = remove_ghost_iterative(current, previous_list)
        save_dicom(res.cleaned, pydicom.dcmread(str(f)), out_dir / f.name)

        if res.applied:
            n_applied += 1
            comparison_figure(previous_list[0], current, res.cleaned, img_id,
                              fig_dir / f"cmp_{img_id:02d}.png")

        log.append({"image": f.name, "applied": res.applied, "reason": res.reason,
                    "ghost_zone_fraction": res.ghost_zone_fraction,
                    "clean_air_fraction": res.clean_air_fraction})
        print(f"  img {img_id:2d}: {'APPLIED ' if res.applied else 'skipped '} {res.reason}")

    with open(out_dir / "removal_log.json", "w") as fp:
        json.dump(log, fp, indent=2)

    print(f"\nApplied to {n_applied} images. DICOMs: {out_dir}/  Figures: {fig_dir}/")


if __name__ == "__main__":
    main()
