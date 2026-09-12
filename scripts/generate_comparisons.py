"""Generate comparison figures showing ghost artifacts at different WC/WW.

The ghost from the previous image affects the current image differently
depending on the window center and width. This script creates figures
that sweep WC/WW to show WHERE and HOW the ghost appears, and whether
the cleaning removes it at each setting.

Usage:
    python scripts/generate_comparisons.py
    python scripts/generate_comparisons.py --indices 4,6,9,19,30
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pydicom

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.physics_model import detect_air_mask


def apply_window(image: np.ndarray, wc: float, ww: float) -> np.ndarray:
    """Apply DICOM-style window center/width, return [0, 1] display array."""
    vmin = wc - ww / 2
    vmax = wc + ww / 2
    return np.clip((image - vmin) / max(vmax - vmin, 1), 0, 1)


def compute_ghost_revealing_windows(
    original: np.ndarray,
) -> list[dict]:
    """Compute WC/WW settings that reveal the ghost at different intensity levels.

    The ghost from the previous image shows as subtle intensity shifts.
    Different WC/WW settings reveal different parts of the ghost because
    narrower windows amplify small pixel differences into visible contrast.
    """
    # Get intensity statistics
    p1, p25, p50, p75, p99 = np.percentile(original, [1, 25, 50, 75, 99])
    full_range = p99 - p1

    air = detect_air_mask(original)
    if air.sum() > 1000:
        air_med = float(np.median(original[air]))
        air_std = float(np.std(original[air]))
    else:
        air_med = float(p75)
        air_std = float(full_range * 0.02)

    # Object region statistics
    obj_vals = original[original < p50]
    if len(obj_vals) > 100:
        obj_med = float(np.median(obj_vals))
    else:
        obj_med = float(p25)

    # Edge/transition zone
    edge_level = (air_med + obj_med) / 2

    return [
        {
            "name": "Full range",
            "wc": (p1 + p99) / 2,
            "ww": full_range,
            "desc": "Standard clinical view",
        },
        {
            "name": f"WC={air_med:.0f} (air background)",
            "wc": air_med,
            "ww": air_std * 8,
            "desc": "Ghost in air/background region",
        },
        {
            "name": f"WC={air_med:.0f} narrow",
            "wc": air_med,
            "ww": air_std * 4,
            "desc": "Ghost in air, enhanced",
        },
        {
            "name": f"WC={edge_level:.0f} (edge zone)",
            "wc": edge_level,
            "ww": full_range * 0.15,
            "desc": "Ghost at object-air boundary",
        },
        {
            "name": f"WC={obj_med:.0f} (object region)",
            "wc": obj_med,
            "ww": full_range * 0.15,
            "desc": "Ghost overlapping object",
        },
        {
            "name": f"WC={air_med:.0f} ultra-narrow",
            "wc": air_med,
            "ww": air_std * 2,
            "desc": "Maximum ghost visibility",
        },
    ]


def generate_wc_ww_sweep(
    original: np.ndarray,
    cleaned: np.ndarray,
    previous: np.ndarray,
    image_idx: int,
    output_path: Path,
) -> None:
    """Grid figure: each row is a different WC/WW, columns are previous/original/cleaned.

    Shows how the ghost shape from the previous image appears at each
    window setting and whether it is removed in the cleaned version.
    """
    windows = compute_ghost_revealing_windows(original)
    n_rows = len(windows)

    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows))

    for row, win in enumerate(windows):
        wc, ww = win["wc"], win["ww"]

        # Previous image at the SAME window setting (shows what the ghost source looks like)
        axes[row, 0].imshow(apply_window(previous, wc, ww), cmap="gray", vmin=0, vmax=1)
        axes[row, 0].axis("off")

        # Original (with ghost) at this window
        axes[row, 1].imshow(apply_window(original, wc, ww), cmap="gray", vmin=0, vmax=1)
        axes[row, 1].axis("off")
        for spine in axes[row, 1].spines.values():
            spine.set_edgecolor("red")
            spine.set_linewidth(2)
            spine.set_visible(True)

        # Cleaned at the SAME window
        axes[row, 2].imshow(apply_window(cleaned, wc, ww), cmap="gray", vmin=0, vmax=1)
        axes[row, 2].axis("off")
        for spine in axes[row, 2].spines.values():
            spine.set_edgecolor("green")
            spine.set_linewidth(2)
            spine.set_visible(True)

        # Row label
        axes[row, 0].set_ylabel(
            f"{win['name']}\nWW={ww:.0f}\n({win['desc']})",
            fontsize=9, fontweight="bold", rotation=0, labelpad=120, va="center",
        )

    # Column headers
    axes[0, 0].set_title(f"Previous image {image_idx - 1}\n(ghost source, same WC/WW)", fontsize=11, fontweight="bold")
    axes[0, 1].set_title(f"Image {image_idx} Original\n(ghost visible)", fontsize=11, fontweight="bold", color="red")
    axes[0, 2].set_title(f"Image {image_idx} Cleaned\n(ghost removed?)", fontsize=11, fontweight="bold", color="green")

    fig.suptitle(
        f"Image {image_idx}: Ghost artifact at different Window Center / Window Width settings",
        fontsize=14, fontweight="bold", y=1.0,
    )
    plt.tight_layout(rect=[0.12, 0, 1, 0.98])
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def generate_correction_figure(
    original: np.ndarray,
    cleaned: np.ndarray,
    previous: np.ndarray,
    image_idx: int,
    output_path: Path,
) -> None:
    """Show the correction map (what was removed) alongside the ghost source."""
    diff = original.astype(np.float64) - cleaned.astype(np.float64)
    d_max = max(np.percentile(np.abs(diff), 99.5), 1)

    # Use a window where ghost is clearly visible
    air = detect_air_mask(original)
    if air.sum() > 1000:
        wc = float(np.median(original[air]))
        ww = float(np.std(original[air]) * 4)
    else:
        p5, p95 = np.percentile(original, [5, 95])
        wc, ww = (p5 + p95) / 2, (p95 - p5) * 0.2

    fig, axes = plt.subplots(1, 4, figsize=(28, 8))

    p2, p98 = np.percentile(previous, [2, 98])
    axes[0].imshow(previous, cmap="gray", vmin=p2, vmax=p98)
    axes[0].set_title(f"Image {image_idx-1} (ghost source)", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(apply_window(original, wc, ww), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Original (narrow window)", fontsize=12, color="red")
    axes[1].axis("off")

    axes[2].imshow(apply_window(cleaned, wc, ww), cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Cleaned (same window)", fontsize=12, color="green")
    axes[2].axis("off")

    im = axes[3].imshow(diff, cmap="RdBu_r", vmin=-d_max, vmax=d_max)
    axes[3].set_title("Correction map\n(original - cleaned)", fontsize=12, color="blue")
    axes[3].axis("off")
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    fig.suptitle(f"Image {image_idx}: Ghost Removal Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate ghost removal comparison figures")
    parser.add_argument("--data-dir", type=str, default="data/raw/残影图像")
    parser.add_argument("--cleaned-dir", type=str, default="data/processed/unet_cleaned")
    parser.add_argument("--output-dir", type=str, default="results/figures/unet_comparisons")
    parser.add_argument("--indices", type=str, default=None,
                        help="Comma-separated image indices (e.g. '4,6,9,19,30'). Default: all with ghost.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    cleaned_dir = Path(args.cleaned_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dcm_files = sorted(data_dir.glob("*.dcm"), key=lambda p: int(p.stem))
    cleaned_files = sorted(cleaned_dir.glob("*.dcm"), key=lambda p: int(p.stem))

    if not cleaned_files:
        print(f"No cleaned DICOM files in {cleaned_dir}/. Run training first.")
        return

    if args.indices:
        indices = [int(x.strip()) for x in args.indices.split(",")]
    else:
        indices = list(range(2, len(dcm_files) + 1))

    file_map = {int(f.stem): f for f in dcm_files}
    cleaned_map = {int(f.stem): f for f in cleaned_files}

    print(f"Generating comparison figures for {len(indices)} images...")

    for img_idx in indices:
        if img_idx not in file_map or img_idx not in cleaned_map:
            print(f"  Skipping image {img_idx} (not found)")
            continue

        prev_idx = img_idx - 1
        if prev_idx not in file_map:
            print(f"  Skipping image {img_idx} (no previous image)")
            continue

        original = pydicom.dcmread(str(file_map[img_idx])).pixel_array.astype(np.float32)
        cleaned = pydicom.dcmread(str(cleaned_map[img_idx])).pixel_array.astype(np.float32)
        previous = pydicom.dcmread(str(file_map[prev_idx])).pixel_array.astype(np.float32)

        generate_wc_ww_sweep(
            original, cleaned, previous, img_idx,
            output_dir / f"wcww_sweep_{img_idx:02d}.png",
        )

        generate_correction_figure(
            original, cleaned, previous, img_idx,
            output_dir / f"correction_{img_idx:02d}.png",
        )

        print(f"  Image {img_idx}: saved WC/WW sweep + correction figures")

    print(f"\nDone. Figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
