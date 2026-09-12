"""Run physics-based ghost removal on the sequential CR image dataset.

Usage:
    python scripts/run_physics_baseline.py
    python scripts/run_physics_baseline.py --data-dir data/raw/残影图像
    python scripts/run_physics_baseline.py --n-previous 5
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.physics_model import process_sequence, detect_air_mask
from src.utils.dicom_utils import load_sequence, save_dicom


def save_comparison_figure(
    original: np.ndarray,
    result,
    previous: np.ndarray | None,
    filename: str,
    output_path: Path,
) -> None:
    """Save comparison figure with matched windows and correction map."""
    has_prev = previous is not None
    n_cols = 4 if has_prev else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 8))

    air = detect_air_mask(original)
    if air.sum() > 100:
        air_vals = original[air]
        center = np.median(air_vals)
        spread = np.std(air_vals)
        vmin_n = center - spread * 6
        vmax_n = center + spread * 6
    else:
        p5, p95 = np.percentile(original, [5, 95])
        vmin_n, vmax_n = p5, p95

    idx = 0
    if has_prev:
        p2p, p98p = np.percentile(previous, [2, 98])
        axes[idx].imshow(previous, cmap="gray", vmin=p2p, vmax=p98p)
        axes[idx].set_title("Previous (ghost source)", fontsize=11)
        axes[idx].axis("off")
        idx += 1

    axes[idx].imshow(original, cmap="gray", vmin=vmin_n, vmax=vmax_n)
    axes[idx].set_title(f"Original: {filename}", fontsize=11)
    axes[idx].axis("off")
    idx += 1

    axes[idx].imshow(result.cleaned, cmap="gray", vmin=vmin_n, vmax=vmax_n)
    axes[idx].set_title(
        f"Cleaned (ghost zone: {result.ghost_zone_fraction*100:.1f}%)",
        fontsize=11,
    )
    axes[idx].axis("off")
    idx += 1

    corr = result.correction
    d_max = max(np.abs(np.percentile(corr, 1)), np.abs(np.percentile(corr, 99)), 1)
    axes[idx].imshow(corr, cmap="RdBu_r", vmin=-d_max, vmax=d_max)
    axes[idx].set_title("Correction applied", fontsize=11)
    axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Physics-based ghost removal")
    parser.add_argument(
        "--data-dir", type=str, default="data/raw/残影图像",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/processed/physics_cleaned",
    )
    parser.add_argument(
        "--figures-dir", type=str, default="results/figures/physics_baseline",
    )
    parser.add_argument(
        "--n-previous", type=int, default=3,
        help="Number of previous images to consider for ghost sources",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    figures_dir = Path(args.figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading images from {args.data_dir}...")
    sequence = load_sequence(args.data_dir)
    print(f"Loaded {len(sequence)} images")

    images = [arr for arr, _, _ in sequence]

    print(f"\nProcessing images (n_previous={args.n_previous})...")
    results = process_sequence(images, n_previous=args.n_previous)

    results_log = []
    for i, ((arr, ds, fname), result) in enumerate(zip(sequence, results)):
        save_dicom(result.cleaned, ds, output_dir / fname)

        results_log.append({
            "filename": fname,
            "index": i,
            "ghost_zone_fraction": result.ghost_zone_fraction,
            "clean_air_fraction": result.clean_air_fraction,
        })

        prev1 = images[i - 1] if i >= 1 else None
        save_comparison_figure(
            original=arr,
            result=result,
            previous=prev1,
            filename=fname,
            output_path=figures_dir / f"comparison_{i:02d}_{Path(fname).stem}.png",
        )

        print(
            f"  [{i+1:2d}/{len(sequence)}] {fname} "
            f"ghost={result.ghost_zone_fraction*100:.1f}% "
            f"clean_air={result.clean_air_fraction*100:.1f}%"
        )

    results_path = output_dir / "physics_baseline_results.json"
    with open(results_path, "w") as f:
        json.dump({"method": "diffusion_inpainting", "per_image": results_log}, f, indent=2)

    print(f"\nDone. Cleaned images: {output_dir}/")
    print(f"Figures: {figures_dir}/")


if __name__ == "__main__":
    main()
