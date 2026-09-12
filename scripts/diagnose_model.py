"""Diagnostic: isolate whether ghost removal fails at training or generalization.

Runs the trained model on:
  (A) a SYNTHETIC ghost pair (known ground truth) - tests if the model
      learned to remove the ghosts it was trained on
  (B) a REAL consecutive pair - tests if it transfers to real ghosts

Both go through the IDENTICAL inference path used in training.

Interpretation:
  - A works, B fails  -> synthetic/real gap. The model is fine; our
                         synthetic ghost model does not match reality.
                         Next step: acquire real paired data.
  - A fails too       -> training-side problem (task ambiguity / under-
                         training). Next step: change the formulation.

Usage:
    python scripts/diagnose_model.py --device cuda
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.synthetic import load_all_images, synthesize_ghost_pair
from src.models.unet import GhostUNet
from src.models.physics_model import detect_air_mask, detect_object_mask


def select_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def normalize(arr, gmin, vrange):
    return (arr - gmin) / vrange


def run_model_full(model, current, previous, gmin, vrange, device, patch=256, stride=128):
    """Run the model over a full image, return the predicted ghost map (pixel space)."""
    model.eval()
    h, w = current.shape
    pad_h = (patch - h % patch) % patch
    pad_w = (patch - w % patch) % patch
    cur = np.pad(current, ((0, pad_h), (0, pad_w)), mode="reflect")
    prv = np.pad(previous, ((0, pad_h), (0, pad_w)), mode="reflect")

    hp, wp = cur.shape
    gsum = np.zeros((hp, wp), dtype=np.float64)
    wsum = np.zeros((hp, wp), dtype=np.float64)

    with torch.no_grad():
        for y in range(0, hp - patch + 1, stride):
            for x in range(0, wp - patch + 1, stride):
                cp = cur[y:y+patch, x:x+patch]
                pp = prv[y:y+patch, x:x+patch]
                inp = np.stack([normalize(cp, gmin, vrange), normalize(pp, gmin, vrange)], 0).astype(np.float32)
                inp_t = torch.from_numpy(inp).unsqueeze(0).to(device)
                _, ghost = model(inp_t)
                g = ghost.squeeze().cpu().numpy() * vrange
                gsum[y:y+patch, x:x+patch] += g
                wsum[y:y+patch, x:x+patch] += 1.0

    wsum = np.maximum(wsum, 1.0)
    return (gsum / wsum)[:h, :w]


def ghost_zone_stats(label, current, reference_obj_src, ghost_map, cleaned):
    """Measure ghost magnitude in the zone where reference had objects."""
    air = detect_air_mask(current)
    obj = detect_object_mask(reference_obj_src)
    zone = air & obj
    clean_air = air & ~obj

    if zone.sum() < 100 or clean_air.sum() < 100:
        print(f"  [{label}] insufficient ghost zone")
        return

    air_level = np.median(current[clean_air])
    before_dev = air_level - np.median(current[zone])
    after_dev = air_level - np.median(cleaned[zone])
    pred_correction = np.median(np.abs(ghost_map[zone]))

    print(f"  [{label}] ghost zone deviation from air:")
    print(f"      before cleaning: {before_dev:8.1f} pixels")
    print(f"      after cleaning:  {after_dev:8.1f} pixels")
    print(f"      model correction applied (median |ghost|): {pred_correction:8.1f} pixels")
    if abs(before_dev) > 1:
        print(f"      reduction: {(1 - abs(after_dev)/abs(before_dev))*100:5.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/raw/残影图像")
    parser.add_argument("--model-dir", type=str, default="data/processed/unet_cleaned")
    parser.add_argument("--output-dir", type=str, default="results/figures/diagnostic")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--synthetic-alpha", type=float, default=0.05)
    parser.add_argument("--clean-idx", type=int, default=24, help="Image index (1-based) to use as clean base for synthetic test")
    parser.add_argument("--ghost-idx", type=int, default=5, help="Image index (1-based) for synthetic ghost source")
    parser.add_argument("--real-idx", type=int, default=4, help="Real image index (1-based) to test")
    args = parser.parse_args()

    device = select_device(args.device)
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load norm stats
    with open(Path(args.model_dir) / "norm_stats.json") as f:
        stats = json.load(f)
    gmin = stats["global_min"]
    vrange = stats.get("value_range", stats["global_max"] - stats["global_min"])
    print(f"Norm: min={gmin:.0f}, range={vrange:.0f}")

    # Load model
    model = GhostUNet(in_channels=2, base_filters=32).to(device)
    ckpt = Path(args.model_dir) / "unet_ghost_best.pt"
    model.load_state_dict(torch.load(str(ckpt), weights_only=True, map_location=device))
    print(f"Loaded {ckpt}")

    images = load_all_images(args.data_dir)

    # ---- TEST A: SYNTHETIC ghost (known ground truth) ----
    print("\n=== TEST A: synthetic ghost (model should remove this) ===")
    clean = images[args.clean_idx - 1]
    ghost_src = images[args.ghost_idx - 1]
    ghosted, true_ghost, _ = synthesize_ghost_pair(
        clean, ghost_src, alpha_range=(args.synthetic_alpha, args.synthetic_alpha), blur_sigma_range=(0, 0),
    )
    ghost_map_A = run_model_full(model, ghosted, ghost_src, gmin, vrange, device)
    cleaned_A = ghosted - ghost_map_A
    ghost_zone_stats("SYNTHETIC", ghosted, ghost_src, ghost_map_A, cleaned_A)
    print(f"      TRUE injected ghost (median |.| in zone): "
          f"{np.median(np.abs(true_ghost[detect_air_mask(ghosted) & detect_object_mask(ghost_src)])):.1f} pixels")

    # ---- TEST B: REAL consecutive pair ----
    print("\n=== TEST B: real consecutive pair (the actual goal) ===")
    real = images[args.real_idx - 1]
    real_prev = images[args.real_idx - 2]
    ghost_map_B = run_model_full(model, real, real_prev, gmin, vrange, device)
    cleaned_B = real - ghost_map_B
    ghost_zone_stats("REAL", real, real_prev, ghost_map_B, cleaned_B)

    # ---- Figure ----
    def narrow(img):
        air = detect_air_mask(img)
        if air.sum() > 1000:
            c = np.median(img[air]); s = np.std(img[air])
        else:
            c = np.median(img); s = (np.percentile(img, 95) - np.percentile(img, 5)) / 4
        return c - 4 * s, c + 4 * s

    fig, axes = plt.subplots(2, 4, figsize=(26, 13))

    vminA, vmaxA = narrow(ghosted)
    axes[0, 0].imshow(clean, cmap="gray", vmin=vminA, vmax=vmaxA); axes[0, 0].set_title("A: TRUE clean (target)"); axes[0, 0].axis("off")
    axes[0, 1].imshow(ghosted, cmap="gray", vmin=vminA, vmax=vmaxA); axes[0, 1].set_title(f"A: synthetic ghosted (alpha={args.synthetic_alpha})", color="red"); axes[0, 1].axis("off")
    axes[0, 2].imshow(cleaned_A, cmap="gray", vmin=vminA, vmax=vmaxA); axes[0, 2].set_title("A: model cleaned", color="green"); axes[0, 2].axis("off")
    err = cleaned_A - clean
    em = max(np.percentile(np.abs(err), 99), 1)
    axes[0, 3].imshow(err, cmap="RdBu_r", vmin=-em, vmax=em); axes[0, 3].set_title("A: residual error (cleaned - truth)"); axes[0, 3].axis("off")

    vminB, vmaxB = narrow(real)
    axes[1, 0].imshow(real_prev, cmap="gray", vmin=np.percentile(real_prev, 2), vmax=np.percentile(real_prev, 98)); axes[1, 0].set_title(f"B: previous image {args.real_idx-1}"); axes[1, 0].axis("off")
    axes[1, 1].imshow(real, cmap="gray", vmin=vminB, vmax=vmaxB); axes[1, 1].set_title(f"B: real image {args.real_idx}", color="red"); axes[1, 1].axis("off")
    axes[1, 2].imshow(cleaned_B, cmap="gray", vmin=vminB, vmax=vmaxB); axes[1, 2].set_title("B: model cleaned", color="green"); axes[1, 2].axis("off")
    gm = max(np.percentile(np.abs(ghost_map_B), 99), 1)
    axes[1, 3].imshow(ghost_map_B, cmap="RdBu_r", vmin=-gm, vmax=gm); axes[1, 3].set_title("B: predicted ghost map"); axes[1, 3].axis("off")

    plt.suptitle("Diagnostic: synthetic (row A) vs real (row B) ghost removal", fontsize=15, fontweight="bold")
    plt.tight_layout()
    fig_path = out_dir / "diagnostic.png"
    plt.savefig(str(fig_path), dpi=130, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nSaved {fig_path}")
    print("\nINTERPRETATION:")
    print("  If row A is clean but row B is not -> synthetic/real gap (need real paired data).")
    print("  If row A still shows the ghost     -> training-side problem (change formulation).")


if __name__ == "__main__":
    main()
