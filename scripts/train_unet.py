"""Train U-Net for ghost removal with direct ghost supervision.

The model predicts the ghost pattern to subtract. The loss is computed
directly on the ghost prediction vs the known ghost contribution,
not on the full reconstructed image. This gives a much stronger
learning signal since the ghost is only ~0.5-1% of the image range.

Usage:
    python scripts/train_unet.py --epochs 100 --device cuda
    python scripts/train_unet.py --epochs 10 --samples-per-epoch 500
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.synthetic import GhostPatchDataset, load_all_images
from src.models.unet import GhostUNet


def ghost_loss(ghost_pred, ghost_target, weight_strength=200.0):
    """Magnitude-weighted L1 loss on the ghost prediction.

    The ghost is large only where the previous image had objects (a small
    fraction of pixels). Plain L1 is dominated by the many near-zero ghost
    pixels, so the model under-predicts the strong-ghost regions. Weighting
    by the true ghost magnitude forces accurate prediction where it matters.
    """
    weight = 1.0 + weight_strength * torch.abs(ghost_target)
    return (torch.abs(ghost_pred - ghost_target) * weight).mean()


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for inp, ghost_target in dataloader:
        inp, ghost_target = inp.to(device), ghost_target.to(device)
        _, ghost_pred = model(inp)

        loss = ghost_loss(ghost_pred, ghost_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0

    for inp, ghost_target in dataloader:
        inp, ghost_target = inp.to(device), ghost_target.to(device)
        _, ghost_pred = model(inp)
        loss = ghost_loss(ghost_pred, ghost_target)
        total_loss += loss.item()

    return total_loss / len(dataloader)


def inference_full_image(
    model, current: np.ndarray, previous: np.ndarray,
    dataset: GhostPatchDataset, device: torch.device,
    patch_size: int = 256, stride: int = 128,
) -> np.ndarray:
    model.eval()
    h, w = current.shape

    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    current_pad = np.pad(current, ((0, pad_h), (0, pad_w)), mode="reflect")
    previous_pad = np.pad(previous, ((0, pad_h), (0, pad_w)), mode="reflect")

    hp, wp = current_pad.shape
    ghost_sum = np.zeros((hp, wp), dtype=np.float64)
    weight = np.zeros((hp, wp), dtype=np.float64)

    with torch.no_grad():
        for y in range(0, hp - patch_size + 1, stride):
            for x in range(0, wp - patch_size + 1, stride):
                curr_p = current_pad[y:y + patch_size, x:x + patch_size]
                prev_p = previous_pad[y:y + patch_size, x:x + patch_size]

                inp = np.stack([
                    dataset.normalize(curr_p),
                    dataset.normalize(prev_p),
                ], axis=0).astype(np.float32)

                inp_t = torch.from_numpy(inp).unsqueeze(0).to(device)
                _, ghost_pred = model(inp_t)
                ghost_norm = ghost_pred.squeeze().cpu().numpy()

                # Convert ghost from normalized space back to pixel space
                ghost_pixels = ghost_norm * dataset.value_range

                ghost_sum[y:y + patch_size, x:x + patch_size] += ghost_pixels
                weight[y:y + patch_size, x:x + patch_size] += 1.0

    weight = np.maximum(weight, 1.0)
    ghost_map = ghost_sum / weight
    ghost_map = ghost_map[:h, :w]

    cleaned = current - ghost_map
    cleaned = np.clip(cleaned, 0, 65535)
    return cleaned


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="Train U-Net for ghost removal")
    parser.add_argument("--data-dir", type=str, default="data/raw/残影图像")
    parser.add_argument("--output-dir", type=str, default="data/processed/unet_cleaned")
    parser.add_argument("--figures-dir", type=str, default="results/figures/unet")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--samples-per-epoch", type=int, default=4000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--skip-inference", action="store_true")
    args = parser.parse_args()

    device = select_device(args.device)
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    figures_dir = Path(args.figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading images from {args.data_dir}...")
    images = load_all_images(args.data_dir)
    print(f"Loaded {len(images)} images ({images[0].shape})")

    n_train = max(len(images) - 6, len(images) * 4 // 5)
    train_images = images[:n_train]
    val_images = images[n_train:]
    print(f"Split: {len(train_images)} train, {len(val_images)} val")

    train_dataset = GhostPatchDataset(
        train_images, patch_size=args.patch_size,
        samples_per_epoch=args.samples_per_epoch, alpha_range=(0.005, 0.15),
    )
    val_dataset = GhostPatchDataset(
        val_images, patch_size=args.patch_size,
        samples_per_epoch=args.samples_per_epoch // 4, alpha_range=(0.005, 0.15),
    )
    val_dataset.global_min = train_dataset.global_min
    val_dataset.global_max = train_dataset.global_max
    val_dataset.value_range = train_dataset.value_range

    nw = args.num_workers if device.type == "cuda" else 0
    pin = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=nw, pin_memory=pin)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=nw, pin_memory=pin)

    model = GhostUNet(in_channels=2, base_filters=32).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    print(f"Normalization: min={train_dataset.global_min:.0f}, max={train_dataset.global_max:.0f}, range={train_dataset.value_range:.0f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    print(f"\nTraining for {args.epochs} epochs (direct ghost supervision)...")
    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / "unet_ghost_best.pt")
            marker = " *best*"

        lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch+1:3d}/{args.epochs}: train={train_loss:.8f}, val={val_loss:.8f}, lr={lr:.2e}, time={elapsed:.1f}s{marker}")

    torch.save(model.state_dict(), output_dir / "unet_ghost_final.pt")
    with open(output_dir / "norm_stats.json", "w") as f:
        json.dump({
            "global_min": train_dataset.global_min,
            "global_max": train_dataset.global_max,
            "value_range": train_dataset.value_range,
        }, f)

    print(f"\nBest val loss: {best_val_loss:.8f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label="Train", alpha=0.8)
    ax.plot(val_losses, label="Val", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Ghost Prediction Loss (L1)")
    ax.set_title("Training Progress (direct ghost supervision)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "training_loss.png", dpi=150)
    plt.close()

    if args.skip_inference:
        print("Skipping inference (--skip-inference)")
        return

    model.load_state_dict(torch.load(output_dir / "unet_ghost_best.pt", weights_only=True, map_location=device))

    print("\nRunning inference on all images...")
    import pydicom as dcm
    from src.models.physics_model import detect_air_mask

    data_dir = Path(args.data_dir)
    dcm_files = sorted(data_dir.glob("*.dcm"), key=lambda p: int(p.stem))

    for i, f in enumerate(dcm_files):
        ds = dcm.dcmread(str(f))
        current = ds.pixel_array.astype(np.float32)

        if i == 0:
            cleaned = current
        else:
            prev_f = dcm_files[i - 1]
            previous = dcm.dcmread(str(prev_f)).pixel_array.astype(np.float32)
            cleaned = inference_full_image(
                model, current, previous, train_dataset, device,
                patch_size=args.patch_size, stride=args.patch_size // 2,
            )

        cleaned_uint16 = np.clip(cleaned, 0, 65535).astype(np.uint16)
        ds_out = ds.copy()
        ds_out.PixelData = cleaned_uint16.tobytes()
        ds_out.save_as(str(output_dir / f.name))

        air = detect_air_mask(current)
        if air.sum() > 100:
            center = np.median(current[air])
            spread = np.std(current[air])
            vn, vx = center - spread * 6, center + spread * 6
        else:
            p5, p95 = np.percentile(current, [5, 95])
            vn, vx = p5, p95

        fig, axes = plt.subplots(1, 2, figsize=(14, 8))
        axes[0].imshow(current, cmap="gray", vmin=vn, vmax=vx)
        axes[0].set_title(f"Image {i+1} Original", fontsize=12, color="red")
        axes[0].axis("off")
        axes[1].imshow(cleaned, cmap="gray", vmin=vn, vmax=vx)
        axes[1].set_title(f"Image {i+1} U-Net Cleaned", fontsize=12, color="green")
        axes[1].axis("off")
        plt.tight_layout()
        plt.savefig(figures_dir / f"comparison_{i:02d}_{f.stem}.png", dpi=100)
        plt.close()

        print(f"  [{i+1:2d}/{len(dcm_files)}] {f.name}")

    print(f"\nDone. Results: {output_dir}/")
    print(f"Figures: {figures_dir}/")


if __name__ == "__main__":
    main()
