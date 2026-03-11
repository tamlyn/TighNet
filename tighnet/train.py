"""Training loop for the onset correction model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .dataset import MidiQuantizationDataset
from .model import DilatedCNN, UNet1D
from .perturbation import PerturbConfig


def onset_weighted_mse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    onset_mask: torch.Tensor,
    non_onset_weight: float = 0.01,
) -> torch.Tensor:
    """MSE loss weighted heavily toward frames that contain onsets.

    Args:
        predictions: (batch, frames, 1) predicted offsets.
        targets: (batch, frames, 1) ground-truth offsets.
        onset_mask: (batch, frames, 1) binary mask — 1 where onsets exist.
        non_onset_weight: Weight for frames without onsets (should predict ~0).
    """
    weights = torch.where(onset_mask > 0, 1.0, non_onset_weight)
    sq_error = (predictions - targets) ** 2
    return (sq_error * weights).mean()


def train(
    midi_dir: str | None = None,
    cache_path: str | None = None,
    output_dir: str = "checkpoints",
    arch: str = "dilated",
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    max_frames: int = 1600,
    bars_per_slice: int = 4,
    val_split: float = 0.1,
    device: str = "cpu",
    seed: int = 42,
) -> Path:
    """Train the onset correction model.

    Args:
        midi_dir: Path to directory containing MIDI files.
        cache_path: Path to pre-sliced JSON cache (from scripts/preprocess.py).
        output_dir: Where to save checkpoints.
        arch: Architecture — "dilated" or "unet".
        epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        max_frames: Maximum number of frames per example.
        bars_per_slice: How many bars per training slice.
        val_split: Fraction of data used for validation.
        device: Torch device.
        seed: Random seed.

    Returns:
        Path to the best checkpoint.
    """
    torch.manual_seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Dataset.
    source = cache_path or midi_dir
    dataset = MidiQuantizationDataset(
        midi_dir=midi_dir,
        cache_path=cache_path,
        max_frames=max_frames,
        bars_per_slice=bars_per_slice,
        perturb_config=PerturbConfig(),
        seed=seed,
    )
    print(f"Loaded {len(dataset)} training examples from {source}")

    if len(dataset) == 0:
        raise ValueError(f"No MIDI examples found in {midi_dir}")

    # Train/val split.
    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model.
    if arch == "unet":
        model = UNet1D()
    else:
        model = DilatedCNN()
    model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {arch}, parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_path = output_path / "best.pt"

    for epoch in range(1, epochs + 1):
        # Training.
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        batch_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for noisy, _clean, offset_target in batch_bar:
            noisy = noisy.to(device)
            offset_target = offset_target.to(device)

            # Onset mask: frames where the noisy input has onsets.
            onset_mask = (noisy[:, :, 1:2] > 0).float()

            predictions = model(noisy)
            loss = onset_weighted_mse(predictions, offset_target, onset_mask)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item() * noisy.size(0)
            train_count += noisy.size(0)
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_train_loss = train_loss_sum / max(train_count, 1)

        # Validation.
        model.eval()
        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for noisy, _clean, offset_target in val_loader:
                noisy = noisy.to(device)
                offset_target = offset_target.to(device)
                onset_mask = (noisy[:, :, 1:2] > 0).float()
                predictions = model(noisy)
                loss = onset_weighted_mse(predictions, offset_target, onset_mask)
                val_loss_sum += loss.item() * noisy.size(0)
                val_count += noisy.size(0)

        avg_val_loss = val_loss_sum / max(val_count, 1)

        print(
            f"Epoch {epoch}/{epochs} — "
            f"train loss: {avg_train_loss:.4f}, val loss: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "arch": arch},
                best_path,
            )
            print(f"  → Saved best model (val loss: {best_val_loss:.4f})")

    print(f"Training complete. Best model saved to {best_path}")
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TighNet onset correction model")
    parser.add_argument("midi_dir", nargs="?", help="Path to directory of MIDI files")
    parser.add_argument("--cache", help="Path to pre-sliced JSON cache (from scripts/preprocess.py)")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--arch", choices=["dilated", "unet"], default="dilated")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-frames", type=int, default=1600)
    parser.add_argument("--bars-per-slice", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.midi_dir and not args.cache:
        parser.error("Either midi_dir or --cache must be provided")

    train(
        midi_dir=args.midi_dir,
        cache_path=args.cache,
        output_dir=args.output_dir,
        arch=args.arch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_frames=args.max_frames,
        bars_per_slice=args.bars_per_slice,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
