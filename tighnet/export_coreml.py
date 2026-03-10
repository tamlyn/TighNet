"""Export a trained PyTorch model to CoreML format for iOS deployment."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import DilatedCNN, UNet1D


def export_to_coreml(
    checkpoint_path: str,
    output_path: str = "TighNet.mlpackage",
    max_frames: int = 1728,  # 1600 + 2*64 circular padding
    num_channels: int = 3,
) -> Path:
    """Convert a trained checkpoint to CoreML.

    Args:
        checkpoint_path: Path to the .pt checkpoint.
        output_path: Where to save the .mlpackage.
        max_frames: Expected input length (frames + circular padding).
        num_channels: Number of input channels.

    Returns:
        Path to the saved CoreML model.
    """
    import coremltools as ct

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    arch = checkpoint.get("arch", "dilated")

    if arch == "unet":
        model = UNet1D()
    else:
        model = DilatedCNN()

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Trace the model with a dummy input.
    dummy_input = torch.randn(1, max_frames, num_channels)
    traced = torch.jit.trace(model, dummy_input)

    # Convert to CoreML.
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="onset_tensor",
                shape=(1, max_frames, num_channels),
                dtype=float,
            )
        ],
        outputs=[ct.TensorType(name="timing_offsets")],
        minimum_deployment_target=ct.target.iOS16,
    )

    out = Path(output_path)
    mlmodel.save(str(out))
    print(f"CoreML model saved to {out}")
    print(f"  Input: onset_tensor (1, {max_frames}, {num_channels})")
    print(f"  Output: timing_offsets (1, {max_frames}, 1)")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Export TighNet to CoreML")
    parser.add_argument("checkpoint", help="Path to .pt checkpoint")
    parser.add_argument("--output", default="TighNet.mlpackage")
    parser.add_argument("--max-frames", type=int, default=1728)
    args = parser.parse_args()

    export_to_coreml(args.checkpoint, args.output, args.max_frames)


if __name__ == "__main__":
    main()
