"""1D convolutional model for onset timing correction.

Two architectures are provided:
  - DilatedCNN: A stack of dilated 1D convolutions (lightweight, fast).
  - UNet1D: A 1D U-Net with downsampling/upsampling (better at multi-scale patterns).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .representation import NUM_CHANNELS


class ConvBlock(nn.Module):
    """A single dilated causal conv block with residual connection."""

    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, dilation=dilation
        )
        self.norm = nn.GroupNorm(1, channels)  # equivalent to LayerNorm over channels
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x + residual


class DilatedCNN(nn.Module):
    """Stack of dilated 1D convolutions for per-frame offset prediction.

    Input:  (batch, frames, NUM_CHANNELS)
    Output: (batch, frames, 1) — predicted timing offset per frame.
    """

    def __init__(
        self,
        in_channels: int = NUM_CHANNELS,
        hidden_channels: int = 64,
        num_layers: int = 6,
        kernel_size: int = 15,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, hidden_channels, 1)
        self.layers = nn.ModuleList(
            [
                ConvBlock(hidden_channels, kernel_size, dilation=2**i)
                for i in range(num_layers)
            ]
        )
        self.output_proj = nn.Conv1d(hidden_channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, frames, channels) -> (batch, channels, frames) for Conv1d
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        x = self.output_proj(x)
        # Back to (batch, frames, 1)
        return x.transpose(1, 2)


class DownBlock(nn.Module):
    """Downsampling block for U-Net: conv + pool."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 15):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.norm = nn.GroupNorm(1, out_ch)
        self.activation = nn.GELU()
        self.pool = nn.MaxPool1d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        skip = x
        x = self.pool(x)
        return x, skip


class UpBlock(nn.Module):
    """Upsampling block for U-Net: upsample + concat skip + conv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, kernel_size: int = 15):
        super().__init__()
        self.conv = nn.Conv1d(in_ch + skip_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.norm = nn.GroupNorm(1, out_ch)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2], mode="linear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class UNet1D(nn.Module):
    """1D U-Net for per-frame offset prediction.

    Captures multi-scale rhythmic structure through downsampling/upsampling.

    Input:  (batch, frames, NUM_CHANNELS)
    Output: (batch, frames, 1)
    """

    def __init__(
        self,
        in_channels: int = NUM_CHANNELS,
        base_channels: int = 32,
        depth: int = 3,
        kernel_size: int = 15,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, base_channels, 1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        ch = base_channels
        for i in range(depth):
            out_ch = ch * 2
            self.down_blocks.append(DownBlock(ch, out_ch, kernel_size))
            ch = out_ch

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(1, ch),
            nn.GELU(),
        )

        # Decoder
        self.up_blocks = nn.ModuleList()
        for i in range(depth):
            skip_ch = ch  # skip connection has same channels as down output
            out_ch = ch // 2
            self.up_blocks.append(UpBlock(ch, skip_ch, out_ch, kernel_size))
            ch = out_ch

        self.output_proj = nn.Conv1d(ch, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, C, T)
        x = self.input_proj(x)

        skips = []
        for block in self.down_blocks:
            x, skip = block(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for block, skip in zip(self.up_blocks, reversed(skips)):
            x = block(x, skip)

        x = self.output_proj(x)
        return x.transpose(1, 2)  # (B, T, 1)
