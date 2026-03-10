"""Tests for the model architectures."""

import torch

from tighnet.model import DilatedCNN, UNet1D


def test_dilated_cnn_output_shape():
    model = DilatedCNN(in_channels=3, hidden_channels=32, num_layers=4, kernel_size=15)
    x = torch.randn(2, 1600, 3)
    out = model(x)
    assert out.shape == (2, 1600, 1)


def test_dilated_cnn_small_input():
    model = DilatedCNN(in_channels=3, hidden_channels=16, num_layers=3, kernel_size=7)
    x = torch.randn(1, 200, 3)
    out = model(x)
    assert out.shape == (1, 200, 1)


def test_unet1d_output_shape():
    model = UNet1D(in_channels=3, base_channels=16, depth=3, kernel_size=7)
    # U-Net with depth 3 needs input divisible by 8
    x = torch.randn(2, 1600, 3)
    out = model(x)
    assert out.shape == (2, 1600, 1)


def test_dilated_cnn_parameter_count():
    model = DilatedCNN(in_channels=3, hidden_channels=64, num_layers=6, kernel_size=15)
    param_count = sum(p.numel() for p in model.parameters())
    # Should be well under 1M parameters for CoreML deployment.
    assert param_count < 1_000_000, f"Model has {param_count} parameters, expected < 1M"


def test_model_gradient_flow():
    model = DilatedCNN(in_channels=3, hidden_channels=32, num_layers=4, kernel_size=15)
    x = torch.randn(1, 400, 3)
    target = torch.randn(1, 400, 1)

    out = model(x)
    loss = ((out - target) ** 2).mean()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.all(param.grad == 0), f"Zero gradient for {name}"
