# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TighNet is a learned MIDI onset correction system for piano loop recordings. It treats quantisation as a denoising task — a neural network predicts per-note timing corrections to recover clean performance from sloppy input, preserving musical character (swing, rubato) rather than snapping to a rigid grid. The target platform is iOS via CoreML.

## Commands

```bash
# Install (uses uv via asdf)
uv sync --extra dev

# Tests
uv run pytest                        # all tests
uv run pytest tests/test_model.py    # single file
uv run pytest -k test_name           # single test

# Lint & format
uv run ruff check tighnet tests
uv run ruff format tighnet tests

# Preprocess MIDI dataset (run once, creates JSON cache)
uv run python scripts/preprocess.py data/adl-piano-midi -o data/sliced.json

# Train
uv run python -m tighnet.train --cache data/sliced.json --device mps --arch dilated --epochs 100

# Export to CoreML
uv run python -m tighnet.export_coreml checkpoints/best.pt --output TighNet.mlpackage
```

## Architecture

**Data flow:** MIDI files → per-track grid alignment filter → bar-aligned slices → JSON cache (`scripts/preprocess.py`) → synthetic perturbation → (noisy, clean, offset) tensors → train model → checkpoint → CoreML export.

**Inference flow:** Recorded MIDI notes → onset tensor → model → per-frame offsets → cluster notes → apply corrections → corrected MIDI.

Key modules in `tighnet/`:

- **representation.py** — Frame-based onset tensor (5ms resolution, 3 channels: intensity, count, sustain density). Pitch-agnostic. Supports circular padding for loop boundaries.
- **model.py** — Two architectures: `DilatedCNN` (lightweight, ~200K params, 6 dilated conv blocks) and `UNet1D` (multi-scale with skip connections). Both: (batch, frames, 3) → (batch, frames, 1).
- **perturbation.py** — Generates synthetic training pairs by composing five noise types: jitter, drift, chord smear, large displacement, swing.
- **dataset.py** — `MidiQuantizationDataset` loads MIDI files, slices into 1/2/4-bar segments, generates training triples on-the-fly.
- **train.py** — Training loop with onset-weighted MSE loss (onsets weighted 1.0, non-onsets 0.01), AdamW + cosine annealing.
- **inference.py** — Quantisation pipeline: cluster notes within 30ms, apply offsets per cluster preserving internal spacing, snap loop boundaries, scale by strength parameter.
- **export_coreml.py** — PyTorch → CoreML conversion targeting iOS 16+.

## Conventions

- Python ≥ 3.10, line length 99 (ruff)
- Tests use pytest, focused on shape invariants and property checks rather than exact values
