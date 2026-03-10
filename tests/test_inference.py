"""Tests for the inference pipeline."""

import torch

from tighnet.inference import cluster_notes, quantize
from tighnet.model import DilatedCNN
from tighnet.representation import MidiNote


def test_cluster_single_notes():
    notes = [
        MidiNote(pitch=60, velocity=100, onset_ms=0.0, offset_ms=400.0),
        MidiNote(pitch=62, velocity=90, onset_ms=500.0, offset_ms=900.0),
        MidiNote(pitch=64, velocity=80, onset_ms=1000.0, offset_ms=1400.0),
    ]
    clusters = cluster_notes(notes, window_ms=30.0)
    assert len(clusters) == 3  # each note is its own cluster


def test_cluster_chord():
    notes = [
        MidiNote(pitch=60, velocity=100, onset_ms=500.0, offset_ms=900.0),
        MidiNote(pitch=64, velocity=90, onset_ms=510.0, offset_ms=910.0),
        MidiNote(pitch=67, velocity=80, onset_ms=520.0, offset_ms=920.0),
    ]
    clusters = cluster_notes(notes, window_ms=30.0)
    assert len(clusters) == 1  # all within 30ms = one cluster
    assert len(clusters[0]) == 3


def test_cluster_empty():
    assert cluster_notes([], window_ms=30.0) == []


def test_quantize_returns_same_count():
    notes = [
        MidiNote(pitch=60, velocity=100, onset_ms=100.0, offset_ms=400.0),
        MidiNote(pitch=62, velocity=90, onset_ms=600.0, offset_ms=900.0),
    ]
    model = DilatedCNN(in_channels=3, hidden_channels=16, num_layers=3, kernel_size=7)
    result = quantize(notes, 2000.0, model)
    assert len(result) == len(notes)


def test_quantize_preserves_duration():
    notes = [
        MidiNote(pitch=60, velocity=100, onset_ms=100.0, offset_ms=400.0),
        MidiNote(pitch=62, velocity=90, onset_ms=600.0, offset_ms=900.0),
    ]
    model = DilatedCNN(in_channels=3, hidden_channels=16, num_layers=3, kernel_size=7)
    result = quantize(notes, 2000.0, model)

    for orig, corrected in zip(notes, result):
        orig_dur = orig.offset_ms - orig.onset_ms
        corrected_dur = corrected.offset_ms - corrected.onset_ms
        assert abs(orig_dur - corrected_dur) < 0.01


def test_quantize_zero_strength():
    notes = [
        MidiNote(pitch=60, velocity=100, onset_ms=100.0, offset_ms=400.0),
        MidiNote(pitch=62, velocity=90, onset_ms=600.0, offset_ms=900.0),
    ]
    model = DilatedCNN(in_channels=3, hidden_channels=16, num_layers=3, kernel_size=7)
    result = quantize(notes, 2000.0, model, strength=0.0)

    for orig, corrected in zip(notes, result):
        # With strength=0, onset should be unchanged (within boundary snap threshold).
        assert abs(orig.onset_ms - corrected.onset_ms) < 0.01


def test_quantize_boundary_snap():
    # Note very close to loop start should snap to 0.
    notes = [
        MidiNote(pitch=60, velocity=100, onset_ms=10.0, offset_ms=400.0),
    ]
    model = DilatedCNN(in_channels=3, hidden_channels=16, num_layers=3, kernel_size=7)
    result = quantize(notes, 2000.0, model, strength=0.0, boundary_snap_ms=15.0)
    assert result[0].onset_ms == 0.0


def test_quantize_empty():
    model = DilatedCNN(in_channels=3, hidden_channels=16, num_layers=3, kernel_size=7)
    result = quantize([], 2000.0, model)
    assert result == []
