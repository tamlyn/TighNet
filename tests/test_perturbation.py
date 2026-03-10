"""Tests for the perturbation pipeline."""

import numpy as np

from tighnet.perturbation import PerturbConfig, perturb
from tighnet.representation import MidiNote


def _make_simple_notes() -> list[MidiNote]:
    """Create a simple 4-beat pattern at 120 BPM (beat every 500ms)."""
    return [
        MidiNote(pitch=60, velocity=100, onset_ms=0.0, offset_ms=400.0),
        MidiNote(pitch=62, velocity=90, onset_ms=500.0, offset_ms=900.0),
        MidiNote(pitch=64, velocity=80, onset_ms=1000.0, offset_ms=1400.0),
        MidiNote(pitch=65, velocity=85, onset_ms=1500.0, offset_ms=1900.0),
    ]


def test_perturb_returns_same_length():
    notes = _make_simple_notes()
    perturbed, offsets = perturb(notes, 2000.0, 120.0, rng=np.random.default_rng(0))
    assert len(perturbed) == len(notes)
    assert len(offsets) == len(notes)


def test_perturb_modifies_onsets():
    notes = _make_simple_notes()
    perturbed, offsets = perturb(notes, 2000.0, 120.0, rng=np.random.default_rng(42))

    # At least some notes should be shifted.
    any_shifted = any(
        abs(p.onset_ms - n.onset_ms) > 0.1 for p, n in zip(perturbed, notes)
    )
    assert any_shifted


def test_perturb_preserves_duration():
    notes = _make_simple_notes()
    perturbed, _ = perturb(notes, 2000.0, 120.0, rng=np.random.default_rng(42))

    for orig, pert in zip(notes, perturbed):
        orig_dur = orig.offset_ms - orig.onset_ms
        pert_dur = pert.offset_ms - pert.onset_ms
        assert abs(orig_dur - pert_dur) < 0.01, "Note duration should be preserved"


def test_perturb_stays_in_bounds():
    notes = _make_simple_notes()
    perturbed, _ = perturb(notes, 2000.0, 120.0, rng=np.random.default_rng(42))

    for note in perturbed:
        assert note.onset_ms >= 0.0
        assert note.onset_ms < 2000.0


def test_perturb_does_not_modify_original():
    notes = _make_simple_notes()
    original_onsets = [n.onset_ms for n in notes]
    perturb(notes, 2000.0, 120.0, rng=np.random.default_rng(42))

    for orig_onset, note in zip(original_onsets, notes):
        assert note.onset_ms == orig_onset, "Original notes should not be modified"


def test_zero_jitter_config():
    config = PerturbConfig(
        jitter_sigma_range=(0.0, 0.0),
        max_drift_ms=0.0,
        chord_smear_max_ms=0.0,
        large_displacement_prob=0.0,
        swing_prob=0.0,
    )
    notes = _make_simple_notes()
    perturbed, offsets = perturb(notes, 2000.0, 120.0, config=config, rng=np.random.default_rng(0))

    for orig, pert in zip(notes, perturbed):
        assert abs(orig.onset_ms - pert.onset_ms) < 0.01


def test_empty_notes():
    perturbed, offsets = perturb([], 2000.0, 120.0, rng=np.random.default_rng(0))
    assert len(perturbed) == 0
    assert len(offsets) == 0
