"""Synthetic perturbation pipeline for generating (clean, noisy) training pairs.

Applies realistic timing imperfections to clean MIDI performances so the
denoising model can learn to correct them.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from .representation import MidiNote


@dataclass
class PerturbConfig:
    """Configuration for the perturbation pipeline."""

    # Gaussian jitter standard deviation range in ms.
    jitter_sigma_range: tuple[float, float] = (10.0, 60.0)

    # Systematic drift: max drift in ms over the full loop.
    max_drift_ms: float = 40.0

    # Chord smearing: max spread in ms for simultaneous onsets.
    chord_smear_max_ms: float = 80.0

    # Probability of a large random displacement per onset.
    large_displacement_prob: float = 0.05
    large_displacement_range: tuple[float, float] = (50.0, 150.0)

    # Swing perturbation probability and max offset.
    swing_prob: float = 0.2
    swing_max_offset_ms: float = 30.0

    # Window (ms) to consider notes as simultaneous (for chord detection).
    chord_window_ms: float = 10.0


def _group_chords(notes: list[MidiNote], window_ms: float) -> list[list[int]]:
    """Group note indices into chord clusters based on onset proximity."""
    if not notes:
        return []

    sorted_indices = sorted(range(len(notes)), key=lambda i: notes[i].onset_ms)
    groups: list[list[int]] = [[sorted_indices[0]]]

    for idx in sorted_indices[1:]:
        if notes[idx].onset_ms - notes[groups[-1][0]].onset_ms <= window_ms:
            groups[-1].append(idx)
        else:
            groups.append([idx])

    return groups


def perturb(
    notes: list[MidiNote],
    duration_ms: float,
    tempo_bpm: float,
    time_sig_numerator: int = 4,
    config: PerturbConfig | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[list[MidiNote], np.ndarray]:
    """Apply realistic timing perturbations to a clean note sequence.

    Args:
        notes: Clean MIDI notes (will not be modified).
        duration_ms: Loop duration in milliseconds.
        tempo_bpm: Tempo in BPM (used for subdivision-aware perturbations).
        time_sig_numerator: Beats per bar.
        config: Perturbation configuration.
        rng: Numpy random generator for reproducibility.

    Returns:
        A tuple of (perturbed_notes, offsets) where offsets[i] is the timing
        shift applied to note i in milliseconds (perturbed - original).
    """
    if config is None:
        config = PerturbConfig()
    if rng is None:
        rng = np.random.default_rng()

    perturbed = copy.deepcopy(notes)
    offsets = np.zeros(len(notes), dtype=np.float32)

    if not notes:
        return perturbed, offsets

    # --- 1. Gaussian jitter ---
    sigma = rng.uniform(*config.jitter_sigma_range)
    jitter = rng.normal(0.0, sigma, size=len(notes)).astype(np.float32)

    # --- 2. Systematic drift ---
    # Linear drift from 0 at loop start to a random value at loop end.
    drift_amount = rng.uniform(-config.max_drift_ms, config.max_drift_ms)
    drift = np.array(
        [drift_amount * (n.onset_ms / duration_ms) for n in notes],
        dtype=np.float32,
    )

    # --- 3. Chord smearing ---
    smear = np.zeros(len(notes), dtype=np.float32)
    groups = _group_chords(notes, config.chord_window_ms)
    for group in groups:
        if len(group) > 1:
            spread = rng.uniform(0.0, config.chord_smear_max_ms)
            group_smear = rng.uniform(-spread / 2, spread / 2, size=len(group)).astype(
                np.float32
            )
            for i, idx in enumerate(group):
                smear[idx] = group_smear[i]

    # --- 4. Random large displacements ---
    large_disp = np.zeros(len(notes), dtype=np.float32)
    for i in range(len(notes)):
        if rng.random() < config.large_displacement_prob:
            mag = rng.uniform(*config.large_displacement_range)
            sign = rng.choice([-1.0, 1.0])
            large_disp[i] = sign * mag

    # --- 5. Swing perturbation ---
    swing = np.zeros(len(notes), dtype=np.float32)
    if rng.random() < config.swing_prob:
        beat_ms = 60_000.0 / tempo_bpm
        subdivision_ms = beat_ms / 2.0  # eighth-note level
        swing_offset = rng.uniform(0.0, config.swing_max_offset_ms)
        for i, note in enumerate(notes):
            # Determine which subdivision this note is closest to.
            subdiv_idx = round(note.onset_ms / subdivision_ms)
            if subdiv_idx % 2 == 1:  # off-beat subdivisions
                swing[i] = swing_offset

    # Combine all perturbations.
    total_offset = jitter + drift + smear + large_disp + swing
    offsets[:] = total_offset

    # Apply offsets to notes, clamping to loop boundaries.
    for i, note in enumerate(perturbed):
        delta = float(total_offset[i])
        duration = note.offset_ms - note.onset_ms
        note.onset_ms = max(0.0, min(note.onset_ms + delta, duration_ms - 1.0))
        note.offset_ms = note.onset_ms + duration

    return perturbed, offsets
