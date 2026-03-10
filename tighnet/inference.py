"""Inference pipeline: apply the trained model's corrections back to MIDI notes.

Implements the full pipeline:
  1. Extract onset representation from recorded MIDI.
  2. Run the model to get per-frame timing offsets.
  3. Cluster MIDI notes into onset groups.
  4. Apply predicted offsets to each cluster.
  5. Enforce loop boundary alignment.
  6. Apply optional strength scaling.
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from .model import DilatedCNN, UNet1D
from .representation import (
    FRAME_MS,
    MidiNote,
    frame_to_ms,
    ms_to_frame,
    notes_to_onset_tensor,
    num_frames,
)


def cluster_notes(notes: list[MidiNote], window_ms: float = 30.0) -> list[list[int]]:
    """Group notes into onset clusters based on temporal proximity.

    Notes within `window_ms` of each other are grouped as a single rhythmic event
    (e.g., a chord or a roll).

    Returns:
        List of clusters, where each cluster is a list of indices into `notes`.
    """
    if not notes:
        return []

    sorted_indices = sorted(range(len(notes)), key=lambda i: notes[i].onset_ms)
    clusters: list[list[int]] = [[sorted_indices[0]]]

    for idx in sorted_indices[1:]:
        # Compare to the earliest note in the current cluster.
        cluster_start = notes[clusters[-1][0]].onset_ms
        if notes[idx].onset_ms - cluster_start <= window_ms:
            clusters[-1].append(idx)
        else:
            clusters.append([idx])

    return clusters


def quantize(
    notes: list[MidiNote],
    duration_ms: float,
    model: DilatedCNN | UNet1D,
    strength: float = 1.0,
    cluster_window_ms: float = 30.0,
    boundary_snap_ms: float = 15.0,
    circular_pad: int = 64,
    device: str = "cpu",
) -> list[MidiNote]:
    """Apply learned quantization to a list of MIDI notes.

    Args:
        notes: The recorded MIDI notes.
        duration_ms: Loop duration in milliseconds.
        model: Trained onset correction model.
        strength: Correction strength from 0.0 (no change) to 1.0 (full correction).
        cluster_window_ms: Window for grouping notes into onset clusters.
        boundary_snap_ms: Threshold for snapping notes to loop boundaries.
        circular_pad: Circular padding frames for the onset representation.
        device: Torch device for inference.

    Returns:
        New list of MidiNote with corrected onset/offset times.
    """
    if not notes:
        return []

    model.eval()

    # 1. Extract onset representation.
    tensor = notes_to_onset_tensor(notes, duration_ms, circular_pad)

    # Pad to a reasonable length for the model.
    n_frames = num_frames(duration_ms) + 2 * circular_pad
    if tensor.shape[0] < n_frames:
        pad_width = [(0, n_frames - tensor.shape[0]), (0, 0)]
        tensor = np.pad(tensor, pad_width, mode="constant")
    else:
        tensor = tensor[:n_frames]

    # 2. Run model.
    input_tensor = torch.from_numpy(tensor).unsqueeze(0).to(device)  # (1, frames, channels)
    with torch.no_grad():
        predicted_offsets = model(input_tensor)  # (1, frames, 1)
    predicted_offsets = predicted_offsets.squeeze(0).squeeze(-1).cpu().numpy()  # (frames,)

    # 3. Cluster notes.
    clusters = cluster_notes(notes, cluster_window_ms)

    # 4. Apply corrections.
    corrected = copy.deepcopy(notes)

    for cluster_indices in clusters:
        # Find the representative frame for this cluster (weighted average by velocity).
        total_vel = sum(notes[i].velocity for i in cluster_indices)
        if total_vel == 0:
            continue

        weighted_onset = sum(
            notes[i].onset_ms * notes[i].velocity for i in cluster_indices
        ) / total_vel

        frame = ms_to_frame(weighted_onset) + circular_pad
        frame = max(0, min(frame, len(predicted_offsets) - 1))

        # Read the predicted offset for this frame.
        offset_ms = float(predicted_offsets[frame])

        # Apply strength scaling.
        offset_ms *= strength

        # Shift all notes in the cluster by the same offset.
        for idx in cluster_indices:
            note = corrected[idx]
            note_duration = note.offset_ms - note.onset_ms
            note.onset_ms = max(0.0, note.onset_ms + offset_ms)
            note.offset_ms = note.onset_ms + note_duration

    # 5. Boundary enforcement.
    for note in corrected:
        # Snap to loop start.
        if note.onset_ms <= boundary_snap_ms:
            delta = -note.onset_ms
            note.onset_ms = 0.0
            note.offset_ms += delta

        # Snap to loop end.
        dist_to_end = abs(note.onset_ms - duration_ms)
        if dist_to_end <= boundary_snap_ms:
            delta = duration_ms - note.onset_ms
            note.onset_ms = duration_ms
            note.offset_ms += delta

    return corrected
