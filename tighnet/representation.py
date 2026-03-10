"""Onset representation: convert MIDI events to/from the frame-based tensor format.

The representation is pitch-agnostic and uses 5ms frames with 3 channels:
  - onset_intensity: normalised sum of velocities of note-onsets in each frame
  - onset_count: number of simultaneous note-onsets per frame
  - sustain_density: number of notes currently sustaining
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Resolution of the temporal grid in milliseconds.
FRAME_MS = 5.0

# Maximum MIDI velocity used for normalisation.
MAX_VELOCITY = 127.0

# Channel indices in the representation tensor.
CH_ONSET_INTENSITY = 0
CH_ONSET_COUNT = 1
CH_SUSTAIN_DENSITY = 2
NUM_CHANNELS = 3


@dataclass
class MidiNote:
    """A single MIDI note event."""

    pitch: int
    velocity: int
    onset_ms: float  # note-on time in milliseconds
    offset_ms: float  # note-off time in milliseconds


def loop_duration_ms(tempo_bpm: float, time_sig_numerator: int, num_bars: int) -> float:
    """Return the duration of the loop in milliseconds."""
    beat_ms = 60_000.0 / tempo_bpm
    return beat_ms * time_sig_numerator * num_bars


def num_frames(duration_ms: float) -> int:
    """Return the number of frames for a given duration."""
    return int(np.ceil(duration_ms / FRAME_MS))


def ms_to_frame(time_ms: float) -> int:
    """Convert a time in milliseconds to a frame index."""
    return int(time_ms / FRAME_MS)


def frame_to_ms(frame_idx: int) -> float:
    """Convert a frame index back to milliseconds."""
    return frame_idx * FRAME_MS


def notes_to_onset_tensor(
    notes: list[MidiNote],
    duration_ms: float,
    circular_pad: int = 0,
) -> np.ndarray:
    """Convert a list of MIDI notes into the frame-based onset representation.

    Args:
        notes: MIDI note events within the loop.
        duration_ms: Total loop duration in milliseconds.
        circular_pad: Number of frames to circularly pad on each side.

    Returns:
        Array of shape (num_frames + 2 * circular_pad, NUM_CHANNELS).
    """
    n_frames = num_frames(duration_ms)
    tensor = np.zeros((n_frames, NUM_CHANNELS), dtype=np.float32)

    # Build a list of (frame, event_type) for sustain tracking.
    # event_type: +1 for onset, -1 for offset.
    events: list[tuple[int, int]] = []

    for note in notes:
        onset_frame = min(ms_to_frame(note.onset_ms), n_frames - 1)
        onset_frame = max(onset_frame, 0)

        # Channel 0: onset intensity (normalised velocity).
        tensor[onset_frame, CH_ONSET_INTENSITY] += note.velocity / MAX_VELOCITY
        # Channel 1: onset count.
        tensor[onset_frame, CH_ONSET_COUNT] += 1.0

        # Record events for sustain computation.
        offset_frame = min(ms_to_frame(note.offset_ms), n_frames)
        offset_frame = max(offset_frame, onset_frame + 1)
        events.append((onset_frame, 1))
        events.append((offset_frame, -1))

    # Channel 2: sustain density via a running sum over sorted events.
    if events:
        events.sort()
        current_sustain = 0
        event_idx = 0
        for f in range(n_frames):
            while event_idx < len(events) and events[event_idx][0] <= f:
                current_sustain += events[event_idx][1]
                event_idx += 1
            tensor[f, CH_SUSTAIN_DENSITY] = max(current_sustain, 0)

    # Apply circular padding.
    if circular_pad > 0:
        pad_start = tensor[-circular_pad:]
        pad_end = tensor[:circular_pad]
        tensor = np.concatenate([pad_start, tensor, pad_end], axis=0)

    return tensor


def onset_frames_from_tensor(tensor: np.ndarray, circular_pad: int = 0) -> list[int]:
    """Return frame indices that contain at least one onset.

    Args:
        tensor: The onset representation tensor (possibly with circular padding).
        circular_pad: Number of padding frames on each side (to offset indices).

    Returns:
        Sorted list of frame indices (relative to the unpadded tensor).
    """
    # Work on the unpadded region.
    if circular_pad > 0:
        core = tensor[circular_pad:-circular_pad]
    else:
        core = tensor
    frames = np.where(core[:, CH_ONSET_COUNT] > 0)[0].tolist()
    return frames
