"""Tests for the onset representation module."""

import numpy as np

from tighnet.representation import (
    FRAME_MS,
    MidiNote,
    frame_to_ms,
    loop_duration_ms,
    ms_to_frame,
    notes_to_onset_tensor,
    num_frames,
    onset_frames_from_tensor,
)


def test_loop_duration_120bpm_4bars():
    # 120 BPM, 4/4, 4 bars = 8 seconds = 8000 ms
    assert loop_duration_ms(120.0, 4, 4) == 8000.0


def test_loop_duration_60bpm_2bars():
    # 60 BPM, 4/4, 2 bars = 8 seconds = 8000 ms
    assert loop_duration_ms(60.0, 4, 2) == 8000.0


def test_num_frames():
    assert num_frames(8000.0) == 1600  # 8000 / 5 = 1600


def test_frame_ms_roundtrip():
    for ms in [0.0, 100.0, 500.0, 7995.0]:
        frame = ms_to_frame(ms)
        recovered = frame_to_ms(frame)
        assert abs(recovered - ms) < FRAME_MS


def test_empty_notes():
    tensor = notes_to_onset_tensor([], 8000.0)
    assert tensor.shape == (1600, 3)
    assert np.all(tensor == 0)


def test_single_note_onset():
    note = MidiNote(pitch=60, velocity=100, onset_ms=500.0, offset_ms=1000.0)
    tensor = notes_to_onset_tensor([note], 8000.0)

    onset_frame = ms_to_frame(500.0)  # frame 100
    assert tensor[onset_frame, 0] > 0  # onset intensity
    assert tensor[onset_frame, 1] == 1.0  # onset count


def test_chord_onset():
    notes = [
        MidiNote(pitch=60, velocity=80, onset_ms=500.0, offset_ms=1000.0),
        MidiNote(pitch=64, velocity=90, onset_ms=500.0, offset_ms=1000.0),
        MidiNote(pitch=67, velocity=70, onset_ms=500.0, offset_ms=1000.0),
    ]
    tensor = notes_to_onset_tensor(notes, 8000.0)

    onset_frame = ms_to_frame(500.0)
    assert tensor[onset_frame, 1] == 3.0  # 3 notes in chord
    expected_intensity = (80 + 90 + 70) / 127.0
    assert abs(tensor[onset_frame, 0] - expected_intensity) < 0.01


def test_sustain_density():
    # A note from 0–4000ms should show sustain in frames throughout that range.
    note = MidiNote(pitch=60, velocity=100, onset_ms=0.0, offset_ms=4000.0)
    tensor = notes_to_onset_tensor([note], 8000.0)

    # Middle of sustain should have density > 0.
    mid_frame = ms_to_frame(2000.0)
    assert tensor[mid_frame, 2] > 0

    # After note-off should have density 0.
    after_frame = ms_to_frame(5000.0)
    assert tensor[after_frame, 2] == 0


def test_circular_padding():
    note = MidiNote(pitch=60, velocity=100, onset_ms=100.0, offset_ms=200.0)
    pad = 32
    tensor = notes_to_onset_tensor([note], 8000.0, circular_pad=pad)

    # Should be padded: original 1600 + 2*32 = 1664
    assert tensor.shape[0] == 1600 + 2 * pad


def test_onset_frames_extraction():
    notes = [
        MidiNote(pitch=60, velocity=100, onset_ms=500.0, offset_ms=600.0),
        MidiNote(pitch=64, velocity=80, onset_ms=1500.0, offset_ms=1600.0),
    ]
    tensor = notes_to_onset_tensor(notes, 8000.0)
    frames = onset_frames_from_tensor(tensor)

    assert ms_to_frame(500.0) in frames
    assert ms_to_frame(1500.0) in frames
    assert len(frames) == 2
