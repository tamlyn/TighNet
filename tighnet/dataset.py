"""Dataset and data loading for training the onset correction model.

Handles:
  - Loading MIDI files and slicing into bar-aligned segments.
  - Loading pre-sliced data from a JSON cache.
  - Converting to the onset representation.
  - Applying synthetic perturbations to generate (noisy, clean, offset) triples.
"""

from __future__ import annotations

import json
from pathlib import Path

import mido
import numpy as np
import torch
from torch.utils.data import Dataset

from .perturbation import PerturbConfig, perturb
from .representation import (
    FRAME_MS,
    MidiNote,
    loop_duration_ms,
    ms_to_frame,
    notes_to_onset_tensor,
    num_frames,
)


def _midi_to_tracks(midi_path: Path, tempo_bpm: float | None = None) -> list[list[dict]]:
    """Parse a MIDI file into per-track note lists with absolute timing in ms.

    Returns a list of tracks, where each track is a list of dicts with keys:
    pitch, velocity, onset_ms, offset_ms, tempo_bpm, time_sig_numerator.
    Tracks with no notes are omitted.
    """
    mid = mido.MidiFile(str(midi_path))

    # Extract tempo and time signature from the file.
    file_tempo = 500_000  # default 120 BPM
    time_sig_num = 4
    for track in mid.tracks:
        for msg in track:
            if msg.type == "set_tempo":
                file_tempo = msg.tempo
            elif msg.type == "time_signature":
                time_sig_num = msg.numerator

    if tempo_bpm is None:
        tempo_bpm = mido.tempo2bpm(file_tempo)

    ticks_per_beat = mid.ticks_per_beat
    all_tracks: list[list[dict]] = []

    for track in mid.tracks:
        notes_on: dict[int, tuple[float, int]] = {}
        result: list[dict] = []
        abs_time_ticks = 0
        current_tempo = file_tempo

        for msg in track:
            abs_time_ticks += msg.time
            abs_time_ms = (
                mido.tick2second(abs_time_ticks, ticks_per_beat, current_tempo) * 1000.0
            )

            if msg.type == "set_tempo":
                current_tempo = msg.tempo
            elif msg.type == "note_on" and msg.velocity > 0:
                notes_on[msg.note] = (abs_time_ms, msg.velocity)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                if msg.note in notes_on:
                    onset_ms, velocity = notes_on.pop(msg.note)
                    result.append(
                        {
                            "pitch": msg.note,
                            "velocity": velocity,
                            "onset_ms": onset_ms,
                            "offset_ms": abs_time_ms,
                            "tempo_bpm": tempo_bpm,
                            "time_sig_numerator": time_sig_num,
                        }
                    )

        if result:
            all_tracks.append(result)

    return all_tracks


def slice_notes_into_bars(
    note_dicts: list[dict],
    tempo_bpm: float,
    time_sig_numerator: int,
    bars_per_slice: int,
) -> list[tuple[list[MidiNote], float]]:
    """Slice a flat list of note dicts into bar-aligned segments.

    Returns a list of (notes, duration_ms) tuples.
    """
    bar_duration = loop_duration_ms(tempo_bpm, time_sig_numerator, 1)
    slice_duration = bar_duration * bars_per_slice

    if not note_dicts:
        return []

    # Find total duration.
    max_offset = max(n["offset_ms"] for n in note_dicts)
    total_slices = int(max_offset // slice_duration)

    slices = []
    for s in range(total_slices):
        start_ms = s * slice_duration
        end_ms = start_ms + slice_duration
        segment_notes = []
        for n in note_dicts:
            if n["onset_ms"] >= start_ms and n["onset_ms"] < end_ms:
                segment_notes.append(
                    MidiNote(
                        pitch=n["pitch"],
                        velocity=n["velocity"],
                        onset_ms=n["onset_ms"] - start_ms,
                        offset_ms=min(n["offset_ms"], end_ms) - start_ms,
                    )
                )
        if segment_notes:  # skip empty slices
            slices.append((segment_notes, slice_duration))

    return slices


def _load_from_cache(cache_path: Path) -> list[tuple[list[MidiNote], float, float, int]]:
    """Load pre-sliced examples from a JSON cache file."""
    with open(cache_path) as f:
        raw = json.load(f)

    examples = []
    for entry in raw:
        notes = [
            MidiNote(
                pitch=n["pitch"],
                velocity=n["velocity"],
                onset_ms=n["onset_ms"],
                offset_ms=n["offset_ms"],
            )
            for n in entry["notes"]
        ]
        examples.append((
            notes,
            entry["duration_ms"],
            entry["tempo_bpm"],
            entry["time_sig_numerator"],
        ))
    return examples


class MidiQuantizationDataset(Dataset):
    """PyTorch dataset that generates training examples from sliced MIDI data.

    Each example is a tuple of:
      - noisy_tensor: (max_frames, 3) onset representation of the perturbed performance
      - clean_tensor: (max_frames, 3) onset representation of the clean performance
      - offset_target: (max_frames, 1) per-frame ground-truth correction offset in ms

    Can load from either a pre-sliced JSON cache (fast) or a raw MIDI directory (slow).
    Use scripts/preprocess.py to generate the cache.
    """

    def __init__(
        self,
        midi_dir: str | Path | None = None,
        cache_path: str | Path | None = None,
        max_frames: int = 1600,
        bars_per_slice: int = 4,
        circular_pad: int = 64,
        perturb_config: PerturbConfig | None = None,
        seed: int = 42,
    ):
        self.max_frames = max_frames
        self.bars_per_slice = bars_per_slice
        self.circular_pad = circular_pad
        self.perturb_config = perturb_config or PerturbConfig()
        self.rng = np.random.default_rng(seed)

        if cache_path is not None:
            self.examples = _load_from_cache(Path(cache_path))
        elif midi_dir is not None:
            self.examples = self._load_from_midi(Path(midi_dir), bars_per_slice)
        else:
            raise ValueError("Either midi_dir or cache_path must be provided")

    def _load_from_midi(
        self, midi_dir: Path, bars_per_slice: int
    ) -> list[tuple[list[MidiNote], float, float, int]]:
        examples: list[tuple[list[MidiNote], float, float, int]] = []
        midi_files = sorted(midi_dir.glob("**/*.mid")) + sorted(midi_dir.glob("**/*.midi"))

        for path in midi_files:
            try:
                tracks = _midi_to_tracks(path)
            except Exception:
                continue

            for note_dicts in tracks:
                if not note_dicts:
                    continue

                tempo_bpm = note_dicts[0]["tempo_bpm"]
                time_sig_num = note_dicts[0]["time_sig_numerator"]

                slices = slice_notes_into_bars(
                    note_dicts, tempo_bpm, time_sig_num, bars_per_slice
                )
                for notes, duration in slices:
                    examples.append((notes, duration, tempo_bpm, time_sig_num))

        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        notes, duration_ms, tempo_bpm, time_sig_num = self.examples[idx]

        # Clean representation.
        clean_tensor = notes_to_onset_tensor(notes, duration_ms, self.circular_pad)

        # Apply perturbation.
        perturbed_notes, offsets = perturb(
            notes,
            duration_ms,
            tempo_bpm,
            time_sig_num,
            self.perturb_config,
            self.rng,
        )

        noisy_tensor = notes_to_onset_tensor(perturbed_notes, duration_ms, self.circular_pad)

        # Build per-frame offset target. The offset is the negative of the perturbation
        # (the model should predict how much to move the note *back*).
        n_frames_total = clean_tensor.shape[0]
        offset_target = np.zeros((n_frames_total, 1), dtype=np.float32)

        for i, note in enumerate(perturbed_notes):
            frame = ms_to_frame(note.onset_ms) + self.circular_pad
            if 0 <= frame < n_frames_total:
                # Negative offset: model predicts how to undo the perturbation.
                offset_target[frame, 0] = -offsets[i]

        # Pad or truncate to max_frames.
        padded_max = self.max_frames + 2 * self.circular_pad

        def _pad_or_truncate(arr: np.ndarray, target_len: int) -> np.ndarray:
            if arr.shape[0] >= target_len:
                return arr[:target_len]
            pad_width = [(0, target_len - arr.shape[0])] + [(0, 0)] * (arr.ndim - 1)
            return np.pad(arr, pad_width, mode="constant")

        clean_tensor = _pad_or_truncate(clean_tensor, padded_max)
        noisy_tensor = _pad_or_truncate(noisy_tensor, padded_max)
        offset_target = _pad_or_truncate(offset_target, padded_max)

        return (
            torch.from_numpy(noisy_tensor),
            torch.from_numpy(clean_tensor),
            torch.from_numpy(offset_target),
        )
