"""Pre-slice MIDI files into bar-aligned segments and save to disk.

Avoids re-parsing 11K+ MIDI files on every training run.
Filters out slices where notes are poorly aligned to the beat grid.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add project root to path so we can import tighnet
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm

from tighnet.dataset import _midi_to_tracks, slice_notes_into_bars
from tighnet.representation import MidiNote


def is_track_grid_aligned(
    note_dicts: list[dict],
    max_median_offset_ms: float = 10.0,
) -> bool:
    """Check whether a track's onsets are reasonably aligned to a subdivision grid.

    Uses 48th-note resolution (1/12 of a beat) which captures straight 16ths,
    triplets, and dotted rhythms. Computes each onset's distance to the nearest
    grid position, then checks whether the median offset is within the threshold.

    Args:
        max_median_offset_ms: Maximum allowed median offset in milliseconds.
    """
    if len(note_dicts) < 3:
        return False

    tempo_bpm = note_dicts[0]["tempo_bpm"]
    # 48th note = 1/12 of a beat. Covers 16ths (1/4 beat) and triplets (1/3 beat).
    subdiv_ms = 60_000.0 / tempo_bpm / 12.0

    offsets = []
    for n in note_dicts:
        remainder = n["onset_ms"] % subdiv_ms
        offset = min(remainder, subdiv_ms - remainder)
        offsets.append(offset)

    median_offset = float(np.median(offsets))
    return median_offset <= max_median_offset_ms


def preprocess(
    midi_dir: str,
    output: str,
    bars_per_slice: int = 4,
    max_median_offset_ms: float = 10.0,
) -> None:
    midi_path = Path(midi_dir)
    midi_files = sorted(midi_path.glob("**/*.mid")) + sorted(midi_path.glob("**/*.midi"))

    if not midi_files:
        print(f"No MIDI files found in {midi_dir}")
        return

    examples = []
    skipped_files = 0
    skipped_tracks = 0
    total_tracks = 0
    skipped_slices = 0
    total_slices = 0

    for path in tqdm(midi_files, desc="Processing MIDI files"):
        try:
            tracks = _midi_to_tracks(path)
        except Exception:
            skipped_files += 1
            continue

        if not tracks:
            skipped_files += 1
            continue

        for note_dicts in tracks:
            if not note_dicts:
                continue

            total_tracks += 1
            if not is_track_grid_aligned(note_dicts, max_median_offset_ms):
                skipped_tracks += 1
                continue

            tempo_bpm = note_dicts[0]["tempo_bpm"]
            time_sig_num = note_dicts[0]["time_sig_numerator"]

            slices = slice_notes_into_bars(
                note_dicts, tempo_bpm, time_sig_num, bars_per_slice
            )
            for notes, duration in slices:
                total_slices += 1
                if len(notes) < 3:
                    skipped_slices += 1
                    continue
                examples.append({
                    "notes": [
                        {
                            "pitch": n.pitch,
                            "velocity": n.velocity,
                            "onset_ms": n.onset_ms,
                            "offset_ms": n.offset_ms,
                        }
                        for n in notes
                    ],
                    "duration_ms": duration,
                    "tempo_bpm": tempo_bpm,
                    "time_sig_numerator": time_sig_num,
                })

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(examples, f)

    print(
        f"Saved {len(examples)} examples to {out_path}\n"
        f"  Files: {skipped_files} skipped (parse errors)\n"
        f"  Tracks: {skipped_tracks}/{total_tracks} skipped (misaligned)\n"
        f"  Slices: {skipped_slices}/{total_slices} skipped (<3 notes)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-slice MIDI files for training")
    parser.add_argument("midi_dir", help="Directory containing MIDI files")
    parser.add_argument("-o", "--output", default="data/sliced.json")
    parser.add_argument("--bars-per-slice", type=int, default=4)
    parser.add_argument(
        "--max-offset-ms",
        type=float,
        default=10.0,
        help="Max median onset offset in ms (default: 10.0)",
    )
    args = parser.parse_args()
    preprocess(args.midi_dir, args.output, args.bars_per_slice, args.max_offset_ms)


if __name__ == "__main__":
    main()
