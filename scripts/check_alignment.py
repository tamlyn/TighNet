"""Export sample aligned and misaligned MIDI files for manual audition."""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm

from tighnet.dataset import _midi_to_tracks
from scripts.preprocess import is_track_grid_aligned


def main() -> None:
    midi_dir = Path("data/adl-piano-midi")
    midi_files = sorted(midi_dir.glob("**/*.mid"))

    aligned = []
    misaligned = []

    for path in tqdm(midi_files, desc="Scanning"):
        try:
            tracks = _midi_to_tracks(path)
        except Exception:
            continue

        for track in tracks:
            if len(track) < 3:
                continue
            if is_track_grid_aligned(track):
                aligned.append(path)
            else:
                misaligned.append(path)
            break  # just check first note-bearing track

    random.seed(42)

    print("\n=== ALIGNED (10 samples) ===")
    for p in random.sample(aligned, min(10, len(aligned))):
        print(f"  {p}")

    print(f"\n=== MISALIGNED (10 samples) ===")
    for p in random.sample(misaligned, min(10, len(misaligned))):
        print(f"  {p}")

    print(f"\nTotals: {len(aligned)} aligned, {len(misaligned)} misaligned "
          f"out of {len(aligned) + len(misaligned)} files with notes")


if __name__ == "__main__":
    main()
