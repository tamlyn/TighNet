"""Show alignment stats for specific files to debug the filter."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tighnet.dataset import _midi_to_tracks


def debug_file(path: Path) -> None:
    print(f"\n{'=' * 60}")
    print(f"{path.name}")
    print(f"{'=' * 60}")

    tracks = _midi_to_tracks(path)
    for i, track in enumerate(tracks):
        if len(track) < 3:
            continue

        tempo_bpm = track[0]["tempo_bpm"]
        eighth_ms = 60_000.0 / tempo_bpm / 2.0
        beat_ms = 60_000.0 / tempo_bpm

        offsets_eighth = []
        offsets_beat = []
        for n in track:
            rem = n["onset_ms"] % eighth_ms
            offsets_eighth.append(min(rem, eighth_ms - rem))
            rem_beat = n["onset_ms"] % beat_ms
            offsets_beat.append(min(rem_beat, beat_ms - rem_beat))

        median_eighth = float(np.median(offsets_eighth))
        median_beat = float(np.median(offsets_beat))

        print(f"  Track {i}: {len(track)} notes, {tempo_bpm:.0f} BPM")
        print(f"    8th note = {eighth_ms:.1f}ms, beat = {beat_ms:.1f}ms")
        print(f"    Median offset from 8th grid: {median_eighth:.1f}ms "
              f"({median_eighth / eighth_ms:.2%} of 8th)")
        print(f"    Median offset from beat grid: {median_beat:.1f}ms "
              f"({median_beat / beat_ms:.2%} of beat)")


def main() -> None:
    check_dir = Path("data/check")
    for subdir in ["misaligned", "aligned"]:
        d = check_dir / subdir
        if d.exists():
            print(f"\n{'#' * 60}")
            print(f"# {subdir.upper()}")
            print(f"{'#' * 60}")
            for f in sorted(d.glob("*.mid")):
                debug_file(f)


if __name__ == "__main__":
    main()
