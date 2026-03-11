"""Export a random sample of sliced loops as .mid files for audition."""

import argparse
import json
import random
from pathlib import Path

import mido


def notes_to_midi(notes: list[dict], duration_ms: float, tempo_bpm: float) -> mido.MidiFile:
    """Convert a list of note dicts to a MIDI file."""
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    tempo = mido.bpm2tempo(tempo_bpm)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

    # Build a list of (time_ms, type, pitch, velocity) events.
    events = []
    for n in notes:
        events.append((n["onset_ms"], "note_on", n["pitch"], n["velocity"]))
        events.append((n["offset_ms"], "note_off", n["pitch"], 0))
    events.sort(key=lambda e: e[0])

    prev_tick = 0
    for time_ms, msg_type, pitch, velocity in events:
        abs_tick = int(mido.second2tick(time_ms / 1000.0, mid.ticks_per_beat, tempo))
        delta = max(0, abs_tick - prev_tick)
        track.append(mido.Message(msg_type, note=pitch, velocity=velocity, time=delta))
        prev_tick = abs_tick

    return mid


def main() -> None:
    parser = argparse.ArgumentParser(description="Export sample sliced loops as MIDI files")
    parser.add_argument("cache", help="Path to sliced.json")
    parser.add_argument("-o", "--output-dir", default="data/samples")
    parser.add_argument("-n", "--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.cache) as f:
        examples = json.load(f)

    random.seed(args.seed)
    samples = random.sample(examples, min(args.count, len(examples)))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, ex in enumerate(samples):
        mid = notes_to_midi(ex["notes"], ex["duration_ms"], ex["tempo_bpm"])
        path = out_dir / f"sample_{i:03d}_{ex['tempo_bpm']:.0f}bpm.mid"
        mid.save(str(path))
        print(f"Saved {path} ({len(ex['notes'])} notes, {ex['duration_ms']:.0f}ms)")

    print(f"\nExported {len(samples)} samples to {out_dir}")


if __name__ == "__main__":
    main()
