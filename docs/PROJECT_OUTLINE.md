# Intelligent MIDI Quantization for Piano Looper

## Problem Statement

A piano looper app records short MIDI sequences (1–4 bars) and plays them back in a loop. Raw recordings from non-professional players contain timing imperfections — sloppy rhythm, inconsistent swing, rushed or dragged notes — that become painfully obvious on repetition. The goal is to automatically tighten the timing so the loop sounds like a competent pianist played it, without forcing notes onto a rigid metronome grid.

This is distinct from traditional DAW quantization, which snaps notes to the nearest grid point. We want to preserve the musical character of the performance — swing feel, laid-back phrasing, expressive rubato — while correcting unintentional sloppiness. The result should sound like the same performance played by a better version of the player.

## Requirements

### Core

- **Style-aware correction**: Detect whether the performance is straight, swung, triplet-based, or mixed, and correct timing within that context rather than imposing a grid.
- **Chord-aware**: Rolled/arpeggiated chords should be recognised as single rhythmic events. Internal chord spacing should be preserved; the group should move together.
- **Ghost note / ornament handling**: Low-velocity notes and grace notes that are intentionally off-grid should be left alone or moved relative to their parent note, not treated as independent rhythmic events.
- **Loop boundary alignment**: Notes at the start and end of the loop must align cleanly with the loop boundaries so the loop cycles seamlessly.
- **Low latency**: Quantization happens after recording, before or during the first playback of the loop. Processing time should be imperceptible to the user (target: under 50ms for a 4-bar loop).
- **On-device**: Must run entirely on iOS via CoreML. No server round-trips.

### Nice to Have

- **Strength control**: A user-facing parameter (0–100%) controlling how aggressively timing is corrected. At 0%, the performance is unchanged. At 100%, maximum correction is applied.
- **Per-loop learning**: If a user records multiple loops, later loops could be influenced by the feel established in earlier ones (e.g., if loop 1 is swung, loop 2 should be corrected to match that swing feel).
- **Mixed subdivision handling**: A run of triplets within an otherwise straight passage should be quantized as triplets locally, not forced to straight timing.

## Constraints

- **Platform**: iOS (Swift, MIDIKit for MIDI I/O, CoreML for inference).
- **Input**: A sequence of MIDI note-on/note-off events with timestamps and velocities, plus known tempo, time signature, and loop length.
- **Output**: The same sequence with corrected note-on timestamps. Note-off times, velocities, and pitches are unchanged (except that note-off times shift by the same delta as their corresponding note-on, preserving duration).

---

## Approach: Learned Onset Correction

### Overview

Treat quantization as a **denoising task**. Train a small neural network to take a "sloppy" onset pattern and predict per-onset timing corrections that recover a clean performance. The model learns what good timing looks like from a corpus of competent piano performances, rather than relying on hand-coded rules about grids and subdivisions.

### Input Representation

Rather than feeding raw MIDI events, compress the performance into a lightweight temporal representation:

- **Time axis**: Divide the loop duration into frames at **5ms resolution**. A 4-bar loop at 120 BPM = 8 seconds = 1,600 frames. At slower tempos or longer loops, the frame count grows but remains manageable (a 4-bar loop at 60 BPM = 3,200 frames).
- **Channels** (per frame):
  1. **Onset intensity**: Sum of velocities of all note-onsets in this frame, normalised. Zero if no onset.
  2. **Onset count**: Number of simultaneous note-onsets in this frame. Helps distinguish single notes from chords.
  3. **Sustain density**: Number of notes currently sustaining (already sounding). Provides harmonic context.
- **Circular padding**: Pad the end of the window with the beginning (and vice versa) so the model can reason about the loop boundary.

This representation is **pitch-agnostic** — the model sees rhythmic patterns, not specific notes. This dramatically reduces model size and makes the learned features transferable across keys and registers.

### Output Representation

For each frame that contains an onset, the model predicts a **signed timing offset in milliseconds** — how far to shift the onset to correct it. Frames with no onset are ignored (or predict zero).

This is a regression task. Loss function is mean squared error on the predicted offsets versus the ground-truth offsets.

### Model Architecture

A **1D convolutional network** operating along the time axis:

```
Input: (batch, frames, channels)  e.g. (32, 1600, 3)
  → Conv1D stack (4–6 layers, kernel size 15–31, dilated)
  → Per-frame output: predicted offset (1 value per frame)
Output: (batch, frames, 1)
```

Dilated convolutions give the model a wide receptive field (several beats of context) without excessive depth. The full model should be well under 1MB — trivial for CoreML on any modern iPhone.

A **1D U-Net** variant is also worth exploring: downsampling captures bar-level rhythmic structure, upsampling restores frame-level precision. Adds modest complexity but may handle mixed subdivisions better.

### Training Data

#### Source Datasets

Ranked by relevance:

1. **ADL Piano MIDI** (~11,000 piano pieces across pop, rock, jazz, blues, country). Best genre match for typical looper users. Extracted from the Lakh MIDI dataset.
2. **PiJAMA** (~120 jazz pianists, ~244 albums). Transcribed from real recordings, so timing captures authentic human performance including swing. Good for learning jazz/swing feel.
3. **GigaMIDI** (1.6M+ expressively performed tracks). Use their expressive performance heuristics to filter to only genuinely human-played piano tracks. Largest scale option.
4. **ATEPP** (~1,000 hours, 49 classical pianists). Classical bias, but performances are aligned to scores — useful for supervised training if score-aligned targets are needed.

#### Synthetic Perturbation Pipeline

Since we're training a denoising model, we need (clean, noisy) pairs:

1. **Slice source MIDI at bar boundaries** using tempo/time-signature metadata. Create 1-bar, 2-bar, and 4-bar examples.
2. **Extract the compressed onset representation** from each slice. This is the clean target.
3. **Apply realistic perturbations** to create the noisy input:
   - **Gaussian timing jitter**: Add N(0, σ) noise to each onset, with σ ranging from 10–60ms to simulate varying degrees of sloppiness.
   - **Systematic drift**: Gradually shift onsets earlier or later across the bar to simulate rushing/dragging.
   - **Chord smearing**: For simultaneous onsets (chords), spread them across 10–80ms to simulate inconsistent chord attacks.
   - **Random large displacements**: With low probability (~5%), shift an onset by 50–150ms to simulate a clearly misplaced note.
   - **Swing perturbation**: For straight performances, apply partial swing (offset every other subdivision) to create examples where the model must decide whether swing is intentional.
4. **Compute the ground-truth offset** for each onset: the difference between the perturbed position and the original position.

The perturbation parameters should be randomised per example so the model sees a wide range of sloppiness levels during training.

### Inference Pipeline (in the app)

```
Recording complete
  → Extract onset representation from recorded MIDI (< 1ms)
  → Run CoreML model on onset tensor (< 10ms estimated)
  → Get per-frame timing offsets
  → Group MIDI notes into onset clusters (notes within 30ms window)
  → Apply the predicted offset for each cluster to all notes in that cluster
  → Hard-snap any note within 15ms of loop start/end to the boundary
  → Play corrected loop
```

Total processing time should be well under 50ms for a 4-bar loop.

### Applying Corrections Back to MIDI

The model operates on the compressed, pitch-agnostic representation but corrections must be applied to actual MIDI notes:

1. **Onset clustering**: Group all note-on events within a configurable window (e.g. 30ms). Each cluster represents a single "rhythmic event" — a chord, a single note, or a roll.
2. **Cluster-to-frame mapping**: Map each cluster to the frame(s) it occupies in the onset representation.
3. **Read the model's predicted offset** for that frame region.
4. **Shift all notes in the cluster** by the predicted offset. Internal spacing within the cluster (e.g. the roll of an arpeggiated chord) is preserved.
5. **Boundary enforcement**: Any cluster whose corrected position falls within a threshold of the loop start or end is hard-snapped to the boundary.
6. **Strength scaling**: If the user has set a strength parameter < 100%, scale all offsets proportionally: `applied_offset = predicted_offset × (strength / 100)`.
