#!/usr/bin/env python3
"""
Generate MIDI music using trained PerformanceRNN model.

Usage:
    python generate.py --checkpoint runs/20260105_120000/best.ckpt --output generated.mid
    python generate.py --checkpoint path/to/checkpoint.ckpt --num_steps 1000 --temperature 1.0
"""

import argparse
from pathlib import Path

import mido
import torch
from performance_net import PerformanceNet
from performance_net.event_encoder import EventEncoder, EventType


def generate_sequence(
    model: PerformanceNet,
    encoder: EventEncoder,
    num_steps: int = 1000,
    temperature: float = 1.0,
    primer_events: list = None,
) -> list:
    """
    Generate a sequence of events using the trained model.

    Args:
        model: Trained PerformanceNet model
        encoder: EventEncoder instance
        num_steps: Number of events to generate
        temperature: Sampling temperature (higher = more random)
        primer_events: Optional list of primer event IDs

    Returns:
        List of generated event IDs
    """
    model.eval()
    device = next(model.parameters()).device

    # Start with primer or default start
    if primer_events is None:
        # Start with a velocity event and first note
        primer_events = [
            encoder.VELOCITY_OFFSET + 16,  # Medium velocity
            encoder.NOTE_ON_OFFSET + 60,  # Middle C
        ]

    generated = list(primer_events)

    with torch.no_grad():
        for _ in range(num_steps):
            # Prepare input (last 512 events or all if shorter)
            context_len = min(512, len(generated))
            input_seq = (
                torch.tensor(generated[-context_len:], dtype=torch.long)
                .unsqueeze(0)
                .to(device)
            )

            # Get predictions
            logits = model(input_seq)  # [1, seq_len, vocab_size]
            next_logits = logits[0, -1, :] / temperature  # Last timestep

            # Sample from distribution
            probs = torch.softmax(next_logits, dim=0)
            next_event = torch.multinomial(probs, 1).item()

            generated.append(next_event)

    return generated


def events_to_midi(event_ids: list, encoder: EventEncoder, output_path: str):
    """
    Convert event IDs to MIDI file.

    Args:
        event_ids: List of event IDs
        encoder: EventEncoder instance
        output_path: Path to save MIDI file
    """
    # Decode events
    events = encoder.decode_sequence(event_ids)

    # Create MIDI file
    midi = mido.MidiFile()
    track = mido.MidiTrack()
    midi.tracks.append(track)

    # Track active notes and current state
    active_notes = {}
    current_velocity = 64
    accumulated_time = 0

    for event in events:
        if event.event_type == EventType.TIME_SHIFT:
            time_seconds = encoder.dequantize_time(event.event_value)
            accumulated_time += time_seconds

        elif event.event_type == EventType.VELOCITY:
            current_velocity = encoder.dequantize_velocity(event.event_value)

        elif event.event_type == EventType.NOTE_ON:
            pitch = event.event_value
            # Convert accumulated time to MIDI ticks
            delta_time = int(accumulated_time * midi.ticks_per_beat * 2)  # 120 BPM
            track.append(
                mido.Message(
                    "note_on", note=pitch, velocity=current_velocity, time=delta_time
                )
            )
            active_notes[pitch] = accumulated_time
            accumulated_time = 0

        elif event.event_type == EventType.NOTE_OFF:
            pitch = event.event_value
            if pitch in active_notes:
                delta_time = int(accumulated_time * midi.ticks_per_beat * 2)
                track.append(
                    mido.Message("note_off", note=pitch, velocity=0, time=delta_time)
                )
                del active_notes[pitch]
                accumulated_time = 0

    # Close any remaining notes
    for pitch in list(active_notes.keys()):
        track.append(mido.Message("note_off", note=pitch, velocity=0, time=0))

    # Save MIDI file
    midi.save(output_path)
    print(f"✓ MIDI file saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate music with PerformanceRNN")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.ckpt file)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="generated_music.mid",
        help="Output MIDI file path",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=1000,
        help="Number of events to generate (default: 1000)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (higher = more random, default: 1.0)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("PerformanceRNN Music Generation")
    print("=" * 60)
    print()

    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"❌ Error: Checkpoint not found: {args.checkpoint}")
        print("\nAvailable checkpoints:")
        runs_dir = Path("runs")
        if runs_dir.exists():
            for ckpt in runs_dir.rglob("*.ckpt"):
                print(f"  - {ckpt}")
        return

    # Load model - handle directory or checkpoint file
    checkpoint_path = Path(args.checkpoint)

    if checkpoint_path.is_dir():
        # Try to find best.ckpt or latest checkpoint
        best_ckpt = checkpoint_path / "best.ckpt"
        if best_ckpt.exists():
            checkpoint_path = best_ckpt
            print(f"Found best checkpoint: {checkpoint_path}")
        else:
            # Find any .ckpt file
            ckpts = list(checkpoint_path.glob("*.ckpt"))
            if not ckpts:
                print(f"Error: No .ckpt files found in {checkpoint_path}")
                return
            checkpoint_path = ckpts[0]
            print(f"Using checkpoint: {checkpoint_path}")

    print(f"Loading model from: {checkpoint_path}")
    model = PerformanceNet.load_from_checkpoint(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"✓ Model loaded on {device}")
    print()

    # Generate
    print(f"Generating {args.num_steps} events (temperature={args.temperature})...")
    encoder = EventEncoder()

    generated_ids = generate_sequence(
        model=model,
        encoder=encoder,
        num_steps=args.num_steps,
        temperature=args.temperature,
    )

    print(f"✓ Generated {len(generated_ids)} events")
    print()

    # Convert to MIDI
    print(f"Converting to MIDI: {args.output}")
    events_to_midi(generated_ids, encoder, args.output)

    print()
    print("=" * 60)
    print("✅ Generation complete!")
    print(f"Output: {args.output}")
    print()
    print("Play the MIDI file with:")
    print(f"  - timidity {args.output}")
    print(f"  - Or open in a DAW (FL Studio, Ableton, etc.)")
    print("=" * 60)


if __name__ == "__main__":
    main()
