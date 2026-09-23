"""Verify an enrollment/verification WAV pair with realtime preprocessing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from src.audio import (
    AudioLoadError,
    AudioValidationError,
    InsufficientSpeechError,
    SpeechWindowMetadata,
)
from src.inference import (
    SpeakerVerifier,
    decision_from_threshold,
    score_embeddings,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("enrollment", type=Path, help="Enrollment WAV/audio file.")
    parser.add_argument("verification", type=Path, help="Verification WAV/audio file.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Manual/development cosine-similarity threshold. If omitted, no "
            "same/different decision is produced."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path (default: checkpoints/best_adp.pt).",
    )
    return parser.parse_args(argv)


def _print_metadata(label: str, metadata: SpeechWindowMetadata) -> None:
    print(f"{label}:")
    print(f"  original duration: {metadata.original_duration_seconds:.3f} s")
    print(f"  speech duration: {metadata.total_speech_seconds:.3f} s")
    print(
        "  selected time range: "
        f"{metadata.selected_start_seconds:.3f}–{metadata.selected_end_seconds:.3f} s"
    )
    print(
        "  speech in selected window: "
        f"{metadata.speech_in_selected_window_seconds:.3f} s"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    verifier = SpeakerVerifier() if args.checkpoint is None else SpeakerVerifier(args.checkpoint)
    try:
        enrollment = verifier.extract_embedding_realtime(args.enrollment)
    except InsufficientSpeechError as exc:
        _print_metadata("Enrollment", exc.metadata)
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except (FileNotFoundError, AudioLoadError, AudioValidationError) as exc:
        print(f"Error processing enrollment audio: {exc}", file=sys.stderr)
        return 2

    try:
        verification = verifier.extract_embedding_realtime(args.verification)
    except InsufficientSpeechError as exc:
        _print_metadata("Enrollment", enrollment.metadata)
        print()
        _print_metadata("Verification", exc.metadata)
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except (FileNotFoundError, AudioLoadError, AudioValidationError) as exc:
        print(f"Error processing verification audio: {exc}", file=sys.stderr)
        return 2

    similarity = score_embeddings(enrollment.embedding, verification.embedding)
    same_speaker = decision_from_threshold(similarity, args.threshold)
    _print_metadata("Enrollment", enrollment.metadata)
    print()
    _print_metadata("Verification", verification.metadata)
    print()
    print("Result:")
    print(f"  cosine similarity: {similarity:.6f}")
    if args.threshold is None:
        print("  operational threshold: not configured")
        print("  decision: unavailable until a validation-calibrated threshold is configured")
    else:
        print(f"  manual/development threshold: {args.threshold:.6f}")
        print(f"  {'SAME SPEAKER' if same_speaker else 'DIFFERENT SPEAKER'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
