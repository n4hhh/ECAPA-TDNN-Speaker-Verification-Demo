"""Print frozen-handoff checkpoint metadata and strict inference compatibility."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.model import build_model_from_checkpoint, load_checkpoint, validate_checkpoint

DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / "checkpoints" / "best_adp.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        nargs="?",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint path (default: checkpoints/best_adp.pt).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    metadata = validate_checkpoint(checkpoint)
    model = build_model_from_checkpoint(checkpoint, device="cpu")

    # Summarize schema, architecture, and training metadata after strict loading;
    # successful construction is stronger evidence than inspecting keys alone.
    print(f"Checkpoint: {args.checkpoint.resolve()}")
    print(f"File size: {args.checkpoint.stat().st_size:,} bytes")
    print(f"Schema: {metadata.schema}")
    print(f"Reason: {metadata.reason}")
    print(f"Model source: {metadata.model_source}")
    print(f"Embedding dimension: {metadata.embedding_dim}")
    print(
        "Best validation EER: "
        + ("not present" if metadata.best_eer is None else f"{metadata.best_eer:.8f}")
    )
    print(
        "embedding_model_state_dict entries: "
        f"{metadata.embedding_state_entries}"
    )
    print(
        "mean_var_norm_state_dict entries: "
        f"{metadata.mean_var_norm_state_entries}"
    )
    if metadata.aam_metadata:
        margin = metadata.aam_metadata.get("margin", "not present")
        scale = metadata.aam_metadata.get("scale", "not present")
        classes = metadata.aam_metadata.get("num_classes", "not present")
        print(f"AAM metadata: margin={margin} scale={scale} classes={classes}")
    else:
        print("AAM metadata: not present")
    print(f"Model mode: {'eval' if not model.training else 'train'}")
    print("Strict inference compatibility: PASS")


if __name__ == "__main__":
    main()
