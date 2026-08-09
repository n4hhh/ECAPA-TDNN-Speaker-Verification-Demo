"""Print a concise checkpoint summary and strict compatibility result."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.model import build_model_from_checkpoint, load_checkpoint, validate_checkpoint

DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "checkpoints"
    / "ecapa_tdnn_finetuned_3.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        nargs="?",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint path (default: project checkpoint).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    num_classes = validate_checkpoint(checkpoint)
    state_dict = checkpoint["model_state_dict"]
    model = build_model_from_checkpoint(checkpoint, device="cpu")

    print(f"Checkpoint: {args.checkpoint.resolve()}")
    print(f"Size: {args.checkpoint.stat().st_size:,} bytes")
    print(f"Top-level keys: {', '.join(checkpoint.keys())}")
    print(f"Epoch: {checkpoint.get('epoch', 'not present')}")
    print(f"Validation accuracy: {checkpoint.get('val_acc', 'not present')}")
    print(f"State entries: {len(state_dict)}")
    print(f"Label mappings: {len(checkpoint['label2id'])} / {len(checkpoint['id2label'])}")
    print(f"Classification head: {tuple(state_dict['fc.weight'].shape)}")
    print(
        "SpeechBrain classifier: "
        f"{tuple(state_dict['spk_classifier.mods.classifier.weight'].shape)}"
    )
    print(f"Inferred classes: {num_classes}")
    print(f"Model mode: {'eval' if not model.training else 'train'}")
    print("Strict compatibility: PASS (all keys matched)")


if __name__ == "__main__":
    main()
