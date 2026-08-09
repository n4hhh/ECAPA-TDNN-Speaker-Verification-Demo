"""Exact offline reconstruction of the notebook's SpeechBrain ECAPA wrapper."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.lobes.features import Fbank
from speechbrain.lobes.models.ECAPA_TDNN import Classifier, ECAPA_TDNN
from speechbrain.processing.features import InputNormalization

from .audio import SEGMENT_SAMPLES

EMBEDDING_DIM = 192
N_MELS = 80
PRETRAINED_CLASSIFIER_CLASSES = 7_205

ECAPA_CHANNELS = [1024, 1024, 1024, 1024, 3072]
ECAPA_KERNEL_SIZES = [5, 3, 3, 3, 1]
ECAPA_DILATIONS = [1, 2, 3, 4, 1]
ECAPA_ATTENTION_CHANNELS = 128

PathLike = Union[str, Path]
DeviceLike = Union[str, torch.device]
Checkpoint = dict[str, Any]


class CheckpointCompatibilityError(RuntimeError):
    """Raised when checkpoint metadata or tensors do not match the model contract."""


class ECAPAFinetuneModel(nn.Module):
    """Inference reconstruction of the wrapper saved by the training notebook."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")

        embedding_normalizer = InputNormalization(
            norm_type="global",
            std_norm=False,
        )
        modules = {
            "compute_features": Fbank(n_mels=N_MELS),
            "mean_var_norm": InputNormalization(
                norm_type="sentence",
                std_norm=False,
            ),
            "embedding_model": ECAPA_TDNN(
                input_size=N_MELS,
                channels=ECAPA_CHANNELS,
                kernel_sizes=ECAPA_KERNEL_SIZES,
                dilations=ECAPA_DILATIONS,
                attention_channels=ECAPA_ATTENTION_CHANNELS,
                lin_neurons=EMBEDDING_DIM,
            ),
            "mean_var_norm_emb": embedding_normalizer,
            "classifier": Classifier(
                input_size=EMBEDDING_DIM,
                out_neurons=PRETRAINED_CLASSIFIER_CLASSES,
            ),
        }
        self.spk_classifier = EncoderClassifier(
            modules=modules,
            hparams={"mean_var_norm_emb": embedding_normalizer},
            run_opts={"device": "cpu"},
        )

        # The notebook registers this shared module under both names. Recreating
        # the alias is necessary for an exact strict state_dict match.
        self.backbone = self.spk_classifier.mods.embedding_model
        self.fc = nn.Linear(EMBEDDING_DIM, num_classes)

    @property
    def device(self) -> torch.device:
        return self.fc.weight.device

    def _validate_model_input(self, waveforms: torch.Tensor) -> torch.Tensor:
        if not isinstance(waveforms, torch.Tensor):
            raise TypeError("waveforms must be a torch.Tensor.")
        if waveforms.ndim == 1:
            waveforms = waveforms.unsqueeze(0)
        if waveforms.ndim != 2 or waveforms.shape[1] != SEGMENT_SAMPLES:
            raise ValueError(
                f"waveforms must have shape [batch, {SEGMENT_SAMPLES}], "
                f"but received {tuple(waveforms.shape)}."
            )
        if waveforms.shape[0] == 0:
            raise ValueError("waveforms batch is empty.")
        if not waveforms.is_floating_point():
            raise ValueError("waveforms must contain floating-point samples.")
        if not torch.isfinite(waveforms).all():
            raise ValueError("waveforms contain NaN or infinite samples.")
        return waveforms.to(device=self.device, dtype=torch.float32)

    def _raw_embeddings(self, waveforms: torch.Tensor) -> torch.Tensor:
        waveforms = self._validate_model_input(waveforms)
        features = self.spk_classifier.mods.compute_features(waveforms)
        lengths = torch.ones(features.shape[0], device=features.device)
        features = self.spk_classifier.mods.mean_var_norm(features, lengths)
        return self.backbone(features, lengths=lengths).squeeze(1)

    def forward(self, waveforms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return notebook-compatible classification logits and raw embeddings."""

        embeddings = self._raw_embeddings(waveforms)
        return self.fc(embeddings), embeddings

    @torch.inference_mode()
    def extract_embedding(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized embeddings with shape ``[batch, 192]``."""

        embeddings = self._raw_embeddings(waveforms)
        normalized = F.normalize(embeddings, p=2, dim=1)
        if not torch.isfinite(normalized).all():
            raise RuntimeError("Model produced a non-finite embedding.")
        return normalized


def load_checkpoint(checkpoint_path: PathLike) -> Checkpoint:
    """Load the tensor-only checkpoint on CPU and validate its top-level type."""

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    if not path.is_file():
        raise CheckpointCompatibilityError(f"Checkpoint path is not a file: {path}")

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise CheckpointCompatibilityError(f"Could not load checkpoint: {path}") from exc

    if not isinstance(checkpoint, dict):
        raise CheckpointCompatibilityError(
            f"Expected a checkpoint dictionary, received {type(checkpoint).__name__}."
        )
    return checkpoint


def validate_checkpoint(checkpoint: Mapping[str, Any]) -> int:
    """Validate metadata and return the checkpoint's number of training classes."""

    required_keys = {"model_state_dict", "label2id", "id2label"}
    missing = sorted(required_keys.difference(checkpoint))
    if missing:
        raise CheckpointCompatibilityError(
            f"Checkpoint is missing required keys: {', '.join(missing)}"
        )

    state_dict = checkpoint["model_state_dict"]
    label2id = checkpoint["label2id"]
    id2label = checkpoint["id2label"]
    if not isinstance(state_dict, Mapping):
        raise CheckpointCompatibilityError("model_state_dict must be a mapping.")
    if not isinstance(label2id, Mapping) or not isinstance(id2label, Mapping):
        raise CheckpointCompatibilityError("label2id and id2label must be mappings.")

    for key in (
        "fc.weight",
        "fc.bias",
        "spk_classifier.mods.classifier.weight",
    ):
        if key not in state_dict or not isinstance(state_dict[key], torch.Tensor):
            raise CheckpointCompatibilityError(f"Missing checkpoint tensor: {key}")

    fc_weight = state_dict["fc.weight"]
    fc_bias = state_dict["fc.bias"]
    pretrained_weight = state_dict["spk_classifier.mods.classifier.weight"]
    if fc_weight.ndim != 2 or fc_weight.shape[1] != EMBEDDING_DIM:
        raise CheckpointCompatibilityError(
            f"fc.weight must have shape [classes, {EMBEDDING_DIM}], "
            f"received {tuple(fc_weight.shape)}."
        )

    num_classes = int(fc_weight.shape[0])
    if tuple(fc_bias.shape) != (num_classes,):
        raise CheckpointCompatibilityError(
            f"fc.bias shape {tuple(fc_bias.shape)} does not match {num_classes} classes."
        )
    if tuple(pretrained_weight.shape) != (
        PRETRAINED_CLASSIFIER_CLASSES,
        EMBEDDING_DIM,
    ):
        raise CheckpointCompatibilityError(
            "SpeechBrain classifier tensor does not match the authoritative "
            f"{PRETRAINED_CLASSIFIER_CLASSES}x{EMBEDDING_DIM} shape."
        )
    if len(label2id) != num_classes or len(id2label) != num_classes:
        raise CheckpointCompatibilityError(
            "Label mapping sizes do not match the classification head."
        )
    if sorted(label2id.values()) != list(range(num_classes)):
        raise CheckpointCompatibilityError("label2id values are not contiguous class IDs.")
    if any(id2label.get(class_id) != label for label, class_id in label2id.items()):
        raise CheckpointCompatibilityError("label2id and id2label are not exact inverses.")

    return num_classes


def build_model_from_checkpoint(
    checkpoint: Mapping[str, Any],
    device: DeviceLike = "cpu",
) -> ECAPAFinetuneModel:
    """Construct the exact architecture and strictly load a validated checkpoint."""

    num_classes = validate_checkpoint(checkpoint)
    model = ECAPAFinetuneModel(num_classes=num_classes)
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except RuntimeError as exc:
        raise CheckpointCompatibilityError(
            "Checkpoint state_dict is incompatible with the reconstructed model."
        ) from exc

    model.to(torch.device(device))
    model.eval()
    return model


def load_model(
    checkpoint_path: PathLike,
    device: Optional[DeviceLike] = None,
) -> ECAPAFinetuneModel:
    """Load the verified inference model on the requested or best available device."""

    selected_device: DeviceLike
    if device is None:
        selected_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        selected_device = device
    checkpoint = load_checkpoint(checkpoint_path)
    return build_model_from_checkpoint(checkpoint, selected_device)
