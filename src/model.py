"""Strict loader for the frozen-handoff ECAPA/AAM thesis checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.lobes.features import Fbank
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
from speechbrain.processing.features import InputNormalization

from .audio import SEGMENT_SAMPLES

CHECKPOINT_SCHEMA = "frozen_handoff_ecapa_aam_training_v1"
MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
EMBEDDING_DIM = 192
N_MELS = 80

# These values must match the SpeechBrain ECAPA topology used to create the frozen
# state_dict. Strict loading below intentionally rejects architecture drift.
ECAPA_CHANNELS = [1024, 1024, 1024, 1024, 3072]
ECAPA_KERNEL_SIZES = [5, 3, 3, 3, 1]
ECAPA_DILATIONS = [1, 2, 3, 4, 1]
ECAPA_ATTENTION_CHANNELS = 128

PathLike = Union[str, Path]
DeviceLike = Union[str, torch.device]
Checkpoint = dict[str, Any]


class CheckpointCompatibilityError(RuntimeError):
    """Raised when a checkpoint does not match the new inference contract."""


@dataclass(frozen=True)
class CheckpointMetadata:
    schema: str
    reason: Any
    model_source: str
    embedding_dim: int
    best_eer: Optional[float]
    embedding_state_entries: int
    mean_var_norm_state_entries: int
    aam_metadata: Mapping[str, Any]


class SpeechBrainECAPAFrontend(nn.Module):
    """SpeechBrain ECAPA frontend/modules used by the ex2 evaluation pipeline.

    The thesis checkpoints store only the fine-tuned ECAPA embedding model and
    mean-var-normalization states. AAM is training-only, so it is intentionally
    not constructed here.
    """

    SOURCE = MODEL_SOURCE
    SAMPLE_RATE = 16_000
    NUM_MEL_BINS = N_MELS
    EMBEDDING_DIM = EMBEDDING_DIM

    def __init__(self, device: DeviceLike = "cpu") -> None:
        super().__init__()
        self._device = torch.device(device)
        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")

        # Preserve the module names used by the original SpeechBrain classifier so
        # checkpoint keys map directly onto the feature, normalization, and ECAPA stages.
        self.classifier = SimpleNamespace(
            mods=nn.ModuleDict(
                {
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
                }
            )
        )
        self.classifier.mods.to(self._device)
        self.classifier.mods.eval()

    @property
    def device(self) -> torch.device:
        return self._device


class FrozenHandoffECAPAModel(nn.Module):
    """Inference-only wrapper exposing normalized ECAPA embeddings."""

    def __init__(self, frontend: SpeechBrainECAPAFrontend) -> None:
        super().__init__()
        self.frontend = frontend
        self.compute_features = frontend.classifier.mods.compute_features
        self.mean_var_norm = frontend.classifier.mods.mean_var_norm
        self.embedding_model = frontend.classifier.mods.embedding_model

    @property
    def device(self) -> torch.device:
        return self.frontend.device

    def _validate_waveforms(self, waveforms: torch.Tensor) -> torch.Tensor:
        if not isinstance(waveforms, torch.Tensor):
            raise TypeError("waveforms must be a torch.Tensor.")
        if waveforms.ndim == 1:
            waveforms = waveforms.unsqueeze(0)
        # Inference accepts [time] or [batch, time], but every item must already obey
        # the fixed 3-second preprocessing contract before feature extraction.
        if waveforms.ndim != 2 or waveforms.shape[1] != SEGMENT_SAMPLES:
            raise ValueError(
                f"waveforms must have shape [batch, {SEGMENT_SAMPLES}], "
                f"but received {tuple(waveforms.shape)}."
            )
        if waveforms.shape[0] == 0:
            raise ValueError("waveforms batch is empty.")
        if not waveforms.is_floating_point():
            raise ValueError("waveforms must contain floating-point samples.")
        waveforms = waveforms.to(device=self.device, dtype=torch.float32)
        if not torch.isfinite(waveforms).all():
            raise ValueError("waveforms contain NaN or infinite samples.")
        return waveforms

    @torch.inference_mode()
    def extract_embedding(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Return finite float32 L2-normalized embeddings shaped ``[B, 192]``."""

        prepared = self._validate_waveforms(waveforms)
        # SpeechBrain Fbank maps [B, 48000] waveforms to [B, frames, 80] log-Mel
        # features. All examples are full-length, so each relative length is 1.0.
        features = self.compute_features(prepared).to(torch.float32)
        if features.ndim != 3 or features.shape[-1] != N_MELS:
            raise RuntimeError(f"Unexpected SpeechBrain feature shape: {tuple(features.shape)}")
        lengths = torch.ones(features.shape[0], device=features.device)
        normalized = self.mean_var_norm(features, lengths)
        # Mixed precision is limited to CUDA ECAPA inference; outputs are converted
        # back to float32 before shape and normalization checks.
        with torch.cuda.amp.autocast(
            enabled=self.device.type == "cuda",
            dtype=torch.float16,
        ):
            embeddings = self.embedding_model(normalized, lengths).squeeze(1).float()
        if embeddings.shape != (prepared.shape[0], EMBEDDING_DIM):
            raise RuntimeError(
                f"Unexpected embedding shape: {tuple(embeddings.shape)}; "
                f"expected [{prepared.shape[0]}, {EMBEDDING_DIM}]."
            )
        # Unit-length embeddings make their dot product equal cosine similarity and
        # keep scoring independent of vector magnitude.
        embeddings = F.normalize(embeddings, p=2, dim=1).to(torch.float32)
        if not torch.isfinite(embeddings).all():
            raise RuntimeError("Model produced a non-finite embedding.")
        norms = torch.linalg.vector_norm(embeddings, dim=1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=0.0):
            raise RuntimeError("Model produced embeddings that are not unit-normalized.")
        return embeddings


def load_checkpoint(checkpoint_path: PathLike) -> Checkpoint:
    """Load a trusted project checkpoint on CPU."""

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    if not path.is_file():
        raise CheckpointCompatibilityError(f"Checkpoint path is not a file: {path}")

    try:
        # Checkpoints are trusted project artifacts and include metadata beyond tensor
        # weights. CPU loading also avoids coupling validation to GPU availability.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise CheckpointCompatibilityError(f"Could not load checkpoint: {path}") from exc

    if not isinstance(checkpoint, dict):
        raise CheckpointCompatibilityError(
            f"Expected a checkpoint dictionary, received {type(checkpoint).__name__}."
        )
    return checkpoint


def _require_mapping(checkpoint: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = checkpoint.get(key)
    if not isinstance(value, Mapping):
        raise CheckpointCompatibilityError(f"{key} must be a mapping.")
    return value


def validate_checkpoint(checkpoint: Mapping[str, Any]) -> CheckpointMetadata:
    """Validate the frozen-handoff schema and return useful metadata."""

    required_keys = {
        "schema",
        "runtime_config",
        "embedding_model_state_dict",
        "mean_var_norm_state_dict",
        "cursor",
    }
    missing = sorted(required_keys.difference(checkpoint))
    if missing:
        raise CheckpointCompatibilityError(
            f"Checkpoint is missing required keys: {', '.join(missing)}"
        )
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise CheckpointCompatibilityError(
            f"Unsupported checkpoint schema: {checkpoint.get('schema')!r}."
        )

    # Schema alone is insufficient: matching source and embedding width prevent a
    # structurally valid checkpoint from being paired with the wrong architecture.
    runtime_config = _require_mapping(checkpoint, "runtime_config")
    model_config = runtime_config.get("model")
    if not isinstance(model_config, Mapping):
        raise CheckpointCompatibilityError("runtime_config.model must be a mapping.")
    model_source = model_config.get("source")
    embedding_dim = model_config.get("embedding_dim")
    if model_source != MODEL_SOURCE or embedding_dim != EMBEDDING_DIM:
        raise CheckpointCompatibilityError(
            "runtime_config.model must specify "
            f"{MODEL_SOURCE!r} with embedding_dim={EMBEDDING_DIM}."
        )

    embedding_state = _require_mapping(checkpoint, "embedding_model_state_dict")
    mean_var_norm_state = _require_mapping(checkpoint, "mean_var_norm_state_dict")
    _require_mapping(checkpoint, "cursor")
    if not embedding_state:
        raise CheckpointCompatibilityError(
            "embedding_model_state_dict must contain at least one tensor."
        )

    cursor = checkpoint["cursor"]
    best_eer = cursor.get("best_eer") if isinstance(cursor, Mapping) else None
    aam_config = runtime_config.get("aam", {})
    return CheckpointMetadata(
        schema=CHECKPOINT_SCHEMA,
        reason=checkpoint.get("reason"),
        model_source=str(model_source),
        embedding_dim=int(embedding_dim),
        best_eer=None if best_eer is None else float(best_eer),
        embedding_state_entries=len(embedding_state),
        mean_var_norm_state_entries=len(mean_var_norm_state),
        aam_metadata=aam_config if isinstance(aam_config, Mapping) else {},
    )


def build_model_from_checkpoint(
    checkpoint: Mapping[str, Any],
    device: DeviceLike = "cpu",
    *,
    frontend_factory: type[SpeechBrainECAPAFrontend] = SpeechBrainECAPAFrontend,
) -> FrozenHandoffECAPAModel:
    """Construct the inference model and strictly load checkpoint weights."""

    validate_checkpoint(checkpoint)
    # Materialize and validate weights on CPU first. The complete, verified model is
    # moved to the requested device only after both state_dict loads succeed strictly.
    frontend = frontend_factory(device="cpu")
    mean_var_norm = frontend.classifier.mods.mean_var_norm
    embedding_model = frontend.classifier.mods.embedding_model
    try:
        mean_var_norm.load_state_dict(
            checkpoint["mean_var_norm_state_dict"],
            strict=True,
        )
        embedding_model.load_state_dict(
            checkpoint["embedding_model_state_dict"],
            strict=True,
        )
    except RuntimeError as exc:
        raise CheckpointCompatibilityError(
            "Checkpoint state_dict is incompatible with the SpeechBrain ECAPA frontend."
        ) from exc

    model = FrozenHandoffECAPAModel(frontend)
    model.to(torch.device(device))
    model.eval()
    return model


def load_model(
    checkpoint_path: PathLike,
    device: Optional[DeviceLike] = None,
) -> FrozenHandoffECAPAModel:
    """Load a strict frozen-handoff inference model on CUDA when available."""

    selected_device: DeviceLike = (
        "cuda:0" if device is None and torch.cuda.is_available() else device or "cpu"
    )
    checkpoint = load_checkpoint(checkpoint_path)
    return build_model_from_checkpoint(checkpoint, selected_device)
