"""High-level embedding extraction and cosine scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TypedDict, Union

import torch

from .audio import SpeechWindowMetadata, load_audio, load_audio_realtime
from .model import EMBEDDING_DIM, DeviceLike, FrozenHandoffECAPAModel, load_model

DEFAULT_CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[1] / "checkpoints" / "best_adp.pt"
)

PathLike = Union[str, Path]


class VerificationResult(TypedDict):
    similarity: float
    threshold: Optional[float]
    same_speaker: Optional[bool]


class RealtimeVerificationResult(VerificationResult):
    enrollment_metadata: SpeechWindowMetadata
    verification_metadata: SpeechWindowMetadata


@dataclass(frozen=True)
class RealtimeEmbedding:
    embedding: torch.Tensor
    metadata: SpeechWindowMetadata


def _single_embedding(embedding: torch.Tensor, *, name: str) -> torch.Tensor:
    if not isinstance(embedding, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if embedding.ndim == 2 and embedding.shape[0] == 1:
        embedding = embedding.squeeze(0)
    if embedding.shape != (EMBEDDING_DIM,):
        raise ValueError(
            f"{name} must have shape [{EMBEDDING_DIM}] or [1, {EMBEDDING_DIM}], "
            f"but received {tuple(embedding.shape)}."
        )
    if not torch.isfinite(embedding).all():
        raise ValueError(f"{name} contains NaN or infinite values.")

    # Scoring is deliberately device-independent and must not retain an autograd graph
    # from model inference or expose reduced-precision CUDA values.
    embedding = embedding.detach().to(device="cpu", dtype=torch.float32)
    norm = float(torch.linalg.vector_norm(embedding).item())
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-3):
        raise ValueError(f"{name} must be L2-normalized; observed norm {norm:.6f}.")
    return embedding


def score_embeddings(
    enrollment_embedding: torch.Tensor,
    verification_embedding: torch.Tensor,
) -> float:
    """Compute cosine similarity as a dot product of normalized embeddings."""

    enrollment = _single_embedding(enrollment_embedding, name="enrollment_embedding")
    verification = _single_embedding(verification_embedding, name="verification_embedding")
    # Both vectors are validated as unit length, so the dot product is cosine similarity.
    return float(torch.dot(enrollment, verification).item())


def _validate_threshold(threshold: Optional[float]) -> Optional[float]:
    if threshold is None:
        return None
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError("threshold must be None or a finite real number.")
    threshold_value = float(threshold)
    if not math.isfinite(threshold_value):
        raise ValueError("threshold must be finite.")
    return threshold_value


def decision_from_threshold(similarity: float, threshold: Optional[float]) -> Optional[bool]:
    """Return a same-speaker decision only when a calibrated threshold exists."""

    threshold_value = _validate_threshold(threshold)
    if threshold_value is None:
        return None
    return similarity >= threshold_value


class SpeakerVerifier:
    """Load one checkpoint once and perform repeated embedding/verification calls."""

    def __init__(
        self,
        checkpoint_path: PathLike = DEFAULT_CHECKPOINT_PATH,
        device: Optional[DeviceLike] = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.model: FrozenHandoffECAPAModel = load_model(self.checkpoint_path, device=device)

    @torch.inference_mode()
    def extract_embedding(self, audio: PathLike) -> torch.Tensor:
        """Return one baseline-preprocessed normalized CPU embedding."""

        waveform = load_audio(audio)
        embedding = self.model.extract_embedding(waveform)
        return embedding.squeeze(0).detach().cpu()

    @torch.inference_mode()
    def extract_embedding_realtime(self, audio: PathLike) -> RealtimeEmbedding:
        """Return a normalized embedding plus realtime selection metadata."""

        prepared = load_audio_realtime(audio)
        embedding = self.model.extract_embedding(prepared.waveform)
        return RealtimeEmbedding(
            embedding=embedding.squeeze(0).detach().cpu(),
            metadata=prepared.metadata,
        )

    def verify(
        self,
        enrollment_audio: PathLike,
        verification_audio: PathLike,
        threshold: Optional[float] = None,
    ) -> VerificationResult:
        # Extract both utterances with the same frozen frontend before applying the
        # threshold calibrated for that model condition.
        enrollment_embedding = self.extract_embedding(enrollment_audio)
        verification_embedding = self.extract_embedding(verification_audio)
        similarity = score_embeddings(enrollment_embedding, verification_embedding)
        threshold_value = _validate_threshold(threshold)
        return {
            "similarity": similarity,
            "threshold": threshold_value,
            "same_speaker": decision_from_threshold(similarity, threshold_value),
        }

    def verify_realtime(
        self,
        enrollment_audio: PathLike,
        verification_audio: PathLike,
        threshold: Optional[float] = None,
    ) -> RealtimeVerificationResult:
        enrollment = self.extract_embedding_realtime(enrollment_audio)
        verification = self.extract_embedding_realtime(verification_audio)
        similarity = score_embeddings(enrollment.embedding, verification.embedding)
        threshold_value = _validate_threshold(threshold)
        return {
            "similarity": similarity,
            "threshold": threshold_value,
            "same_speaker": decision_from_threshold(similarity, threshold_value),
            "enrollment_metadata": enrollment.metadata,
            "verification_metadata": verification.metadata,
        }
