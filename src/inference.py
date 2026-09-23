"""High-level embedding extraction and cosine scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TypedDict, Union

import torch

from .audio import (
    MultiSegmentMetadata,
    SpeechWindowMetadata,
    load_audio,
    load_audio_multisegment,
    load_audio_realtime,
)
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


class MultiSegmentVerificationResult(VerificationResult):
    enrollment_metadata: MultiSegmentMetadata
    verification_metadata: MultiSegmentMetadata


@dataclass(frozen=True)
class RealtimeEmbedding:
    embedding: torch.Tensor
    metadata: SpeechWindowMetadata


@dataclass(frozen=True)
class MultiSegmentEmbedding:
    embedding: torch.Tensor
    metadata: MultiSegmentMetadata


def aggregate_segment_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    """Arithmetic-mean segment embeddings, then return one normalized CPU vector."""

    if not isinstance(embeddings, torch.Tensor):
        raise TypeError("embeddings must be a torch.Tensor.")
    if embeddings.ndim != 2 or embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"embeddings must have shape [N, {EMBEDDING_DIM}], "
            f"but received {tuple(embeddings.shape)}."
        )
    if embeddings.shape[0] < 2:
        raise ValueError("At least two segment embeddings are required for aggregation.")
    values = embeddings.detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(values).all():
        raise ValueError("embeddings contain NaN or infinite values.")

    mean_embedding = values.mean(dim=0)
    mean_norm = torch.linalg.vector_norm(mean_embedding)
    if (
        not bool(torch.isfinite(mean_norm).item())
        or float(mean_norm.item()) <= torch.finfo(torch.float32).eps
    ):
        raise RuntimeError("Segment embeddings produced a zero or non-finite mean.")
    aggregated = (mean_embedding / mean_norm).to(torch.float32)
    if aggregated.shape != (EMBEDDING_DIM,) or not torch.isfinite(aggregated).all():
        raise RuntimeError("Aggregated embedding is invalid.")
    if not torch.isclose(
        torch.linalg.vector_norm(aggregated),
        torch.tensor(1.0),
        atol=1e-6,
        rtol=0.0,
    ):
        raise RuntimeError("Aggregated embedding is not L2-normalized.")
    return aggregated


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

    @torch.inference_mode()
    def extract_embedding_multisegment(self, audio: PathLike) -> MultiSegmentEmbedding:
        """Batch valid windows and aggregate them into one normalized embedding."""

        prepared = load_audio_multisegment(audio)
        # FrozenHandoffECAPAModel accepts [B, 48000], so all valid windows share one
        # ECAPA call rather than being inferred independently in Python.
        segment_embeddings = self.model.extract_embedding(prepared.windows)
        aggregated = aggregate_segment_embeddings(segment_embeddings)
        return MultiSegmentEmbedding(
            embedding=aggregated,
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

    def verify_multisegment(
        self,
        enrollment_audio: PathLike,
        verification_audio: PathLike,
    ) -> MultiSegmentVerificationResult:
        """Score aggregated embeddings without applying single-window thresholds."""

        enrollment = self.extract_embedding_multisegment(enrollment_audio)
        verification = self.extract_embedding_multisegment(verification_audio)
        similarity = score_embeddings(enrollment.embedding, verification.embedding)
        return {
            "similarity": similarity,
            "threshold": None,
            "same_speaker": None,
            "enrollment_metadata": enrollment.metadata,
            "verification_metadata": verification.metadata,
        }
