from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import torch

from src.audio import (
    MultiSegmentMetadata,
    PreparedMultiSegmentAudio,
    ValidSegmentMetadata,
)
from src.inference import (
    MultiSegmentEmbedding,
    SpeakerVerifier,
    aggregate_segment_embeddings,
)
from src.model import EMBEDDING_DIM


def metadata(count: int = 3) -> MultiSegmentMetadata:
    segments = tuple(
        ValidSegmentMetadata(index * 1.5, index * 1.5 + 3.0, 2.5)
        for index in range(count)
    )
    return MultiSegmentMetadata(
        original_duration_seconds=6.0,
        total_speech_seconds=5.0,
        candidate_window_count=count,
        valid_window_count=count,
        window_seconds=3.0,
        hop_seconds=1.5,
        valid_segments=segments,
        sufficient_speech=True,
    )


class RecordingBatchModel:
    def __init__(self) -> None:
        self.calls: list[torch.Tensor] = []

    def extract_embedding(self, windows: torch.Tensor) -> torch.Tensor:
        self.calls.append(windows)
        result = torch.zeros(windows.shape[0], EMBEDDING_DIM)
        for index in range(windows.shape[0]):
            result[index, index] = 1.0
        return result


class MultiSegmentInferenceTests(unittest.TestCase):
    def test_valid_windows_use_one_batch_model_call(self) -> None:
        windows = torch.randn(3, 48_000)
        prepared = PreparedMultiSegmentAudio(windows=windows, metadata=metadata())
        verifier = object.__new__(SpeakerVerifier)
        verifier.model = RecordingBatchModel()

        with patch("src.inference.load_audio_multisegment", return_value=prepared):
            result = verifier.extract_embedding_multisegment("sample.wav")

        self.assertEqual(len(verifier.model.calls), 1)
        self.assertIs(verifier.model.calls[0], windows)
        self.assertEqual(tuple(verifier.model.calls[0].shape), (3, 48_000))
        self.assertEqual(result.metadata, prepared.metadata)
        self.assertEqual(result.embedding.shape, (EMBEDDING_DIM,))
        self.assertEqual(result.embedding.dtype, torch.float32)
        self.assertTrue(torch.isfinite(result.embedding).all())
        self.assertAlmostEqual(
            float(torch.linalg.vector_norm(result.embedding)),
            1.0,
            places=6,
        )

    def test_aggregation_is_arithmetic_mean_then_l2_normalization(self) -> None:
        embeddings = torch.zeros(2, EMBEDDING_DIM, dtype=torch.float64)
        embeddings[0, 0] = 1.0
        embeddings[1, 1] = 1.0
        result = aggregate_segment_embeddings(embeddings)
        expected = torch.zeros(EMBEDDING_DIM)
        expected[0] = 2**-0.5
        expected[1] = 2**-0.5
        self.assertTrue(torch.allclose(result, expected, atol=1e-7, rtol=0.0))
        self.assertEqual(result.dtype, torch.float32)

    def test_aggregation_is_deterministic(self) -> None:
        embeddings = torch.zeros(3, EMBEDDING_DIM)
        embeddings[:, 0] = torch.tensor([1.0, 0.8, 0.6])
        embeddings[:, 1] = torch.tensor([0.0, 0.6, 0.8])
        first = aggregate_segment_embeddings(embeddings)
        second = aggregate_segment_embeddings(embeddings.clone())
        self.assertTrue(torch.equal(first, second))

    def test_multisegment_verification_never_returns_a_decision(self) -> None:
        verifier = object.__new__(SpeakerVerifier)
        first = torch.zeros(EMBEDDING_DIM)
        second = torch.zeros(EMBEDDING_DIM)
        first[0] = 1.0
        second[0] = 0.8
        second[1] = 0.6
        verifier.extract_embedding_multisegment = Mock(
            side_effect=[
                MultiSegmentEmbedding(first, metadata(2)),
                MultiSegmentEmbedding(second, metadata(2)),
            ]
        )

        result = verifier.verify_multisegment("enroll.wav", "verify.wav")

        self.assertAlmostEqual(result["similarity"], 0.8)
        self.assertIsNone(result["threshold"])
        self.assertIsNone(result["same_speaker"])


if __name__ == "__main__":
    unittest.main()
