from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

import torch

from scripts.verify_pair import main
from src.audio import InsufficientSpeechError, SpeechWindowMetadata
from src.inference import RealtimeEmbedding
from src.model import EMBEDDING_DIM


def metadata(*, sufficient: bool = True) -> SpeechWindowMetadata:
    return SpeechWindowMetadata(
        original_duration_seconds=5.0,
        selected_start_seconds=1.0,
        selected_end_seconds=4.0,
        speech_in_selected_window_seconds=2.4 if sufficient else 0.4,
        total_speech_seconds=3.0 if sufficient else 0.4,
        sufficient_speech=sufficient,
    )


def unit_embedding() -> torch.Tensor:
    embedding = torch.zeros(EMBEDDING_DIM)
    embedding[0] = 1.0
    return embedding


class VerifyPairCliTests(unittest.TestCase):
    @patch("scripts.verify_pair.SpeakerVerifier")
    def test_success_output(self, verifier_class) -> None:
        verifier_class.return_value.extract_embedding_realtime.side_effect = [
            RealtimeEmbedding(unit_embedding(), metadata()),
            RealtimeEmbedding(unit_embedding(), metadata()),
        ]
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(["enrollment.wav", "verification.wav"])

        self.assertEqual(exit_code, 0)
        rendered = output.getvalue()
        self.assertIn("Enrollment:", rendered)
        self.assertIn("Verification:", rendered)
        self.assertIn("cosine similarity: 1.000000", rendered)
        self.assertIn("operational threshold: not configured", rendered)
        self.assertIn("decision: unavailable", rendered)

    @patch("scripts.verify_pair.SpeakerVerifier")
    def test_manual_threshold_output(self, verifier_class) -> None:
        verifier_class.return_value.extract_embedding_realtime.side_effect = [
            RealtimeEmbedding(unit_embedding(), metadata()),
            RealtimeEmbedding(unit_embedding(), metadata()),
        ]
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(["enrollment.wav", "verification.wav", "--threshold", "0.5"])

        self.assertEqual(exit_code, 0)
        rendered = output.getvalue()
        self.assertIn("manual/development threshold: 0.500000", rendered)
        self.assertIn("SAME SPEAKER", rendered)

    @patch("scripts.verify_pair.SpeakerVerifier")
    def test_insufficient_speech_returns_clear_error(self, verifier_class) -> None:
        verifier_class.return_value.extract_embedding_realtime.side_effect = (
            InsufficientSpeechError(metadata(sufficient=False))
        )
        output = io.StringIO()
        errors = io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            exit_code = main(["enrollment.wav", "verification.wav"])

        self.assertEqual(exit_code, 2)
        self.assertIn("Enrollment:", output.getvalue())
        self.assertIn("Not enough speech detected", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
