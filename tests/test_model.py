from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from src.audio import SEGMENT_SAMPLES
from src.inference import decision_from_threshold, score_embeddings
from src.model import (
    CHECKPOINT_SCHEMA,
    EMBEDDING_DIM,
    MODEL_SOURCE,
    CheckpointCompatibilityError,
    build_model_from_checkpoint,
    validate_checkpoint,
)


class FakeFeatures(nn.Module):
    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        return torch.zeros(waveforms.shape[0], 301, 80, device=waveforms.device)


class FakeNorm(nn.Module):
    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return features


class FakeEmbeddingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(EMBEDDING_DIM))

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        value = self.weight.view(1, 1, EMBEDDING_DIM).repeat(features.shape[0], 1, 1)
        return value


class FakeFrontend(nn.Module):
    def __init__(self, device: str = "cpu") -> None:
        super().__init__()
        self._device = torch.device(device)
        self.classifier = SimpleNamespace(
            mods=nn.ModuleDict(
                {
                    "compute_features": FakeFeatures(),
                    "mean_var_norm": FakeNorm(),
                    "embedding_model": FakeEmbeddingModel(),
                }
            )
        )

    @property
    def device(self) -> torch.device:
        return self._device


def checkpoint() -> dict[str, object]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "reason": "unit-test",
        "runtime_config": {
            "model": {"source": MODEL_SOURCE, "embedding_dim": EMBEDDING_DIM},
            "aam": {"margin": 0.2, "scale": 30.0, "num_classes": 2},
        },
        "embedding_model_state_dict": FakeEmbeddingModel().state_dict(),
        "mean_var_norm_state_dict": {},
        "cursor": {"best_eer": 0.25},
    }


class CheckpointValidationTests(unittest.TestCase):
    def test_new_checkpoint_schema_is_accepted(self) -> None:
        metadata = validate_checkpoint(checkpoint())
        self.assertEqual(metadata.schema, CHECKPOINT_SCHEMA)
        self.assertEqual(metadata.embedding_dim, EMBEDDING_DIM)
        self.assertEqual(metadata.mean_var_norm_state_entries, 0)
        self.assertEqual(metadata.best_eer, 0.25)

    def test_old_checkpoint_schema_is_rejected(self) -> None:
        with self.assertRaises(CheckpointCompatibilityError):
            validate_checkpoint(
                {
                    "model_state_dict": {},
                    "label2id": {},
                    "id2label": {},
                }
            )

    def test_missing_embedding_state_is_rejected(self) -> None:
        value = checkpoint()
        value.pop("embedding_model_state_dict")
        with self.assertRaisesRegex(CheckpointCompatibilityError, "embedding_model"):
            validate_checkpoint(value)

    def test_wrong_schema_is_rejected(self) -> None:
        value = checkpoint()
        value["schema"] = "generic_ecapa_aam_training_v3"
        with self.assertRaisesRegex(CheckpointCompatibilityError, "Unsupported"):
            validate_checkpoint(value)

    def test_strict_load_and_embedding_shape(self) -> None:
        model = build_model_from_checkpoint(
            checkpoint(),
            device="cpu",
            frontend_factory=FakeFrontend,
        )
        embedding = model.extract_embedding(torch.zeros(1, SEGMENT_SAMPLES))
        self.assertEqual(tuple(embedding.shape), (1, EMBEDDING_DIM))
        self.assertTrue(torch.isfinite(embedding).all())
        norm = float(torch.linalg.vector_norm(embedding, dim=1).item())
        self.assertAlmostEqual(norm, 1.0, places=6)

    def test_threshold_none_does_not_make_decision(self) -> None:
        self.assertIsNone(decision_from_threshold(0.9, None))
        self.assertTrue(decision_from_threshold(0.9, 0.5))

    def test_cosine_scoring(self) -> None:
        left = torch.zeros(EMBEDDING_DIM)
        right = torch.zeros(EMBEDDING_DIM)
        left[0] = 1.0
        right[0] = -1.0
        self.assertAlmostEqual(score_embeddings(left, right), -1.0)


if __name__ == "__main__":
    unittest.main()
