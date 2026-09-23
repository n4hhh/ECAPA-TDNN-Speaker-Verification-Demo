"""Validation-calibrated operational thresholds for thesis model conditions.

Thresholds were recovered from the frozen validation protocol using the
empirical EER operating point.

Decision rule:
    same speaker if cosine_similarity >= threshold

Final-test data is not used for threshold selection.
"""

from __future__ import annotations

from typing import Optional

MODEL_THRESHOLDS: dict[str, Optional[float]] = {
    "RAW": 0.16837078332901,
    "RANDOM": 0.16918502748012543,
    "ADAPTIVE": 0.172615185379982,
}


def threshold_for_model(label: str) -> Optional[float]:
    """Return the validation-calibrated threshold for a case-insensitive model label."""

    return MODEL_THRESHOLDS.get(label.upper())
