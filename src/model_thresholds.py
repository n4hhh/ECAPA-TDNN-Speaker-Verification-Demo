"""Historical single-window thresholds for thesis model conditions.

Thresholds were recovered from the frozen validation protocol using the
empirical EER operating point.

Decision rule:
    same speaker if cosine_similarity >= threshold

Final-test data is not used for threshold selection.
"""

from __future__ import annotations

from typing import Optional

SINGLE_WINDOW_MODEL_THRESHOLDS: dict[str, Optional[float]] = {
    "RAW": 0.16837078332901,
    "RANDOM": 0.16918502748012543,
    "ADAPTIVE": 0.172615185379982,
}

# Backward-compatible public name used by existing callers. These values must never
# be interpreted as calibrated operating points for multi-segment aggregation.
MODEL_THRESHOLDS = SINGLE_WINDOW_MODEL_THRESHOLDS


def single_window_threshold_for_model(label: str) -> Optional[float]:
    """Return the historical single-window threshold for a model label."""

    return SINGLE_WINDOW_MODEL_THRESHOLDS.get(label.upper())


def threshold_for_model(label: str) -> Optional[float]:
    """Backward-compatible alias for :func:`single_window_threshold_for_model`."""

    return single_window_threshold_for_model(label)
