"""Separate historical single-window and multi-segment demo operating points.

Single-window thresholds came from the frozen validation protocol. Multi-segment
thresholds came from the separate operational demo calibration report.

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

# Operational demo calibration of aggregated multi-segment recording embeddings.
# These values are separate from the historical single-window operating points.
MULTISEGMENT_MODEL_THRESHOLDS: dict[str, float] = {
    "RAW": 0.20708201825618744,
    "RANDOM": 0.2140672206878662,
    "ADAPTIVE": 0.21084168553352356,
}

# Backward-compatible public name used by existing callers. These values must never
# be interpreted as calibrated operating points for multi-segment aggregation.
MODEL_THRESHOLDS = SINGLE_WINDOW_MODEL_THRESHOLDS


def single_window_threshold_for_model(label: str) -> Optional[float]:
    """Return the historical single-window threshold for a model label."""

    return SINGLE_WINDOW_MODEL_THRESHOLDS.get(label.upper())


def multisegment_threshold_for_model(label: str) -> Optional[float]:
    """Return the model's calibrated multi-segment demo operating threshold."""

    return MULTISEGMENT_MODEL_THRESHOLDS.get(label.upper())


def threshold_for_model(label: str) -> Optional[float]:
    """Backward-compatible alias for :func:`single_window_threshold_for_model`."""

    return single_window_threshold_for_model(label)
