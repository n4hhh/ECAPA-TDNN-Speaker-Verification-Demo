"""Pure trial and empirical operating-point logic for demo calibration.

The nearest empirical FAR/FRR point matches the thesis notebook's EER
convention: choose the first (highest-threshold) ROC point minimizing
``abs(FAR - FRR)``, then report ``(FAR + FRR) / 2``. There is no interpolation.
All observed unique similarity scores are evaluated, so no attainable
operating point is skipped.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import torch

from src.audio import TARGET_SAMPLE_RATE

CALIBRATION_DURATION_SECONDS = 10.0
CALIBRATION_SAMPLES = int(CALIBRATION_DURATION_SECONDS * TARGET_SAMPLE_RATE)
EXPECTED_SPEAKERS = 9
EXPECTED_RECORDINGS_PER_SPEAKER = 3
EXPECTED_TOTAL_RECORDINGS = EXPECTED_SPEAKERS * EXPECTED_RECORDINGS_PER_SPEAKER


@dataclass(frozen=True)
class Recording:
    """One independent source recording; identity comes from its parent folder."""

    speaker_id: str
    recording_id: str
    relative_path: str


@dataclass(frozen=True)
class Trial:
    trial_id: str
    label: int
    speaker_a: str
    recording_a: str
    path_a: str
    speaker_b: str
    recording_b: str
    path_b: str


@dataclass(frozen=True)
class OperatingPoint:
    eer: float
    eer_threshold: float
    far_at_operating_point: float
    frr_at_operating_point: float


def center_crop_10_seconds(waveform: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """Return exactly 160,000 centered mono 16 kHz samples without padding."""

    if not isinstance(waveform, torch.Tensor) or waveform.ndim != 1:
        raise ValueError("Center crop requires a mono waveform.")
    if waveform.numel() < CALIBRATION_SAMPLES:
        raise ValueError("Source audio is shorter than the required 10-second crop.")
    start = (waveform.numel() - CALIBRATION_SAMPLES) // 2
    end = start + CALIBRATION_SAMPLES
    return waveform[start:end].contiguous(), start, end


def make_trials(recordings: Sequence[Recording]) -> tuple[Trial, ...]:
    """Generate every unique unordered recording pair in a stable order."""

    ordered = tuple(sorted(recordings, key=lambda r: (r.speaker_id, r.recording_id)))
    if len({r.relative_path for r in ordered}) != len(ordered):
        raise ValueError("Recording paths must be unique.")
    return tuple(
        Trial(
            trial_id=f"T{index:04d}",
            label=int(left.speaker_id == right.speaker_id),
            speaker_a=left.speaker_id,
            recording_a=left.recording_id,
            path_a=left.relative_path,
            speaker_b=right.speaker_id,
            recording_b=right.recording_id,
            path_b=right.relative_path,
        )
        for index, (left, right) in enumerate(combinations(ordered, 2), start=1)
    )


def empirical_eer_operating_point(
    labels: Sequence[int], scores: Sequence[float]
) -> OperatingPoint:
    """Match the thesis notebook's nearest empirical FAR/FRR EER convention.

    A trial is accepted exactly when ``similarity >= threshold``. Candidate
    thresholds are one finite value above the maximum and every observed score,
    descending. Ties in ``abs(FAR-FRR)`` choose the highest threshold, just as
    ``np.nanargmin`` chooses the first ROC point. ``eer`` is a fraction, not percent.
    """

    if len(labels) != len(scores) or not labels:
        raise ValueError("Labels and scores must have the same nonzero length.")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("Trial labels must be 0 or 1.")
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("Similarity scores must be finite.")
    genuine_count = sum(label == 1 for label in labels)
    impostor_count = len(labels) - genuine_count
    if not genuine_count or not impostor_count:
        raise ValueError("Both genuine and impostor trials are required.")

    unique_scores = sorted(set(scores), reverse=True)
    above_max = math.nextafter(unique_scores[0], math.inf)
    if not math.isfinite(above_max):
        raise ValueError("Cannot construct a finite threshold above the maximum score.")
    candidates = (above_max, *unique_scores)
    points = []
    for threshold in candidates:
        false_accepts = sum(
            label == 0 and score >= threshold
            for label, score in zip(labels, scores)
        )
        false_rejects = sum(
            label == 1 and score < threshold
            for label, score in zip(labels, scores)
        )
        far = false_accepts / impostor_count
        frr = false_rejects / genuine_count
        points.append((abs(far - frr), threshold, far, frr))

    # Candidate order is descending; min() keeps the first tied point.
    _, threshold, far, frr = min(points, key=lambda point: point[0])
    return OperatingPoint(
        eer=(far + frr) / 2.0,
        eer_threshold=threshold,
        far_at_operating_point=far,
        frr_at_operating_point=frr,
    )


def score_statistics(values: Sequence[float]) -> dict[str, float]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("Score statistics require finite, nonempty values.")
    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
    }
