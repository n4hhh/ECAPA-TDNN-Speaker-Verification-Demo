from __future__ import annotations

import math
import unittest

import torch

from src.audio import (
    MULTISEGMENT_HOP_SAMPLES,
    MULTISEGMENT_MIN_SPEECH_PER_WINDOW_SECONDS,
    MULTISEGMENT_WINDOW_SAMPLES,
    TARGET_SAMPLE_RATE,
    InsufficientMultiSegmentSpeechError,
    InsufficientValidSegmentsError,
    RecordingTooLongError,
    multisegment_window_starts,
    select_multisegment_windows,
)
from src.vad import SpeechActivity, VAD_FRAME_DURATION_MS

FRAME_SAMPLES = TARGET_SAMPLE_RATE * VAD_FRAME_DURATION_MS // 1_000


def make_activity(num_samples: int, intervals: list[tuple[float, float]]) -> SpeechActivity:
    frame_count = math.ceil(num_samples / FRAME_SAMPLES)
    flags = []
    for index in range(frame_count):
        frame_start = index * FRAME_SAMPLES / TARGET_SAMPLE_RATE
        frame_end = min((index + 1) * FRAME_SAMPLES, num_samples) / TARGET_SAMPLE_RATE
        flags.append(any(frame_start < end and frame_end > start for start, end in intervals))
    return SpeechActivity(
        sample_rate=TARGET_SAMPLE_RATE,
        frame_duration_ms=VAD_FRAME_DURATION_MS,
        frame_samples=FRAME_SAMPLES,
        num_samples=num_samples,
        frame_is_speech=tuple(flags),
    )


def make_waveform(seconds: float) -> torch.Tensor:
    return torch.arange(
        int(seconds * TARGET_SAMPLE_RATE),
        dtype=torch.float32,
    ) / TARGET_SAMPLE_RATE


class MultiSegmentWindowTests(unittest.TestCase):
    def test_window_and_hop_constants_are_exact(self) -> None:
        self.assertEqual(MULTISEGMENT_WINDOW_SAMPLES, 48_000)
        self.assertEqual(MULTISEGMENT_HOP_SAMPLES, 24_000)

    def test_candidate_starts_are_deterministic_and_tail_is_discarded(self) -> None:
        num_samples = int(9.2 * TARGET_SAMPLE_RATE)
        expected = (0, 24_000, 48_000, 72_000, 96_000)
        self.assertEqual(multisegment_window_starts(num_samples), expected)
        self.assertLess(expected[-1] + MULTISEGMENT_WINDOW_SAMPLES, num_samples)

        waveform = make_waveform(9.2)
        activity = make_activity(waveform.numel(), [(0.0, 9.2)])
        prepared = select_multisegment_windows(waveform, speech_activity=activity)
        self.assertEqual(prepared.windows.shape, (5, 48_000))
        for index, start in enumerate(expected):
            self.assertTrue(
                torch.equal(
                    prepared.windows[index],
                    waveform[start : start + MULTISEGMENT_WINDOW_SAMPLES],
                )
            )

    def test_separated_speech_is_never_concatenated(self) -> None:
        waveform = make_waveform(9.0)
        activity = make_activity(
            waveform.numel(),
            [(0.0, 2.0), (3.0, 5.0), (6.0, 8.0)],
        )
        prepared = select_multisegment_windows(waveform, speech_activity=activity)
        for window, segment in zip(
            prepared.windows,
            prepared.metadata.valid_segments,
            strict=True,
        ):
            start = round(segment.start_seconds * TARGET_SAMPLE_RATE)
            self.assertTrue(
                torch.equal(window, waveform[start : start + MULTISEGMENT_WINDOW_SAMPLES])
            )

    def test_exact_speech_threshold_is_inclusive(self) -> None:
        waveform = make_waveform(6.0)
        activity = make_activity(waveform.numel(), [(1.5, 4.5)])
        prepared = select_multisegment_windows(waveform, speech_activity=activity)
        self.assertEqual(prepared.metadata.valid_window_count, 3)
        self.assertAlmostEqual(
            prepared.metadata.valid_segments[0].speech_seconds,
            MULTISEGMENT_MIN_SPEECH_PER_WINDOW_SECONDS,
        )
        self.assertAlmostEqual(
            prepared.metadata.valid_segments[-1].speech_seconds,
            MULTISEGMENT_MIN_SPEECH_PER_WINDOW_SECONDS,
        )

    def test_window_below_speech_threshold_is_filtered(self) -> None:
        waveform = make_waveform(7.5)
        activity = make_activity(
            waveform.numel(),
            [(0.0, 1.48), (3.0, 6.0)],
        )
        prepared = select_multisegment_windows(waveform, speech_activity=activity)
        starts = [segment.start_seconds for segment in prepared.metadata.valid_segments]
        self.assertNotIn(0.0, starts)
        self.assertEqual(starts, [1.5, 3.0, 4.5])

    def test_total_speech_below_three_seconds_is_rejected_with_metadata(self) -> None:
        waveform = make_waveform(6.0)
        activity = make_activity(waveform.numel(), [(0.0, 2.98)])
        with self.assertRaises(InsufficientMultiSegmentSpeechError) as caught:
            select_multisegment_windows(waveform, speech_activity=activity)
        self.assertAlmostEqual(caught.exception.metadata.total_speech_seconds, 2.98)
        self.assertFalse(caught.exception.metadata.sufficient_speech)

    def test_fewer_than_two_valid_windows_is_rejected(self) -> None:
        waveform = make_waveform(10.0)
        activity = make_activity(
            waveform.numel(),
            [(0.0, 1.0), (4.0, 5.0), (8.0, 9.0)],
        )
        with self.assertRaises(InsufficientValidSegmentsError) as caught:
            select_multisegment_windows(waveform, speech_activity=activity)
        self.assertGreaterEqual(caught.exception.metadata.total_speech_seconds, 3.0)
        self.assertLess(caught.exception.metadata.valid_window_count, 2)

    def test_recording_longer_than_twenty_seconds_is_rejected(self) -> None:
        waveform = make_waveform(20.02)
        with self.assertRaises(RecordingTooLongError):
            select_multisegment_windows(waveform)


if __name__ == "__main__":
    unittest.main()

