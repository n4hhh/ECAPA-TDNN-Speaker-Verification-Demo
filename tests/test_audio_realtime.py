from __future__ import annotations

import math
import unittest

import torch

from src.audio import (
    INSUFFICIENT_SPEECH_MESSAGE,
    MIN_SPEECH_IN_SELECTED_WINDOW_SECONDS,
    MIN_TOTAL_SPEECH_SECONDS,
    SEGMENT_SAMPLES,
    TARGET_SAMPLE_RATE,
    AudioValidationError,
    InsufficientSpeechError,
    convert_to_mono_16k,
    prepare_audio_realtime,
    select_speech_rich_window,
)
from src.vad import SpeechActivity, VAD_FRAME_DURATION_MS, detect_speech_activity


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
    samples = int(seconds * TARGET_SAMPLE_RATE)
    return torch.linspace(-0.25, 0.25, samples, dtype=torch.float32)


class RealtimeAudioTests(unittest.TestCase):
    def test_silence_before_speech_selects_speech_region(self) -> None:
        waveform = make_waveform(6.0)
        activity = make_activity(waveform.numel(), [(2.0, 5.0)])
        prepared = select_speech_rich_window(waveform, speech_activity=activity)
        self.assertAlmostEqual(prepared.metadata.selected_start_seconds, 2.0, places=6)
        self.assertAlmostEqual(prepared.metadata.selected_end_seconds, 5.0, places=6)

    def test_silence_after_speech_selects_speech_region(self) -> None:
        waveform = make_waveform(6.0)
        activity = make_activity(waveform.numel(), [(0.5, 3.5)])
        prepared = select_speech_rich_window(waveform, speech_activity=activity)
        self.assertAlmostEqual(prepared.metadata.selected_start_seconds, 0.5, places=6)

    def test_pauses_and_equal_scores_use_earliest_start(self) -> None:
        waveform = make_waveform(6.0)
        activity = make_activity(
            waveform.numel(),
            [(1.0, 2.0), (2.5, 3.5), (4.0, 5.0)],
        )
        first = select_speech_rich_window(waveform, speech_activity=activity)
        second = select_speech_rich_window(waveform, speech_activity=activity)
        self.assertAlmostEqual(first.metadata.selected_start_seconds, 0.5, places=6)
        self.assertEqual(first.metadata, second.metadata)
        self.assertTrue(torch.equal(first.waveform, second.waveform))

    def test_long_selection_is_exact_contiguous_slice(self) -> None:
        waveform = make_waveform(7.0)
        activity = make_activity(waveform.numel(), [(3.0, 6.0)])
        prepared = select_speech_rich_window(waveform, speech_activity=activity)
        start = int(prepared.metadata.selected_start_seconds * TARGET_SAMPLE_RATE)
        self.assertEqual(prepared.waveform.shape, (SEGMENT_SAMPLES,))
        self.assertTrue(
            torch.equal(prepared.waveform, waveform[start : start + SEGMENT_SAMPLES])
        )

    def test_short_recording_validates_then_right_pads(self) -> None:
        waveform = make_waveform(2.5)
        activity = make_activity(waveform.numel(), [(0.0, 2.5)])
        prepared = select_speech_rich_window(waveform, speech_activity=activity)
        self.assertEqual(prepared.waveform.shape, (SEGMENT_SAMPLES,))
        self.assertTrue(torch.equal(prepared.waveform[: waveform.numel()], waveform))
        self.assertEqual(torch.count_nonzero(prepared.waveform[waveform.numel() :]), 0)
        self.assertAlmostEqual(prepared.metadata.selected_end_seconds, 2.5)

    def test_stereo_conversion_precedes_activity_selection(self) -> None:
        mono = make_waveform(3.0)
        stereo = torch.stack((mono, mono * 0.5))
        converted = convert_to_mono_16k(stereo, TARGET_SAMPLE_RATE)
        activity = make_activity(converted.numel(), [(0.0, 3.0)])
        prepared = prepare_audio_realtime(
            stereo,
            TARGET_SAMPLE_RATE,
            speech_activity=activity,
        )
        self.assertEqual(prepared.waveform.shape, (SEGMENT_SAMPLES,))
        self.assertTrue(torch.isfinite(prepared.waveform).all())

    def test_non_16khz_input_is_resampled_before_selection(self) -> None:
        source_rate = 44_100
        waveform = torch.linspace(-0.2, 0.2, source_rate * 3, dtype=torch.float32)
        converted = convert_to_mono_16k(waveform, source_rate)
        activity = make_activity(converted.numel(), [(0.0, 3.0)])
        prepared = prepare_audio_realtime(
            waveform,
            source_rate,
            speech_activity=activity,
        )
        self.assertEqual(prepared.waveform.shape, (SEGMENT_SAMPLES,))
        self.assertAlmostEqual(prepared.metadata.original_duration_seconds, 3.0)

    def test_silence_is_rejected_by_real_vad(self) -> None:
        silence = torch.zeros(TARGET_SAMPLE_RATE * 4, dtype=torch.float32)
        activity = detect_speech_activity(silence, TARGET_SAMPLE_RATE)
        self.assertEqual(activity.total_speech_samples, 0)
        with self.assertRaisesRegex(InsufficientSpeechError, INSUFFICIENT_SPEECH_MESSAGE):
            select_speech_rich_window(silence, speech_activity=activity)

    def test_insufficient_speech_exposes_failed_metadata(self) -> None:
        waveform = make_waveform(5.0)
        activity = make_activity(waveform.numel(), [(1.0, 2.0)])
        with self.assertRaises(InsufficientSpeechError) as caught:
            select_speech_rich_window(waveform, speech_activity=activity)
        self.assertFalse(caught.exception.metadata.sufficient_speech)
        self.assertLess(
            caught.exception.metadata.total_speech_seconds,
            MIN_TOTAL_SPEECH_SECONDS,
        )

    def test_total_and_window_minimums_are_independent(self) -> None:
        waveform = make_waveform(8.0)
        activity = make_activity(
            waveform.numel(),
            [(0.0, 1.0), (4.0, 5.0)],
        )
        self.assertGreaterEqual(activity.total_speech_seconds, MIN_TOTAL_SPEECH_SECONDS)
        with self.assertRaises(InsufficientSpeechError) as caught:
            select_speech_rich_window(waveform, speech_activity=activity)
        self.assertLess(
            caught.exception.metadata.speech_in_selected_window_seconds,
            MIN_SPEECH_IN_SELECTED_WINDOW_SECONDS,
        )

    def test_non_finite_rejected(self) -> None:
        waveform = torch.zeros(TARGET_SAMPLE_RATE * 3)
        waveform[10] = float("nan")
        with self.assertRaises(AudioValidationError):
            prepare_audio_realtime(waveform, TARGET_SAMPLE_RATE)


if __name__ == "__main__":
    unittest.main()
