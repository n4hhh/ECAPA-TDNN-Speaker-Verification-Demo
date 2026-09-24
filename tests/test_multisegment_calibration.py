from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.calibrate_multisegment_thresholds import (
    CHECKPOINTS,
    DatasetAudit,
    filename_matches_parent,
    recording_from_path,
    run_calibration,
)
from scripts.multisegment_calibration import (
    CALIBRATION_SAMPLES,
    Recording,
    center_crop_10_seconds,
    empirical_eer_operating_point,
    make_trials,
)
from src.audio import (
    RecordingTooLongError,
    select_multisegment_windows,
)
from src.vad import SpeechActivity, VAD_FRAME_DURATION_MS


class CalibrationCropTests(unittest.TestCase):
    def test_crop_has_exact_shape_and_is_deterministic_for_odd_length(self) -> None:
        waveform = torch.arange(CALIBRATION_SAMPLES + 5, dtype=torch.float32)
        first, start, end = center_crop_10_seconds(waveform)
        second, second_start, second_end = center_crop_10_seconds(waveform)
        self.assertEqual((start, end), (2, CALIBRATION_SAMPLES + 2))
        self.assertEqual((second_start, second_end), (start, end))
        self.assertEqual(first.shape, (160_000,))
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first, waveform[start:end]))

    def test_source_longer_than_twenty_seconds_can_be_cropped_then_prepared(self) -> None:
        waveform = torch.ones(16_000 * 22, dtype=torch.float32) * 0.1
        with self.assertRaises(RecordingTooLongError):
            select_multisegment_windows(waveform)

        crop, _, _ = center_crop_10_seconds(waveform)
        frame_samples = 16_000 * VAD_FRAME_DURATION_MS // 1_000
        activity = SpeechActivity(
            sample_rate=16_000,
            frame_duration_ms=VAD_FRAME_DURATION_MS,
            frame_samples=frame_samples,
            num_samples=crop.numel(),
            frame_is_speech=(True,) * (crop.numel() // frame_samples),
        )
        prepared = select_multisegment_windows(crop, speech_activity=activity)
        self.assertEqual(prepared.windows.shape, (5, 48_000))
        self.assertEqual(prepared.metadata.candidate_window_count, 5)

    def test_short_source_is_rejected_without_padding(self) -> None:
        with self.assertRaisesRegex(ValueError, "shorter"):
            center_crop_10_seconds(torch.zeros(CALIBRATION_SAMPLES - 1))


class CalibrationTrialTests(unittest.TestCase):
    @staticmethod
    def recordings() -> list[Recording]:
        return [
            Recording(
                f"speaker{speaker}",
                f"speaker{speaker}{index}",
                f"audio_val/speaker{speaker}/speaker{speaker}{index}.wav",
            )
            for speaker in range(9)
            for index in range(1, 4)
        ]

    def test_parent_directory_is_authoritative_and_names_are_audited(self) -> None:
        valid = Path("audio_val/bomman/bomman2.wav")
        invalid = Path("audio_val/Bomman/bomman2.wav")
        self.assertEqual(recording_from_path(valid).speaker_id, "bomman")
        self.assertEqual(recording_from_path(invalid).speaker_id, "Bomman")
        self.assertTrue(filename_matches_parent(valid))
        self.assertFalse(filename_matches_parent(invalid))

    def test_all_unique_unordered_pairs_have_expected_counts_and_order(self) -> None:
        recordings = self.recordings()
        first = make_trials(recordings)
        second = make_trials(list(reversed(recordings)))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 351)
        self.assertEqual(sum(trial.label == 1 for trial in first), 27)
        self.assertEqual(sum(trial.label == 0 for trial in first), 324)
        self.assertEqual(len({trial.trial_id for trial in first}), 351)
        pairs = [frozenset((trial.path_a, trial.path_b)) for trial in first]
        self.assertEqual(len(set(pairs)), 351)
        self.assertTrue(all(trial.path_a != trial.path_b for trial in first))

    def test_all_models_receive_one_identical_frozen_trial_sequence(self) -> None:
        recordings = tuple(self.recordings())
        audit = DatasetAudit(
            summary={
                "observed_speakers": 9,
                "observed_total_recordings": 27,
                "calibration_ready": True,
            },
            provenance=[],
            recordings=recordings,
            prepared_by_path={},
        )
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            checkpoints = {name: root / f"{name}.pt" for name in CHECKPOINTS}
            for path in checkpoints.values():
                path.write_bytes(b"fixture")
            consumed: list[tuple[str, tuple[str, ...], int]] = []

            def fake_score_model(name, _path, _audit, trials, _device, _output_dir):
                consumed.append((name, tuple(t.trial_id for t in trials), id(trials)))
                return {"model": name}

            with patch(
                "scripts.calibrate_multisegment_thresholds.audit_dataset",
                return_value=audit,
            ), patch.dict(
                "scripts.calibrate_multisegment_thresholds.CHECKPOINTS", checkpoints, clear=True
            ), patch(
                "scripts.calibrate_multisegment_thresholds._score_model",
                side_effect=fake_score_model,
            ):
                result = run_calibration(root / "audio_val", root / "reports")

            self.assertEqual(result, 0)
            self.assertEqual([name for name, _, _ in consumed], ["RAW", "RANDOM", "ADAPTIVE"])
            self.assertEqual(len({ids for _, ids, _ in consumed}), 1)
            self.assertEqual(len({identity for _, _, identity in consumed}), 1)
            with (root / "reports" / "trials.csv").open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 351)


class CalibrationOperatingPointTests(unittest.TestCase):
    def test_known_empirical_eer_and_inclusive_decision_boundary(self) -> None:
        labels = [1, 1, 0, 0]
        scores = [0.9, 0.4, 0.6, 0.2]
        point = empirical_eer_operating_point(labels, scores)
        self.assertEqual(point.eer_threshold, 0.6)
        self.assertEqual(point.eer, 0.5)
        self.assertEqual(point.far_at_operating_point, 0.5)
        self.assertEqual(point.frr_at_operating_point, 0.5)
        self.assertTrue(0.6 >= point.eer_threshold)
        self.assertFalse(math.nextafter(0.6, -math.inf) >= point.eer_threshold)
        self.assertEqual(point, empirical_eer_operating_point(labels, scores))

    def test_failed_audit_writes_provenance_but_prevents_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            (root / "audio_val" / "WrongCase").mkdir(parents=True)
            (root / "audio_val" / "WrongCase" / "wrong1.wav").write_bytes(b"invalid")
            output = root / "reports"
            with patch("scripts.calibrate_multisegment_thresholds._score_model") as scorer:
                exit_code = run_calibration(root / "audio_val", output)
            self.assertEqual(exit_code, 2)
            scorer.assert_not_called()
            self.assertTrue((output / "crop_provenance.csv").is_file())
            self.assertTrue((output / "dataset_audit.json").is_file())
            self.assertFalse((output / "trials.csv").exists())
            self.assertFalse((output / "calibration_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
