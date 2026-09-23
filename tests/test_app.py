from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf
import torch

import app
from src.audio import (
    AudioLoadError,
    MultiSegmentMetadata,
    SpeechWindowMetadata,
    ValidSegmentMetadata,
    decode_audio,
)
from src.inference import MultiSegmentEmbedding, RealtimeEmbedding
from src.model import EMBEDDING_DIM
from ui.components import metadata_rows, multisegment_metadata_rows, render_result_card


def make_metadata() -> SpeechWindowMetadata:
    return SpeechWindowMetadata(
        original_duration_seconds=5.0,
        selected_start_seconds=1.0,
        selected_end_seconds=4.0,
        speech_in_selected_window_seconds=2.5,
        total_speech_seconds=3.5,
        sufficient_speech=True,
    )


def make_embedding() -> torch.Tensor:
    embedding = torch.zeros(EMBEDDING_DIM)
    embedding[0] = 1.0
    return embedding


def make_multisegment_metadata() -> MultiSegmentMetadata:
    return MultiSegmentMetadata(
        original_duration_seconds=8.5,
        total_speech_seconds=7.0,
        candidate_window_count=4,
        valid_window_count=3,
        window_seconds=3.0,
        hop_seconds=1.5,
        valid_segments=(
            ValidSegmentMetadata(0.0, 3.0, 2.8),
            ValidSegmentMetadata(1.5, 4.5, 2.7),
            ValidSegmentMetadata(4.5, 7.5, 2.6),
        ),
        sufficient_speech=True,
    )


class AppStateTests(unittest.TestCase):
    def test_initialization_preserves_existing_enrollment(self) -> None:
        existing = make_embedding()
        state = {"enrollment_embedding": existing, "enrollment_saved": True}
        app.initialize_session_state(state)
        self.assertIs(state["enrollment_embedding"], existing)
        self.assertTrue(state["enrollment_saved"])
        self.assertIn("latest_verification_result", state)

    def test_changing_model_invalidates_enrollment(self) -> None:
        state: dict[str, object] = {}
        app.initialize_session_state(state)
        app.activate_model(state, "first.pt")
        app.save_enrollment_state(
            state,
            RealtimeEmbedding(make_embedding(), make_metadata()),
            "first.pt",
            b"wav",
            "audio/wav",
        )

        invalidated = app.activate_model(state, "second.pt")
        self.assertTrue(invalidated)
        self.assertFalse(state["enrollment_saved"])
        self.assertIsNone(state["enrollment_embedding"])
        self.assertIsNone(state["latest_verification_result"])

    def test_same_model_preserves_enrollment_and_repeated_results(self) -> None:
        state: dict[str, object] = {}
        app.initialize_session_state(state)
        app.activate_model(state, "model.pt")
        app.save_enrollment_state(
            state,
            RealtimeEmbedding(make_embedding(), make_metadata()),
            "model.pt",
            b"wav",
            "audio/wav",
        )
        state["latest_verification_result"] = {"similarity": 1.0}

        self.assertFalse(app.activate_model(state, "model.pt"))
        self.assertTrue(app.enrollment_is_ready(state, "model.pt"))
        self.assertEqual(state["latest_verification_result"], {"similarity": 1.0})

    def test_enrollment_stores_aggregated_embedding_and_metadata(self) -> None:
        state: dict[str, object] = {}
        app.initialize_session_state(state)
        app.activate_model(state, "model.pt")
        aggregate = make_embedding()
        metadata = make_multisegment_metadata()

        app.save_enrollment_state(
            state,
            MultiSegmentEmbedding(aggregate, metadata),
            "model.pt",
            b"wav",
            "audio/wav",
        )

        self.assertTrue(torch.equal(state["enrollment_embedding"], aggregate))
        self.assertEqual(state["enrollment_metadata"], metadata)
        self.assertTrue(app.enrollment_is_ready(state, "model.pt"))
        self.assertTrue(app.multisegment_enrollment_is_ready(state, "model.pt"))

    def test_historical_enrollment_is_not_reused_by_multisegment_app(self) -> None:
        state: dict[str, object] = {}
        app.initialize_session_state(state)
        app.activate_model(state, "model.pt")
        app.save_enrollment_state(
            state,
            RealtimeEmbedding(make_embedding(), make_metadata()),
            "model.pt",
            b"wav",
            "audio/wav",
        )

        self.assertTrue(app.enrollment_is_ready(state, "model.pt"))
        self.assertFalse(app.multisegment_enrollment_is_ready(state, "model.pt"))

    def test_clear_removes_enrollment_audio_and_result(self) -> None:
        state: dict[str, object] = {}
        app.initialize_session_state(state)
        state.update(
            enrollment_saved=True,
            enrollment_embedding=make_embedding(),
            enrollment_audio_bytes=b"wav",
            latest_verification_result={"similarity": 1.0},
        )
        initial_version = state["enrollment_input_version"]
        app.clear_enrollment_state(state)
        self.assertFalse(state["enrollment_saved"])
        self.assertIsNone(state["enrollment_audio_bytes"])
        self.assertIsNone(state["latest_verification_result"])
        self.assertEqual(state["enrollment_input_version"], initial_version + 1)

    def test_verification_requires_matching_enrollment_model(self) -> None:
        state: dict[str, object] = {}
        app.initialize_session_state(state)
        self.assertFalse(app.enrollment_is_ready(state, "model.pt"))
        state.update(
            enrollment_saved=True,
            enrollment_embedding=make_embedding(),
            enrollment_model_path=str(Path("other.pt").resolve()),
        )
        self.assertFalse(app.enrollment_is_ready(state, "model.pt"))

    def test_verify_again_retains_enrollment_and_resets_input(self) -> None:
        state: dict[str, object] = {}
        app.initialize_session_state(state)
        state.update(
            enrollment_saved=True,
            enrollment_embedding=make_embedding(),
            latest_verification_result={"similarity": 1.0},
        )
        initial_version = state["verification_input_version"]

        app.reset_verification_state(state)

        self.assertTrue(state["enrollment_saved"])
        self.assertIsNotNone(state["enrollment_embedding"])
        self.assertIsNone(state["latest_verification_result"])
        self.assertEqual(state["verification_input_version"], initial_version + 1)


class AppHelperTests(unittest.TestCase):
    def test_multisegment_metadata_rows_use_real_backend_values(self) -> None:
        rows = dict(multisegment_metadata_rows(make_multisegment_metadata()))
        self.assertEqual(rows["Recording duration"], "8.50 s")
        self.assertEqual(rows["Detected speech"], "7.00 s")
        self.assertEqual(rows["Candidate windows"], "4")
        self.assertEqual(rows["Valid segments"], "3")
        self.assertEqual(rows["Window / hop"], "3.0 s / 1.5 s")

    def test_metadata_rows_only_show_vad_measurements_when_used(self) -> None:
        realtime_rows = dict(metadata_rows(make_metadata()))
        self.assertEqual(realtime_rows["Recording duration"], "5.00 s")
        self.assertEqual(realtime_rows["Detected speech"], "3.50 s")
        self.assertEqual(realtime_rows["Selected model segment"], "1.00–4.00 s")

        exact_window = SpeechWindowMetadata(
            original_duration_seconds=3.0,
            selected_start_seconds=0.0,
            selected_end_seconds=3.0,
            speech_in_selected_window_seconds=0.0,
            total_speech_seconds=0.0,
            sufficient_speech=True,
            vad_used=False,
        )
        exact_rows = dict(metadata_rows(exact_window))
        self.assertNotIn("Detected speech", exact_rows)
        self.assertNotIn("Speech in selected segment", exact_rows)

    def test_checkpoint_discovery_is_recursive_and_suffix_limited(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            (root / "nested").mkdir()
            (root / "a.pt").write_bytes(b"a")
            (root / "nested" / "b.PTH").write_bytes(b"b")
            (root / "ignore.txt").write_text("x", encoding="utf-8")
            found = app.discover_checkpoint_files(root)
        self.assertEqual([path.name for path in found], ["a.pt", "b.PTH"])

    def test_thesis_model_labels_and_ordering(self) -> None:
        paths = [
            Path("checkpoints/best_random.pt").resolve(),
            Path("checkpoints/extra.pt").resolve(),
            Path("checkpoints/best_adp.pt").resolve(),
            Path("checkpoints/best_raw.pt").resolve(),
        ]
        ordered = app._ordered_model_options(paths)
        self.assertEqual(
            [app._checkpoint_label(path) for path in ordered[:3]],
            ["RAW", "RANDOM", "ADAPTIVE"],
        )
        self.assertEqual(app._model_label(ordered[2]), "ADAPTIVE")

    def test_microphone_and_upload_bytes_share_one_realtime_method(self) -> None:
        verifier = Mock()
        seen_suffixes: list[str] = []

        def read_temporary_file(path: Path) -> RealtimeEmbedding:
            self.assertEqual(Path(path).read_bytes(), b"RIFF-test")
            seen_suffixes.append(Path(path).suffix)
            return RealtimeEmbedding(make_embedding(), make_metadata())

        verifier.extract_embedding_realtime.side_effect = read_temporary_file
        first = app.process_audio_bytes(verifier, b"RIFF-test")
        second = app.process_audio_bytes(verifier, b"RIFF-test", suffix=".mp3")
        self.assertEqual(verifier.extract_embedding_realtime.call_count, 2)
        self.assertEqual(seen_suffixes, [".wav", ".mp3"])
        self.assertTrue(torch.equal(first.embedding, second.embedding))

    def test_app_multisegment_audio_uses_explicit_multisegment_method(self) -> None:
        verifier = Mock()

        def inspect_temporary_file(path: Path) -> MultiSegmentEmbedding:
            self.assertEqual(Path(path).read_bytes(), b"RIFF-test")
            return MultiSegmentEmbedding(make_embedding(), make_multisegment_metadata())

        verifier.extract_embedding_multisegment.side_effect = inspect_temporary_file
        result = app.process_audio_bytes_multisegment(verifier, b"RIFF-test")
        verifier.extract_embedding_multisegment.assert_called_once()
        verifier.extract_embedding_realtime.assert_not_called()
        self.assertEqual(result.metadata.valid_window_count, 3)

    def test_app_scores_aggregates_without_threshold_or_decision(self) -> None:
        verification_embedding = torch.zeros(EMBEDDING_DIM)
        verification_embedding[0] = 0.8
        verification_embedding[1] = 0.6
        result = app.build_multisegment_verification_result(
            make_embedding(),
            MultiSegmentEmbedding(
                verification_embedding,
                make_multisegment_metadata(),
            ),
            "ADAPTIVE",
        )
        self.assertAlmostEqual(result["similarity"], 0.8)
        self.assertIsNone(result["threshold"])
        self.assertIsNone(result["same_speaker"])

    @patch("ui.components.st.html")
    def test_neutral_result_does_not_claim_same_or_different_speaker(self, html_mock) -> None:
        render_result_card(0.4281, None, None, "ADAPTIVE")
        rendered = "".join(call.args[0] for call in html_mock.call_args_list)
        self.assertIn("Speaker Similarity", rendered)
        self.assertIn("Not calibrated", rendered)
        self.assertIn("NOT AVAILABLE", rendered)
        self.assertNotIn("Identity Match", rendered)
        self.assertNotIn("Identity Not Matched", rendered)

    def test_empty_audio_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            app.process_audio_bytes(Mock(), b"")

    def test_current_soundfile_stack_decodes_real_mp3(self) -> None:
        self.assertTrue(app.mp3_decoder_available())
        with tempfile.TemporaryDirectory() as temp_name:
            path = Path(temp_name) / "fixture.mp3"
            sample_rate = 44_100
            time_axis = np.arange(sample_rate, dtype=np.float32) / sample_rate
            waveform = (0.05 * np.sin(2 * np.pi * 220 * time_axis)).astype(np.float32)
            sf.write(
                path,
                waveform,
                sample_rate,
                format="MP3",
                subtype="MPEG_LAYER_III",
            )
            decoded, decoded_rate = decode_audio(path)
        self.assertEqual(decoded_rate, sample_rate)
        self.assertEqual(decoded.shape[0], 1)
        self.assertTrue(torch.isfinite(decoded).all())

    @patch("app.mp3_decoder_available", return_value=True)
    def test_mp3_decode_failure_has_clean_message(self, _availability) -> None:
        verifier = Mock()
        verifier.extract_embedding_realtime.side_effect = AudioLoadError("decoder detail")
        with self.assertRaisesRegex(app.Mp3DecodeError, "Unable to decode this MP3"):
            app.process_audio_bytes(verifier, b"bad mp3", suffix=".mp3")

    @patch("app.mp3_decoder_available", return_value=False)
    def test_mp3_unavailable_does_not_disable_wav(self, _availability) -> None:
        verifier = Mock()
        verifier.extract_embedding_realtime.return_value = RealtimeEmbedding(
            make_embedding(),
            make_metadata(),
        )
        result = app.process_audio_bytes(verifier, b"RIFF-test", suffix=".wav")
        self.assertEqual(result.metadata, make_metadata())

    @patch("app.mp3_decoder_available", return_value=True)
    @patch("app.st.file_uploader")
    @patch("app.st.radio", return_value="Upload audio")
    def test_uploader_accepts_wav_and_mp3(
        self,
        _radio,
        file_uploader,
        _availability,
    ) -> None:
        uploaded = Mock()
        uploaded.name = "sample.MP3"
        uploaded.type = "audio/mpeg"
        uploaded.getvalue.return_value = b"mp3 bytes"
        file_uploader.return_value = uploaded

        data, mime, suffix = app.render_audio_input("enrollment", 0)

        self.assertEqual(data, b"mp3 bytes")
        self.assertEqual(mime, "audio/mpeg")
        self.assertEqual(suffix, ".mp3")
        self.assertEqual(file_uploader.call_args.kwargs["type"], ["wav", "mp3"])

    @patch("app.SpeakerVerifier")
    def test_resource_cache_reuses_verifier(self, verifier_class) -> None:
        app.get_cached_verifier.clear()
        verifier_class.return_value = object()
        first = app.get_cached_verifier("model.pt", 10, 20)
        second = app.get_cached_verifier("model.pt", 10, 20)
        self.assertIs(first, second)
        verifier_class.assert_called_once_with("model.pt")
        app.get_cached_verifier.clear()


if __name__ == "__main__":
    unittest.main()
