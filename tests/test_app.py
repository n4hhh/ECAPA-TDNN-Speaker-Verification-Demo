from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf
import torch

import app
from src.audio import AudioLoadError, SpeechWindowMetadata, decode_audio
from src.inference import RealtimeEmbedding
from src.model import EMBEDDING_DIM


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


class AppHelperTests(unittest.TestCase):
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
