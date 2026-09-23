from __future__ import annotations

import unittest
from pathlib import Path

import torch
from streamlit.testing.v1 import AppTest

from src.audio import MultiSegmentMetadata, ValidSegmentMetadata
from src.model import EMBEDDING_DIM


def embedding() -> torch.Tensor:
    value = torch.zeros(EMBEDDING_DIM)
    value[0] = 1.0
    return value


def metadata() -> MultiSegmentMetadata:
    return MultiSegmentMetadata(
        original_duration_seconds=8.0,
        total_speech_seconds=6.0,
        candidate_window_count=4,
        valid_window_count=3,
        window_seconds=3.0,
        hop_seconds=1.5,
        valid_segments=(
            ValidSegmentMetadata(0.0, 3.0, 2.5),
            ValidSegmentMetadata(1.5, 4.5, 2.5),
            ValidSegmentMetadata(3.0, 6.0, 2.5),
        ),
        sufficient_speech=True,
    )


class StreamlitRenderTests(unittest.TestCase):
    def test_initial_render_loads_model_and_guards_verification(self) -> None:
        app = AppTest.from_file("app.py", default_timeout=60).run()
        self.assertEqual(len(app.exception), 0)
        self.assertIn("Model ready", [item.value for item in app.success])
        self.assertIn(
            "Create a voice profile before verification.",
            [item.value for item in app.info],
        )
        self.assertEqual(app.selectbox[0].label, "Model")
        self.assertTrue(app.selectbox[0].value.endswith("best_adp.pt"))
        buttons = {button.label: button for button in app.button}
        self.assertFalse(buttons["CREATE VOICE PROFILE"].disabled)
        self.assertTrue(buttons["VERIFY SPEAKER"].disabled)

    def test_aggregated_enrollment_and_neutral_result_render(self) -> None:
        checkpoint = str(Path("checkpoints/best_adp.pt").resolve())
        app = AppTest.from_file("app.py", default_timeout=60)
        initial_state = {
            "active_model_path": checkpoint,
            "enrollment_embedding": embedding(),
            "enrollment_model_path": checkpoint,
            "enrollment_metadata": metadata(),
            "enrollment_audio_bytes": None,
            "enrollment_audio_mime": None,
            "enrollment_saved": True,
            "latest_verification_result": {
                "similarity": 0.4281,
                "threshold": None,
                "same_speaker": None,
                "model_label": "ADAPTIVE",
                "metadata": metadata(),
            },
            "enrollment_input_version": 0,
            "verification_input_version": 0,
        }
        for key, value in initial_state.items():
            app.session_state[key] = value
        app.run()

        self.assertEqual(len(app.exception), 0)
        buttons = {button.label: button for button in app.button}
        self.assertFalse(buttons["VERIFY SPEAKER"].disabled)
        self.assertFalse(buttons["Clear enrollment"].disabled)
        self.assertFalse(buttons["Verify again"].disabled)
        expander_labels = {expander.label for expander in app.expander}
        self.assertIn("Enrollment segment details", expander_labels)
        self.assertIn("Verification sample details", expander_labels)


if __name__ == "__main__":
    unittest.main()
