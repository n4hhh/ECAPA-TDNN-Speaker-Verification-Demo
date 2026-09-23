from __future__ import annotations

import unittest

from streamlit.testing.v1 import AppTest


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


if __name__ == "__main__":
    unittest.main()
