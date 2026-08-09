from __future__ import annotations

import unittest

from streamlit.testing.v1 import AppTest


class StreamlitRenderTests(unittest.TestCase):
    def test_initial_render_loads_model_and_guards_verification(self) -> None:
        app = AppTest.from_file("app.py", default_timeout=60).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.title[0].value, "Vietnamese Speaker Verification Demo")
        self.assertIn("Model ready", [item.value for item in app.success])
        self.assertIn(
            "Save an enrollment utterance before verification.",
            [item.value for item in app.info],
        )
        buttons = {button.label: button for button in app.button}
        self.assertTrue(buttons["VERIFY SPEAKER"].disabled)
        self.assertTrue(buttons["CLEAR ENROLLMENT"].disabled)


if __name__ == "__main__":
    unittest.main()
