import unittest

from shadow_renderer import ShadowRenderer


class RendererRecallTests(unittest.TestCase):
    def test_accepts_legacy_landmarks_outside_unit_range(self):
        renderer = ShadowRenderer()
        legacy_landmarks = [
            {"x": 0.1, "y": 2.4},
            {"x": 0.2, "y": 2.6},
            {"x": 0.3, "y": 2.1},
            {"x": 0.4, "y": 2.3},
            {"x": 0.5, "y": 2.2},
            {"x": 0.6, "y": 2.0},
            {"x": 0.7, "y": 2.1},
            {"x": 0.8, "y": 2.2},
        ]
        self.assertTrue(renderer._is_valid_landmarks(legacy_landmarks))


if __name__ == "__main__":
    unittest.main()
