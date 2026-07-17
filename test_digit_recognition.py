import unittest

import numpy as np

from preprocess import preprocess_canvas
from predict import predict_from_canvas


def preprocess_for_model(canvas):
    return preprocess_canvas(canvas).tolist()


class DigitRecognitionTests(unittest.TestCase):
    def test_preprocess_returns_expected_shape(self):
        canvas = [[0] * 28 for _ in range(28)]
        for row in range(10, 18):
            canvas[row][12] = 1

        features = preprocess_for_model(canvas)

        self.assertEqual(len(features), 784)
        self.assertGreaterEqual(features[0], 0.0)
        self.assertLessEqual(features[-1], 1.0)

    def test_predicts_from_canvas_grid(self):
        canvas = np.zeros((28, 28), dtype=float)
        for row in range(10, 18):
            canvas[row][12] = 1.0

        result = predict_from_canvas(canvas)

        self.assertIn("prediction", result)
        self.assertIn("confidence", result)
        self.assertGreaterEqual(result["prediction"], 0)
        self.assertLessEqual(result["prediction"], 9)
        self.assertGreaterEqual(result["confidence"], 0.0)

    def test_vertical_stroke_is_recognized_as_one(self):
        canvas = np.zeros((28, 28), dtype=float)
        for row in range(4, 24):
            canvas[row][14] = 1.0

        result = predict_from_canvas(canvas)

        self.assertEqual(result["prediction"], 1)


if __name__ == "__main__":
    unittest.main()
