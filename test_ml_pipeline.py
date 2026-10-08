import unittest
import os
import json


class TestMLPipeline(unittest.TestCase):

    def test_model_file_exists(self):
        self.assertTrue(os.path.exists("student_result_model.pkl"))

    def test_metrics_file_exists(self):
        self.assertTrue(os.path.exists("metrics.json"))

    def test_accuracy_exists(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        self.assertIn("accuracy", metrics)

    def test_accuracy_is_valid(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        accuracy = metrics["accuracy"]

        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)


if __name__ == "__main__":
    unittest.main()
