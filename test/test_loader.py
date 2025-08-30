import unittest
import json
import os
from loader import load_results_for_reflection

class TestLoader(unittest.TestCase):

    def setUp(self):
        self.test_file_path = 'test_data.jsonl'
        # Create a dummy data file for testing
        with open(self.test_file_path, 'w') as f:
            # Line 1: User History (can be empty for this test)
            f.write(json.dumps({"user_id": "test_user"}) + '\n')
            # Line 2: User Holdout (can be empty for this test)
            f.write(json.dumps({}) + '\n')
            # Line 3: Persona Data
            persona_data = {
                "user_id": "test_user",
                "runs": [{
                    "run_id": "test_run",
                    "persona": "test_persona",
                    "imitations": []
                }]
            }
            f.write(json.dumps(persona_data) + '\n')
            # Line 4: Evaluation Data
            eval_data = {
                "user_id": "test_user",
                "evaluations": [{
                    "run_id": "test_run",
                    "timestamp": "2023-01-01T00:00:00",
                    "evaluation_results": {
                        "overall": {
                            "bleu": {"score": 0.5},
                            "rouge": {"score": 0.6},
                            "new_metric_1": {"score": 0.7},
                            "new_metric_2": {"score": 0.8}
                        },
                        "best_predictions": [],
                        "worst_predictions": []
                    }
                }]
            }
            f.write(json.dumps(eval_data) + '\n')

    def tearDown(self):
        os.remove(self.test_file_path)

    def test_load_results_for_reflection_with_configurable_metrics(self):
        """
        Test that load_results_for_reflection correctly loads all metrics,
        including new configurable ones.
        """
        run_id = "test_run"
        results = load_results_for_reflection(run_id, self.test_file_path)

        self.assertIsNotNone(results)
        self.assertIn('bleu', results)
        self.assertIn('rouge', results)
        self.assertIn('new_metric_1', results)
        self.assertIn('new_metric_2', results)
        self.assertEqual(results['bleu']['score'], 0.5)
        self.assertEqual(results['rouge']['score'], 0.6)
        self.assertEqual(results['new_metric_1']['score'], 0.7)
        self.assertEqual(results['new_metric_2']['score'], 0.8)

if __name__ == '__main__':
    unittest.main()