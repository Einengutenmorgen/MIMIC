import unittest
from unittest.mock import patch
import pandas as pd

class TestConfigurableMetrics(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.sample_data = pd.DataFrame({
            'llm_response': ["The quick brown fox", "jumps over the lazy dog"],
            'ground_truth': ["The quick brown fox", "jumps over the lazy dog"],
        })

    @patch('eval.yaml.safe_load')
    def test_all_metrics_enabled(self, mock_safe_load):
        # Import the function locally to ensure the patch is active
        from eval import evaluate_with_individual_scores

        # Mock config with all metrics enabled
        mock_safe_load.return_value = {
            'metrics': {
                'bleu': True,
                'rouge': True,
                'bert_score': True,
                'gpt_eval': True,
                'perplexity': True
            }
        }
        
        # Call the function
        result = evaluate_with_individual_scores(self.sample_data, config=mock_safe_load.return_value)
        
        # Assert that all metric keys are present in the individual scores
        for score in result['individual_scores']:
            self.assertIn('bleu', score)
            self.assertIn('rouge', score)
            self.assertIn('bertscore', score)
            self.assertIn('llm_evaluation', score)
            self.assertIn('perplexity', score)

    @patch('eval.yaml.safe_load')
    def test_subset_of_metrics_enabled(self, mock_safe_load):
        # Import the function locally
        from eval import evaluate_with_individual_scores

        # Mock config with a subset of metrics enabled
        mock_safe_load.return_value = {
            'metrics': {
                'bleu': True,
                'rouge': False,
                'bert_score': True,
                'gpt_eval': True,
                'perplexity': False
            }
        }
        
        # Call the function
        result = evaluate_with_individual_scores(self.sample_data, config=mock_safe_load.return_value)
        
        # Assert that only the enabled metric keys are present
        for score in result['individual_scores']:
            self.assertIn('bleu', score)
            self.assertNotIn('rouge', score)
            self.assertIn('bertscore', score)
            self.assertIn('llm_evaluation', score)
            self.assertNotIn('perplexity', score)

    @patch('eval.yaml.safe_load')
    def test_all_metrics_disabled(self, mock_safe_load):
        # Import the function locally
        from eval import evaluate_with_individual_scores

        # Mock config with all metrics disabled
        mock_safe_load.return_value = {
            'metrics': {
                'bleu': False,
                'rouge': False,
                'bert_score': False,
                'gpt_eval': False,
                'perplexity': False
            }
        }
        
        # Call the function
        result = evaluate_with_individual_scores(self.sample_data, config=mock_safe_load.return_value)
        
        # Assert that no metric keys are present
        for score in result['individual_scores']:
            self.assertNotIn('bleu', score)
            self.assertNotIn('rouge', score)
            self.assertNotIn('bertscore', score)
            self.assertNotIn('llm_evaluation', score)
            self.assertNotIn('perplexity', score)

if __name__ == '__main__':
    unittest.main()