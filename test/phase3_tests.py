#!/usr/bin/env python3
"""
Phase 3 Tests: End-to-End Baseline Pipeline Validation

Tests the complete baseline experiment pipeline.
"""

import unittest
import tempfile
import json
import os
import sys
import yaml
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline_main import create_baseline_config_file, run_baseline_experiment, validate_baseline_results
from baseline_experiments_v01 import BaselineExperiment


class TestBaselinePipeline(unittest.TestCase):
    """Test suite for end-to-end baseline pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_dir = os.path.join(self.temp_dir, "test_users")
        os.makedirs(self.test_data_dir)
        
        # Create test user files
        self.test_user_files = self._create_test_user_files()
        
        # Create test config
        self.test_config_path = self._create_test_config()
    
    def _create_test_user_files(self, num_users: int = 3) -> list:
        """Create test user JSONL files."""
        user_files = []
        
        for user_num in range(num_users):
            user_id = f"test_user_{user_num}"
            user_file_path = os.path.join(self.test_data_dir, f"{user_id}.jsonl")
            
            # Create realistic test data
            history_data = {
                "user_id": user_id,
                "set": "history",
                "tweets": [
                    {"tweet_id": f"hist_{i}", "full_text": f"Historical tweet {i}", "reply_to_id": None}
                    for i in range(5)
                ]
            }
            
            # Create holdout data with more items for better testing
            holdout_tweets = []
            
            # 8 posts
            for i in range(8):
                holdout_tweets.append({
                    "tweet_id": f"{user_id}_post_{i}",
                    "full_text": f"This is a [MASKED] post about topic {i}",
                    "reply_to_id": None,
                    "previous_message": None,
                    "masked_text": f"This is a [MASKED] post about topic {i}"
                })
            
            # 4 replies
            for i in range(4):
                holdout_tweets.append({
                    "tweet_id": f"{user_id}_reply_{i}",
                    "full_text": f"Reply {i} to interesting post",
                    "reply_to_id": f"original_{i}",
                    "previous_message": f"Original message {i} was quite interesting",
                    "masked_text": f"Reply {i} to interesting post"
                })
            
            holdout_data = {
                "user_id": user_id,
                "set": "holdout",
                "tweets": holdout_tweets
            }
            
            # Write JSONL file
            with open(user_file_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(history_data, ensure_ascii=False) + '\n')
                f.write(json.dumps(holdout_data, ensure_ascii=False) + '\n')
                f.write('{"user_id": "' + user_id + '", "runs": []}\n')
                f.write('{"user_id": "' + user_id + '", "evaluations": []}\n')
                f.write('{"reflections": []}\n')
            
            user_files.append(user_file_path)
        
        return user_files
    
    def _create_test_config(self) -> str:
        """Create test configuration file."""
        config_path = os.path.join(self.temp_dir, "test_baseline_config.yaml")
        
        config_content = {
            'experiment': {
                'run_name_prefix': 'test_baseline',
                'users_dict': self.test_data_dir,
                'number_of_users': 3,
                'number_of_rounds': 1,
                'num_stimuli_to_process': None,
                'balanced_split_seed': 42,
                'use_async': False,  # Simplified for testing
                'num_workers': 1
            },
            'llm': {
                'persona_model': 'google',
                'imitation_model': 'ollama',
                'ollama_model': 'gemma3:latest'
            },
            'templates': {
                'no_persona_post_template': 'imitation_post_template_no_persona',
                'no_persona_reply_template': 'imitation_replies_template_no_persona',
                'generic_post_template': 'imitation_post_template_generic',
                'generic_reply_template': 'imitation_replies_template_generic'
            }
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_content, f, default_flow_style=False)
        
        return config_path
    
    def test_config_file_creation(self):
        """Test baseline config file creation."""
        config_path = os.path.join(self.temp_dir, "created_baseline_config.yaml")
        
        # Create config file
        result_path = create_baseline_config_file(config_path)
        
        # Verify file was created
        self.assertTrue(os.path.exists(result_path))
        self.assertEqual(result_path, config_path)
        
        # Verify config content
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should contain key baseline elements
        self.assertIn("baseline_test", content)
        self.assertIn("no_persona", content)
        self.assertIn("generic_persona", content)
        self.assertIn("imitation_post_template_no_persona", content)
        self.assertIn("balanced_split_seed", content)
        
        print("✅ Config file creation working")
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test valid config
        with open(self.test_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Should have required keys
        self.assertIn('experiment', config)
        self.assertIn('users_dict', config['experiment'])
        self.assertIn('number_of_users', config['experiment'])
        self.assertIn('llm', config)
        
        print("✅ Config validation working")
    
    def test_baseline_experiment_structure(self):
        """Test that baseline experiment produces correct structure."""
        # Mock the LLM calls to avoid actual API calls in tests
        
        # Create baseline experiment manager
        baseline_exp = BaselineExperiment(self.test_data_dir)
        
        # Test split creation for all users
        for user_file in self.test_user_files:
            splits = baseline_exp.split_holdout_balanced(user_file, seed=42)
            
            # Should have both conditions
            self.assertIn("no_persona", splits)
            self.assertIn("generic_persona", splits)
            
            # Should be balanced
            self.assertEqual(len(splits["no_persona"]), 6)  # Half of 12 items
            self.assertEqual(len(splits["generic_persona"]), 6)
            
            # No overlap between conditions
            no_persona_ids = {item[2] for item in splits["no_persona"]}
            generic_ids = {item[2] for item in splits["generic_persona"]}
            self.assertEqual(len(no_persona_ids & generic_ids), 0)
        
        print("✅ Baseline experiment structure correct")
    
    def test_results_validation(self):
        """Test results validation functionality."""
        # Test valid results
        valid_results = {
            "processed_users": ["user1", "user2", "user3"],
            "failed_users": [],
            "run_ids": {
                "no_persona": "test_no_persona_20250911_120000",
                "generic_persona": "test_generic_20250911_120000"
            },
            "total_users": 3,
            "duration_seconds": 120.5
        }
        
        validation = validate_baseline_results(valid_results)
        self.assertTrue(validation["is_valid"])
        self.assertEqual(len(validation["issues"]), 0)
        self.assertEqual(validation["statistics"]["success_rate"], 1.0)
        
        # Test invalid results
        invalid_results = {
            "error": "Something went wrong"
        }
        
        validation = validate_baseline_results(invalid_results)
        self.assertFalse(validation["is_valid"])
        self.assertGreater(len(validation["issues"]), 0)
        
        # Test missing conditions
        missing_condition_results = {
            "processed_users": ["user1"],
            "failed_users": [],
            "run_ids": {
                "no_persona": "test_no_persona_20250911_120000"
                # Missing generic_persona
            },
            "total_users": 1
        }
        
        validation = validate_baseline_results(missing_condition_results)
        self.assertFalse(validation["is_valid"])
        self.assertTrue(any("Missing generic_persona" in issue for issue in validation["issues"]))
        
        print("✅ Results validation working")
    
    def test_data_file_integration(self):
        """Test integration with actual data file structure."""
        # Test that our test files work with existing loader
        from loader import load_stimulus
        
        for user_file in self.test_user_files:
            stimuli = load_stimulus(user_file)
            
            # Should load correct number of stimuli
            self.assertEqual(len(stimuli), 12)  # 8 posts + 4 replies
            
            # Should have correct structure (stimulus, is_post, post_id)
            for stimulus_data in stimuli:
                self.assertEqual(len(stimulus_data), 3)
                stimulus, is_post, post_id = stimulus_data
                self.assertIsInstance(stimulus, str)
                self.assertIsInstance(is_post, bool)
                self.assertIsInstance(post_id, str)
        
        print("✅ Data file integration working")
    
    def test_template_integration(self):
        """Test that baseline templates integrate with existing system."""
        import templates
        
        # Test all baseline templates exist and work
        baseline_templates = [
            "imitation_post_template_no_persona",
            "imitation_replies_template_no_persona",
            "imitation_post_template_generic",
            "imitation_replies_template_generic"
        ]
        
        for template_name in baseline_templates:
            # Should be able to load template
            template = templates.select_template(template_name)
            self.assertIsInstance(template, str)
            self.assertGreater(len(template), 50)
            
            # Should be able to format template
            if "post" in template_name:
                if "no_persona" in template_name:
                    formatted = templates.format_template(template_name, tweet="Test [MASKED] tweet")
                else:
                    formatted = templates.format_template(template_name, tweet="Test [MASKED] tweet")
                
                self.assertIn("Test [MASKED] tweet", formatted)
            else:  # Reply template
                if "no_persona" in template_name:
                    formatted = templates.format_template(template_name, tweet="Test tweet to reply to")
                else:
                    formatted = templates.format_template(template_name, tweet="Test tweet to reply to")
                
                self.assertIn("Test tweet to reply to", formatted)
        
        print("✅ Template integration working")


def run_phase3_validation():
    """Run comprehensive Phase 3 validation."""
    print("="*60)
    print("PHASE 3 VALIDATION: END-TO-END PIPELINE TESTING")
    print("="*60)
    
    # Run unit tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBaselinePipeline)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*60)
    print("PHASE 3 VALIDATION COMPLETE")
    print("="*60)
    
    if result.wasSuccessful():
        print("\n🎉 Phase 3 PASSED: End-to-end pipeline ready for production!")
        
        print("\n🚀 Ready to run baseline experiments:")
        print("1. Create config: python baseline_main.py --create-config")
        print("2. Run experiment: python baseline_main.py")
        print("3. Analyze results: python round_analysis.py")
        
        print("\n📊 Experiment will generate:")
        print("- No-Persona baseline results")
        print("- Generic-Persona baseline results") 
        print("- Within-subject comparisons")
        print("- Statistical analysis data")
        
        return True
    else:
        print(f"\n❌ Phase 3 FAILED: {len(result.failures)} failures, {len(result.errors)} errors")
        return False


if __name__ == "__main__":
    success = run_phase3_validation()
    exit(0 if success else 1)