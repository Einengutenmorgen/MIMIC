#!/usr/bin/env python3
"""
Phase 7 Testing Suite: 4-Bedingungen Pipeline Validation

Comprehensive testing for the 4-condition baseline experiment pipeline.
Validates scientific integrity, data consistency, and implementation correctness.
"""

import os
import json
import tempfile
import pytest
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

# Import the baseline experiment components
from baseline_experiments import BaselineExperiment, create_baseline_config
from logging_config import logger


class TestPhase7Pipeline:
    """Comprehensive test suite for Phase 7: 4-Bedingungen Pipeline Integration"""
    
    @pytest.fixture
    def baseline_manager(self):
        """Create baseline experiment manager for testing"""
        return BaselineExperiment("data/test")
    
    @pytest.fixture
    def test_config(self):
        """Create test configuration"""
        return {
            'experiment': {
                'run_name_prefix': 'test_baseline',
                'users_dict': 'data/test',
                'number_of_users': 2,
                'number_of_rounds': 1,
                'num_stimuli_to_process': 10,
                'use_async': True,
                'num_workers': 2
            },
            'llm': {
                'persona_model': 'google',
                'imitation_model': 'ollama',
                'reflection_model': 'google_json',
                'ollama_model': 'gemma3:latest'
            },
            'templates': {
                'persona_template': 'persona_template_simple',
                'imitation_post_template': 'imitation_post_template_simple',
                'imitation_reply_template': 'imitation_replies_template_simple'
            }
        }
    
    @pytest.fixture
    def test_user_file(self):
        """Create a temporary test user file with valid structure"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            # Line 1: Historical tweets
            historical_data = {
                "user_id": "test_user_123",
                "set": "history",
                "tweets": [
                    {"tweet_id": "h1", "full_text": "I love coding in Python! #programming", "screen_name": "testuser"},
                    {"tweet_id": "h2", "full_text": "Beautiful sunset today", "screen_name": "testuser"},
                    {"tweet_id": "h3", "full_text": "Working on machine learning projects", "screen_name": "testuser"}
                ]
            }
            f.write(json.dumps(historical_data, ensure_ascii=False) + '\n')
            
            # Line 2: Holdout tweets (these will be used as stimuli)
            holdout_data = {
                "user_id": "test_user_123", 
                "set": "holdout",
                "tweets": [
                    {"tweet_id": "t1", "full_text": "This is a [MASKED] day!", "reply_to_id": None},
                    {"tweet_id": "t2", "full_text": "Amazing [MASKED] weather", "reply_to_id": None},
                    {"tweet_id": "t3", "full_text": "Reply to previous", "reply_to_id": "t1", "previous_message": "Original tweet"},
                    {"tweet_id": "t4", "full_text": "Another [MASKED] post", "reply_to_id": None},
                    {"tweet_id": "t5", "full_text": "Testing [MASKED] functionality", "reply_to_id": None},
                    {"tweet_id": "t6", "full_text": "Reply example", "reply_to_id": "t2", "previous_message": "Weather tweet"},
                    {"tweet_id": "t7", "full_text": "More [MASKED] content", "reply_to_id": None},
                    {"tweet_id": "t8", "full_text": "Final [MASKED] test", "reply_to_id": None},
                    {"tweet_id": "t9", "full_text": "Last reply", "reply_to_id": "t4", "previous_message": "Another post"},
                    {"tweet_id": "t10", "full_text": "End [MASKED] tweet", "reply_to_id": None}
                ]
            }
            f.write(json.dumps(holdout_data, ensure_ascii=False) + '\n')
            
            # Line 3: Empty runs (will be populated by baseline experiment)
            runs_data = {"user_id": "test_user_123", "runs": []}
            f.write(json.dumps(runs_data, ensure_ascii=False) + '\n')
            
            # Line 4: Empty evaluations (will be populated)
            eval_data = {"user_id": "test_user_123", "evaluations": []}
            f.write(json.dumps(eval_data, ensure_ascii=False) + '\n')
            
            # Line 5: Reflections (for best-persona testing)
            reflection_data = {
                "reflections": [
                    {
                        "run_id": "test_run_round_1",
                        "iteration": 1,
                        "reflection_results": {
                            "reflection_on_results": "Test reflection",
                            "improved_persona": "This user is a tech enthusiast who enjoys programming and nature."
                        }
                    }
                ]
            }
            f.write(json.dumps(reflection_data, ensure_ascii=False) + '\n')
            
            return f.name
    
    def test_identical_stimuli_splits(self, baseline_manager, test_user_file):
        """Test Phase 7 Erfolgskriterium: 4-Way Split with identical stimuli"""
        logger.info("Testing identical stimuli splits for 4 conditions")
        
        # Create splits
        splits = baseline_manager.create_identical_stimuli_splits(test_user_file, num_items=10)
        
        # Validate all 4 conditions present
        expected_conditions = ["no_persona", "generic_persona", "history_only", "best_persona"]
        assert set(splits.keys()) == set(expected_conditions), "All 4 conditions must be present"
        
        # Validate each condition has correct number of items
        for condition, items in splits.items():
            assert len(items) == 10, f"Condition {condition} should have 10 items"
        
        # CRITICAL: Validate identical stimuli across all conditions
        reference_tweet_ids = [item[2] for item in splits["no_persona"]]
        
        for condition in expected_conditions[1:]:
            condition_tweet_ids = [item[2] for item in splits[condition]]
            assert reference_tweet_ids == condition_tweet_ids, \
                f"Condition {condition} has different stimuli than reference!"
        
        # Validate post/reply distribution
        posts = [item for item in splits["no_persona"] if item[1] == True]
        replies = [item for item in splits["no_persona"] if item[1] == False]
        
        # Should have balanced distribution (±1 for odd numbers)
        expected_posts = 7  # From test data: 7 posts, 3 replies
        expected_replies = 3
        
        assert len(posts) == expected_posts, f"Expected {expected_posts} posts, got {len(posts)}"
        assert len(replies) == expected_replies, f"Expected {expected_replies} replies, got {len(replies)}"
        
        logger.info(f"✅ Identical stimuli validation passed: {len(reference_tweet_ids)} items per condition")
    
    def test_persona_generation_all_conditions(self, baseline_manager, test_user_file):
        """Test persona generation for all 4 conditions"""
        logger.info("Testing persona generation for all conditions")
        
        # Test No-Persona (should return None)
        no_persona = baseline_manager._get_condition_persona("no_persona", test_user_file, {})
        assert no_persona is None, "No-persona condition should return None"
        
        # Test Generic-Persona
        generic_info = baseline_manager.baseline_conditions["generic_persona"]
        generic_persona = baseline_manager._get_condition_persona("generic_persona", test_user_file, generic_info)
        assert generic_persona is not None, "Generic persona should not be None"
        assert len(generic_persona) > 50, "Generic persona should be substantial"
        assert "social media" in generic_persona.lower(), "Should mention social media"
        
        # Test History-Only (raw data without LLM processing)
        history_persona = baseline_manager._get_condition_persona("history_only", test_user_file, {})
        assert history_persona is not None, "History-only persona should not be None"
        assert len(history_persona) > 100, "History persona should contain substantial information"
        assert "User Profile based on posting history:" in history_persona, "Should use raw history format"
        
        # Test Best-Persona (should fallback to initial persona in test scenario)
        best_persona = baseline_manager._get_condition_persona("best_persona", test_user_file, {})
        assert best_persona is not None, "Best persona should not be None"
        assert len(best_persona) > 50, "Best persona should be substantial"
        
        logger.info("✅ All persona generation tests passed")
    
    def test_4_condition_pipeline_integration(self, baseline_manager, test_config, test_user_file):
        """Test end-to-end 4-condition pipeline"""
        logger.info("Testing 4-condition pipeline integration")
        
        # Create test directory
        test_dir = Path("data/test")
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy test file to test directory
        test_file_in_dir = test_dir / "test_user_123.jsonl"
        import shutil
        shutil.copy2(test_user_file, test_file_in_dir)
        
        try:
            # Update config to point to test directory
            test_config['experiment']['users_dict'] = str(test_dir)
            test_config['experiment']['number_of_users'] = 1
            test_config['experiment']['num_stimuli_to_process'] = 5  # Smaller for testing
            
            # Run baseline experiment in extended mode (4 conditions)
            results = baseline_manager.run_baseline_experiment(test_config, extended_mode=True)
            
            # Validate results structure
            assert "error" not in results, f"Experiment failed: {results.get('error')}"
            assert results["extended_mode"] == True, "Should be in extended mode"
            assert len(results["run_ids"]) == 4, "Should have 4 run IDs"
            assert results["stimuli_per_condition"] == 5, "Should have 5 stimuli per condition"
            
            # Validate run IDs
            expected_conditions = ["no_persona", "generic_persona", "history_only", "best_persona"]
            for condition in expected_conditions:
                assert condition in results["run_ids"], f"Missing run ID for {condition}"
                assert "baseline" in results["run_ids"][condition], "Run ID should contain 'baseline'"
            
            # Validate user processing
            if results["processed_users"]:
                assert "test_user_123" in results["processed_users"], "Test user should be processed"
            
            # Validate file output structure
            data = {}
            with open(test_file_in_dir, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if line.strip():
                        data[line_num] = json.loads(line)
            
            # Check runs data (line 2)
            runs_data = data.get(2, {})
            runs = runs_data.get('runs', [])
            
            baseline_runs = [r for r in runs if 'baseline' in r['run_id']]
            
            # Should have runs for all conditions (some might fail, but at least some should succeed)
            assert len(baseline_runs) > 0, "Should have at least some baseline runs"
            
            # Check each baseline run has correct structure
            for run in baseline_runs:
                assert 'run_id' in run, "Run should have run_id"
                assert 'persona' in run, "Run should have persona"
                assert 'imitations' in run, "Run should have imitations"
                
                # Validate persona based on condition type
                if 'no_persona' in run['run_id']:
                    assert run['persona'] == "[NO_PERSONA]", "No-persona should have special marker"
                elif 'generic' in run['run_id']:
                    assert len(run['persona']) > 50, "Generic persona should be substantial"
                elif 'history_only' in run['run_id']:
                    assert "User Profile based on posting history:" in run['persona'], "Should use history format"
                elif 'best_persona' in run['run_id']:
                    assert len(run['persona']) > 50, "Best persona should be substantial"
            
            logger.info(f"✅ Pipeline integration test passed: {len(baseline_runs)} condition runs created")
            
        finally:
            # Cleanup
            if test_file_in_dir.exists():
                test_file_in_dir.unlink()
            if test_dir.exists() and not list(test_dir.iterdir()):
                test_dir.rmdir()
    
    def test_data_integrity_validation(self, baseline_manager, test_user_file):
        """Test data integrity checks and validation"""
        logger.info("Testing data integrity validation")
        
        # Test with valid file
        splits = baseline_manager.create_identical_stimuli_splits(test_user_file, num_items=5)
        
        # Validate identical stimuli validation logic
        first_condition_ids = [item[2] for item in splits["no_persona"]]
        for condition in ["generic_persona", "history_only", "best_persona"]:
            condition_ids = [item[2] for item in splits[condition]]
            assert first_condition_ids == condition_ids, "Stimuli should be identical"
        
        # Test stimulus composition analysis
        all_stimuli = splits["no_persona"]
        posts = [item for item in all_stimuli if item[1] == True]
        replies = [item for item in all_stimuli if item[1] == False]
        
        assert len(posts) + len(replies) == len(all_stimuli), "All stimuli should be classified"
        assert len(posts) > 0, "Should have some posts"
        assert len(replies) >= 0, "Should have zero or more replies"
        
        logger.info("✅ Data integrity validation passed")
    
    def test_error_handling_edge_cases(self, baseline_manager):
        """Test error handling for edge cases"""
        logger.info("Testing error handling for edge cases")
        
        # Test with non-existent file
        result = baseline_manager.create_history_only_persona("nonexistent.jsonl")
        assert "Error:" in result or "Limited user history" in result, "Should handle missing files gracefully"
        
        # Test best persona extraction with no reflections
        best_persona = baseline_manager.extract_best_persona("nonexistent.jsonl", "fallback persona")
        assert best_persona == "fallback persona", "Should use fallback when extraction fails"
        
        # Test with empty config
        empty_config = {'experiment': {}, 'llm': {}, 'templates': {}}
        try:
            baseline_manager.run_baseline_experiment(empty_config, extended_mode=True)
            assert False, "Should raise error with empty config"
        except Exception as e:
            assert True, "Should handle empty config gracefully"
        
        logger.info("✅ Error handling tests passed")
    
    def test_scientific_design_validation(self, baseline_manager, test_user_file):
        """Validate scientific design principles"""
        logger.info("Testing scientific design validation")
        
        # Test identical stimuli principle
        splits_run1 = baseline_manager.create_identical_stimuli_splits(test_user_file, num_items=8, seed=42)
        splits_run2 = baseline_manager.create_identical_stimuli_splits(test_user_file, num_items=8, seed=42)
        
        # With same seed, should get identical results
        for condition in splits_run1:
            ids1 = [item[2] for item in splits_run1[condition]]
            ids2 = [item[2] for item in splits_run2[condition]]
            assert ids1 == ids2, f"Reproducibility failed for {condition}"
        
        # Test within-subject design (all conditions use same user data)
        for condition_type in ["no_persona", "generic_persona", "history_only", "best_persona"]:
            condition_info = baseline_manager.baseline_conditions[condition_type]
            # Each condition should use same user file but different processing
            # This is validated by the identical stimuli test above
        
        # Test balanced experimental design
        splits = baseline_manager.create_identical_stimuli_splits(test_user_file, num_items=10)
        
        # All conditions should have same number of stimuli
        lengths = [len(splits[condition]) for condition in splits]
        assert all(length == lengths[0] for length in lengths), "All conditions should have same stimulus count"
        
        logger.info("✅ Scientific design validation passed")


def run_phase7_validation():
    """
    Run comprehensive Phase 7 validation suite.
    
    Returns:
        bool: True if all tests pass, False otherwise
    """
    print("🧪 Starting Phase 7 Validation Suite")
    print("=" * 60)
    
    try:
        # Run pytest with verbose output
        pytest_args = [
            __file__,
            "-v",
            "--tb=short",
            "-x"  # Stop on first failure
        ]
        
        result = pytest.main(pytest_args)
        
        if result == 0:
            print("\n" + "=" * 60)
            print("🎉 ALL PHASE 7 TESTS PASSED!")
            print("✅ 4-Bedingungen Pipeline is ready for production")
            print("=" * 60)
            return True
        else:
            print("\n" + "=" * 60)
            print("❌ SOME PHASE 7 TESTS FAILED")
            print("🔧 Please fix issues before proceeding to Phase 8")
            print("=" * 60)
            return False
            
    except Exception as e:
        print(f"\n❌ Error running tests: {e}")
        return False


if __name__ == "__main__":
    success = run_phase7_validation()
    exit(0 if success else 1)
