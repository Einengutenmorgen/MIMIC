#!/usr/bin/env python3
"""
Baseline Experiments V2 Test Suite

Comprehensive tests for the 4-condition baseline experiment implementation.
Tests scientific validity, identical stimuli design, and dynamic persona generation.
"""

import unittest
import tempfile
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline_experiments import BaselineExperiment, create_baseline_config


class TestBaselineExperimentV2(unittest.TestCase):
    """Comprehensive test suite for V2 baseline experiments."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.baseline_exp = BaselineExperiment(self.temp_dir)
        
        # Create comprehensive test user file
        self.test_user_data = self._create_comprehensive_test_file()
        self.minimal_user_data = self._create_minimal_test_file()
    
    def _create_comprehensive_test_file(self) -> str:
        """Create test user file with rich data for all conditions."""
        user_file_path = os.path.join(self.temp_dir, "comprehensive_test_user.jsonl")
        
        # Line 1: Rich user history (for History-Only condition)
        history_tweets = []
        
        # Diverse tweet content for realistic history
        tweet_contents = [
            "Just finished reading an amazing book about artificial intelligence! The future is fascinating.",
            "Coffee shop vibes today. Working on some creative projects.",
            "Really disappointed with the new movie. The plot was predictable and the acting felt forced.",
            "Beautiful sunset tonight! Sometimes you need to stop and appreciate nature.",
            "Debugging code all day... finally found the issue in a single missing semicolon",
            "Anyone else think pizza is the perfect food? Never gets old.",
            "Excited to announce I'm starting a new job next month! New challenges ahead.",
            "Traffic was horrible today. Need better public transportation in this city.",
            "Weekend hiking trip was incredible! Fresh air and amazing views.",
            "Just discovered this amazing jazz album from the 70s. Music never goes out of style."
        ]
        
        for i, content in enumerate(tweet_contents):
            history_tweets.append({
                "tweet_id": f"hist_{i}",
                "full_text": content,
                "screen_name": "testuser",
                "reply_to_id": None,
                "created_at": f"2023-08-{10+i:02d} 12:00:00+00:00",
                "original_user_id": "test_user_123"
            })
        
        # Add replies for conversation patterns
        reply_contents = [
            "I totally agree! That's exactly what I was thinking.",
            "Thanks for sharing this. Really insightful perspective.",
            "Hmm, I see your point but I think there's another way to look at it.",
            "This is so true! Had the exact same experience last week.",
            "Great question! I think the answer depends on the context."
        ]
        
        for i, content in enumerate(reply_contents):
            history_tweets.append({
                "tweet_id": f"reply_hist_{i}",
                "full_text": content,
                "screen_name": "testuser",
                "reply_to_id": f"original_{i}",
                "previous_message": f"Original message {i} that prompted this thoughtful reply",
                "created_at": f"2023-08-{20+i:02d} 15:00:00+00:00",
                "original_user_id": "test_user_123"
            })
        
        history_data = {
            "user_id": "test_user_123",
            "set": "history",
            "tweets": history_tweets
        }
        
        # Line 2: Holdout data (sufficient for 4-way split testing)
        holdout_tweets = []
        
        # Create 20 holdout items (5 per condition for testing)
        post_contents = [
            "This is a [MASKED] post about technology trends",
            "Having a [MASKED] day at the beach with friends", 
            "The new restaurant downtown has [MASKED] food",
            "Working late tonight on this [MASKED] project",
            "Just watched a [MASKED] documentary about climate change",
            "My cat did something absolutely [MASKED] today",
            "The weather has been [MASKED] all week long",
            "Found a [MASKED] book at the local bookstore",
            "This coffee tastes [MASKED] this morning",
            "Planning a [MASKED] vacation for next summer"
        ]
        
        reply_prompts = [
            "Previous message: What do you think about remote work?\nUser replied:",
            "Previous message: Anyone seen the latest Marvel movie?\nUser replied:",
            "Previous message: Best pizza place in town?\nUser replied:",
            "Previous message: How was your weekend?\nUser replied:",
            "Previous message: Any book recommendations?\nUser replied:",
            "Previous message: Traffic is terrible today!\nUser replied:",
            "Previous message: Great concert last night!\nUser replied:",
            "Previous message: New coffee shop opened nearby\nUser replied:",
            "Previous message: Working from home today\nUser replied:",
            "Previous message: Beautiful sunset this evening\nUser replied:"
        ]
        
        # Add posts
        for i, content in enumerate(post_contents):
            holdout_tweets.append({
                "tweet_id": f"holdout_post_{i}",
                "full_text": content,
                "reply_to_id": None,
                "masked_text": content,
                "original_user_id": "test_user_123"
            })
        
        # Add replies
        for i, prompt in enumerate(reply_prompts):
            holdout_tweets.append({
                "tweet_id": f"holdout_reply_{i}",
                "full_text": f"This is my [MASKED] response to the question",
                "reply_to_id": f"original_question_{i}",
                "previous_message": prompt,
                "masked_text": f"This is my [MASKED] response to the question",
                "original_user_id": "test_user_123"
            })
        
        holdout_data = {
            "user_id": "test_user_123",
            "set": "holdout",
            "tweets": holdout_tweets
        }
        
        # Line 3: Runs (with some existing runs for Best-Persona testing)
        runs_data = {
            "user_id": "test_user_123",
            "runs": [
                {
                    "run_id": "20240101_120000_round_1",
                    "persona": "Initial test persona created by LLM for baseline comparison",
                    "imitations": []
                },
                {
                    "run_id": "20240101_120000_round_3",
                    "persona": "Improved test persona after iterative refinement",
                    "imitations": []
                }
            ]
        }
        
        # Line 4: Evaluations (for Best-Persona identification)
        evaluations_data = {
            "user_id": "test_user_123",
            "evaluations": [
                {
                    "run_id": "20240101_120000_round_1",
                    "timestamp": "2024-01-01T12:30:00",
                    "evaluation_results": {
                        "overall": {
                            "rouge": {"rouge1": 0.45, "rouge2": 0.30, "rougeL": 0.42},
                            "bleu": {"bleu": 0.35}
                        }
                    }
                },
                {
                    "run_id": "20240101_120000_round_3",
                    "timestamp": "2024-01-01T14:30:00", 
                    "evaluation_results": {
                        "overall": {
                            "rouge": {"rouge1": 0.52, "rouge2": 0.38, "rougeL": 0.48},
                            "bleu": {"bleu": 0.42}
                        }
                    }
                }
            ]
        }
        
        # Line 5: Reflections (for Best-Persona extraction)
        reflections_data = {
            "reflections": [
                {
                    "run_id": "20240101_120000_round_1",
                    "iteration": 1,
                    "reflection_results": {
                        "reflection_on_results": "Initial persona needs refinement...",
                        "improved_persona": "Refined test persona with better characteristics"
                    }
                },
                {
                    "run_id": "20240101_120000_round_3", 
                    "iteration": 3,
                    "reflection_results": {
                        "reflection_on_results": "Much better performance after iterations...",
                        "improved_persona": "Highly optimized test persona with excellent performance metrics"
                    }
                }
            ]
        }
        
        # Write JSONL file
        with open(user_file_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(history_data, ensure_ascii=False) + '\n')
            f.write(json.dumps(holdout_data, ensure_ascii=False) + '\n')
            f.write(json.dumps(runs_data, ensure_ascii=False) + '\n')
            f.write(json.dumps(evaluations_data, ensure_ascii=False) + '\n')
            f.write(json.dumps(reflections_data, ensure_ascii=False) + '\n')
        
        return user_file_path
    
    def _create_minimal_test_file(self) -> str:
        """Create minimal test file for edge case testing."""
        user_file_path = os.path.join(self.temp_dir, "minimal_test_user.jsonl")
        
        # Minimal data structure
        minimal_history = {
            "user_id": "minimal_user",
            "set": "history",
            "tweets": [
                {"tweet_id": "min_1", "full_text": "Short tweet", "reply_to_id": None}
            ]
        }
        
        minimal_holdout = {
            "user_id": "minimal_user", 
            "set": "holdout",
            "tweets": [
                {"tweet_id": "holdout_1", "full_text": "Test [MASKED]", "reply_to_id": None}
            ]
        }
        
        with open(user_file_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(minimal_history, ensure_ascii=False) + '\n')
            f.write(json.dumps(minimal_holdout, ensure_ascii=False) + '\n')
            f.write('{"user_id": "minimal_user", "runs": []}\n')
            f.write('{"user_id": "minimal_user", "evaluations": []}\n')
            f.write('{"reflections": []}\n')
        
        return user_file_path

    # ========== CORE V2 FUNCTIONALITY TESTS ==========
    
    def test_v2_baseline_conditions_definition(self):
        """Test that all 4 V2 baseline conditions are properly defined."""
        conditions = self.baseline_exp.baseline_conditions
        
        # Must have all 4 V2 conditions
        expected_conditions = ["no_persona", "generic_persona", "history_only", "best_persona"]
        for condition in expected_conditions:
            self.assertIn(condition, conditions, f"Missing baseline condition: {condition}")
        
        # Validate condition structures
        for condition_name, condition_info in conditions.items():
            self.assertIn("name", condition_info)
            self.assertIn("description", condition_info)
            self.assertIn("post_template", condition_info)
            self.assertIn("reply_template", condition_info)
            self.assertIn("persona", condition_info)
        
        # Validate dynamic conditions
        self.assertEqual(conditions["history_only"]["persona"], "DYNAMIC")
        self.assertEqual(conditions["best_persona"]["persona"], "DYNAMIC")
        
        # Validate static conditions
        self.assertIsNone(conditions["no_persona"]["persona"])
        self.assertIsInstance(conditions["generic_persona"]["persona"], str)
        self.assertGreater(len(conditions["generic_persona"]["persona"]), 50)
        
        print("V2 baseline conditions properly defined")
    
    def test_identical_stimuli_splits(self):
        """CRITICAL: Test identical stimuli across all conditions."""
        # Test with sufficient data
        splits = self.baseline_exp.create_identical_stimuli_splits(
            self.test_user_data, 
            num_items=16,  # Use 16 items (we have 20 total)
            seed=42
        )
        
        # Must have all 4 conditions
        expected_conditions = ["no_persona", "generic_persona", "history_only", "best_persona"]
        for condition in expected_conditions:
            self.assertIn(condition, splits)
            self.assertEqual(len(splits[condition]), 16, f"{condition} should have 16 items")
        
        # CRITICAL: Verify identical stimuli across ALL conditions
        first_condition_ids = [item[2] for item in splits[expected_conditions[0]]]  # tweet_ids
        
        for condition in expected_conditions[1:]:
            condition_ids = [item[2] for item in splits[condition]]
            self.assertEqual(
                first_condition_ids, condition_ids, 
                f"Stimuli not identical between {expected_conditions[0]} and {condition}"
            )
        
        # Verify posts/replies are balanced
        posts_count = sum(1 for item in splits[expected_conditions[0]] if item[1] == True)
        replies_count = sum(1 for item in splits[expected_conditions[0]] if item[1] == False)
        
        self.assertGreater(posts_count, 0, "Should have some posts")
        self.assertGreater(replies_count, 0, "Should have some replies")
        self.assertEqual(posts_count + replies_count, 16, "Total should equal requested items")
        
        print(f"IDENTICAL stimuli validated: {posts_count} posts, {replies_count} replies across 4 conditions")
    
    def test_identical_stimuli_edge_cases(self):
        """Test identical stimuli with edge cases."""
        # Test with insufficient data (only 1 item available)
        splits = self.baseline_exp.create_identical_stimuli_splits(
            self.minimal_user_data,
            num_items=10,  # Request more than available
            seed=42
        )
        
        # Should use all available items
        for condition in ["no_persona", "generic_persona", "history_only", "best_persona"]:
            self.assertEqual(len(splits[condition]), 1, "Should use all available items")
        
        # Should still be identical
        first_ids = [item[2] for item in splits["no_persona"]]
        for condition in ["generic_persona", "history_only", "best_persona"]:
            condition_ids = [item[2] for item in splits[condition]]
            self.assertEqual(first_ids, condition_ids, "Should still be identical with insufficient data")
        
        print("Edge case handling for identical stimuli working")

    # ========== DYNAMIC PERSONA GENERATION TESTS ==========
    
    def test_history_only_persona_creation(self):
        """Test History-Only persona creation - NO LLM abstraction."""
        history_persona = self.baseline_exp.create_history_only_persona(self.test_user_data)
        
        # Should be substantial
        self.assertGreater(len(history_persona), 200, "History-Only persona should be substantial")
        
        # Should contain formatting structure
        self.assertIn("Based on this user's posting history:", history_persona)
        self.assertIn("authentic voice and communication patterns", history_persona)
        
        # Should contain ACTUAL tweet content (not LLM summaries)
        self.assertIn("artificial intelligence", history_persona, "Should contain actual tweet content")
        self.assertIn("Coffee shop vibes", history_persona, "Should contain actual tweet content")
        self.assertIn("pizza is the perfect food", history_persona, "Should contain actual tweet content")
        
        # Should NOT contain LLM abstraction language
        abstraction_indicators = [
            "tends to", "often posts about", "typically expresses", 
            "personality traits", "communication style suggests"
        ]
        for indicator in abstraction_indicators:
            self.assertNotIn(indicator.lower(), history_persona.lower(), 
                           f"History-Only should not contain LLM abstraction: '{indicator}'")
        
        print(f"History-Only persona created: {len(history_persona)} characters")
    
    def test_best_persona_extraction(self):
        """Test Best-Persona extraction from iterative rounds."""
        # Test with data that has evaluations
        best_persona = self.baseline_exp.extract_best_persona(self.test_user_data)
        
        self.assertIsNotNone(best_persona)
        self.assertGreater(len(best_persona), 50, "Best persona should be substantial")
        
        # Should be the optimized persona (highest scoring round)
        self.assertIn("optimized", best_persona.lower(), "Should be the optimized version")
        
        # Test best round identification
        best_run_id = self.baseline_exp.identify_best_round(self.test_user_data)
        self.assertEqual(best_run_id, "20240101_120000_round_3", "Should identify round 3 as best")
        
        print(f"Best-Persona extracted: {len(best_persona)} characters from {best_run_id}")
    
    def test_best_persona_fallback(self):
        """Test Best-Persona fallback when no iterative data available."""
        # Test with minimal data (no evaluations)
        best_persona = self.baseline_exp.extract_best_persona(self.minimal_user_data)
        
        self.assertIsNotNone(best_persona)
        self.assertGreater(len(best_persona), 30, "Fallback persona should be substantial")
        
        # Should indicate fallback was used
        fallback_indicators = ["social media user", "Generic", "Error:"]
        has_fallback = any(indicator in best_persona for indicator in fallback_indicators)
        self.assertTrue(has_fallback, "Should use fallback persona when no iterative data")
        
        print("Best-Persona fallback working correctly")
    
    def test_condition_persona_generation(self):
        """Test persona generation for all conditions."""
        condition_info = self.baseline_exp.baseline_conditions
        
        # Test all 4 conditions
        test_cases = [
            ("no_persona", None),
            ("generic_persona", str),
            ("history_only", str), 
            ("best_persona", str)
        ]
        
        for condition_type, expected_type in test_cases:
            persona = self.baseline_exp._get_condition_persona(
                condition_type, self.test_user_data, condition_info[condition_type]
            )
            
            if expected_type is None:
                self.assertIsNone(persona, f"{condition_type} should return None")
            else:
                self.assertIsInstance(persona, expected_type, f"{condition_type} should return {expected_type}")
                if condition_type != "no_persona":
                    self.assertGreater(len(persona), 50, f"{condition_type} persona should be substantial")
        
        print("Condition persona generation working for all 4 conditions")

    # ========== SCIENTIFIC VALIDITY TESTS ==========
    
    def test_scientific_validity_abstraction_effect(self):
        """Test scientific validity: History-Only vs Individual persona comparison."""
        
        # Create both persona types
        history_persona = self.baseline_exp.create_history_only_persona(self.test_user_data)
        individual_persona = self.baseline_exp.create_initial_persona(self.test_user_data)
        
        # Both should be substantial
        self.assertGreater(len(history_persona), 100)
        self.assertGreater(len(individual_persona), 50)
        
        # History-Only should contain raw data
        self.assertIn("artificial intelligence", history_persona)
        self.assertIn("Coffee shop vibes", history_persona)
        
        # Individual should contain LLM abstraction (if LLM working)
        # Note: This test might need mocking for LLM calls
        self.assertNotEqual(history_persona, individual_persona, 
                          "History-Only and Individual personas should be different")
        
        print("Scientific validity confirmed: History-Only vs Individual differentiation")
    
    def test_within_subject_design_validation(self):
        """Test within-subject design: same user, all conditions."""
        splits = self.baseline_exp.create_identical_stimuli_splits(self.test_user_data, num_items=8)
        
        # Validate that design is truly within-subject
        user_id = "test_user_123"
        
        for condition in ["no_persona", "generic_persona", "history_only", "best_persona"]:
            # All conditions should be for the same user
            # (This is ensured by file structure, but validates design principle)
            condition_stimuli = splits[condition]
            self.assertEqual(len(condition_stimuli), 8, f"Each condition should have same number of items")
        
        # Validate identical stimuli (critical for within-subject design)
        baseline_tweet_ids = {item[2] for item in splits["no_persona"]}
        
        for condition in ["generic_persona", "history_only", "best_persona"]:
            condition_tweet_ids = {item[2] for item in splits[condition]}
            self.assertEqual(baseline_tweet_ids, condition_tweet_ids, 
                           f"Within-subject design violated: {condition} has different stimuli")
        
        print("Within-subject design validated: identical stimuli across all conditions")

    
    def test_error_handling_robustness(self):
        """Test error handling for various edge cases."""
        
        # Test with non-existent file
        result = self.baseline_exp.create_history_only_persona("nonexistent_file.jsonl")
        self.assertIsInstance(result, str)
        self.assertIn("Error:", result)
        
        # Test with empty file path
        empty_file = os.path.join(self.temp_dir, "empty.jsonl")
        with open(empty_file, 'w') as f:
            f.write('')
        
        result = self.baseline_exp.create_history_only_persona(empty_file)
        self.assertIsInstance(result, str)
        
        print("Error handling robustness confirmed")

    # ========== BACKWARD COMPATIBILITY TESTS ==========
    
    def test_v1_backward_compatibility(self):
        """Test that V2 maintains V1 functionality."""
        # Test 2-way split (V1 mode)
        v1_splits = self.baseline_exp.split_holdout_balanced(
            self.test_user_data, 
            seed=42, 
            num_conditions=2  # V1 mode
        )
        
        # Should have only V1 conditions
        self.assertEqual(set(v1_splits.keys()), {"no_persona", "generic_persona"})
        
        # Should have balanced splits
        total_items = len(v1_splits["no_persona"]) + len(v1_splits["generic_persona"])
        self.assertGreater(total_items, 10, "Should have substantial data for V1 mode")
        
        # Items should not overlap
        no_persona_ids = {item[2] for item in v1_splits["no_persona"]}
        generic_ids = {item[2] for item in v1_splits["generic_persona"]}
        overlap = no_persona_ids & generic_ids
        self.assertEqual(len(overlap), 0, "V1 conditions should not overlap")
        
        print("V1 backward compatibility maintained")


def run_comprehensive_validation():
    """Run comprehensive V2 baseline validation."""
    print("="*60)
    print("BASELINE EXPERIMENTS V2 - COMPREHENSIVE VALIDATION")
    print("="*60)
    
    # Run all tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBaselineExperimentV2)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*60)
    print("BASELINE V2 VALIDATION COMPLETE")
    print("="*60)
    
    if result.wasSuccessful():
        print("\nBASELINE V2 IMPLEMENTATION VALIDATED")
        
        print("\nScientific Design Confirmed:")
        print("- Identical stimuli across all conditions")
        print("- Within-subject experimental design")
        print("- 4-condition persona comparison")
        print("- Dynamic persona generation")
        print("- Robust error handling")
        print("- V1 backward compatibility")
        
        print("\nReady for Production:")
        print("- Run: python baseline_experiments.py --mode v2")
        print("- Dry run: python baseline_experiments.py --dry-run")
        print("- V1 mode: python baseline_experiments.py --mode v1")
        
        return True
    else:
        print(f"\nVALIDATION FAILED: {len(result.failures)} failures, {len(result.errors)} errors")
        
        if result.failures:
            print("\nFailures:")
            for test, failure in result.failures:
                print(f"- {test}: {failure.split('AssertionError:')[-1].strip()}")
        
        if result.errors:
            print("\nErrors:")
            for test, error in result.errors:
                print(f"- {test}: {error.split('Exception:')[-1].strip()}")
        
        return False


if __name__ == "__main__":
    success = run_comprehensive_validation()
    exit(0 if success else 1)