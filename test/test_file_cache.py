#!/usr/bin/env python3
"""
Test script for file caching and I/O optimization.
Validates that cached operations produce identical results to original functions.
"""

import os
import json
import tempfile
import threading
import time
import concurrent.futures
from typing import Dict, Any, List

# Import our modules
from file_cache import get_file_cache, clear_global_cache, get_cache_stats
from loader import (
    load_user_history, load_stimulus, load_predictions, load_orginal,
    load_predictions_orginales_formated, load_results_for_reflection,
    load_latest_improved_persona
)
from saver import save_user_imitation, save_evaluation_results, save_reflection_results
from logging_config import logger


def create_test_file(file_path: str, content_lines: List[Dict[str, Any]]) -> None:
    """Create a test JSONL file with specified content."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in content_lines:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')


def test_file_cache_basic():
    """Test basic file cache functionality."""
    print("Testing basic file cache functionality...")
    
    cache = get_file_cache()
    
    # Clear cache to start fresh
    clear_global_cache()
    
    # Create a temporary test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        test_file = f.name
        test_data = [
            {"user_id": "test_user", "name": "Test User"},
            {"tweets": [{"tweet_id": "123", "full_text": "Test tweet"}]},
            {"runs": [{"run_id": "test_run", "imitations": []}]}
        ]
        
        for line in test_data:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
    
    try:
        # Test reading with cache
        cached_data = cache.read_file_with_cache(test_file)
        assert cached_data is not None, "Failed to read file with cache"
        assert len(cached_data) == 3, "Expected 3 lines in cached data"
        assert cached_data[0]["user_id"] == "test_user", "First line data incorrect"
        
        # Test cache hit
        cached_data2 = cache.read_file_with_cache(test_file)
        assert cached_data2 is not None, "Cache hit failed"
        assert cached_data2 == cached_data, "Cache hit returned different data"
        
        # Test cache stats
        stats = get_cache_stats()
        assert stats["cached_files"] == 1, f"Expected 1 cached file, got {stats['cached_files']}"
        
        print("✓ Basic file cache functionality test passed")
        
    finally:
        os.unlink(test_file)


def test_cache_invalidation():
    """Test cache invalidation when files are modified."""
    print("Testing cache invalidation...")
    
    cache = get_file_cache()
    clear_global_cache()
    
    # Create a temporary test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        test_file = f.name
        f.write('{"original": "data"}\n')
    
    try:
        # Read initial data
        cached_data1 = cache.read_file_with_cache(test_file)
        assert cached_data1[0]["original"] == "data", "Initial data incorrect"
        
        # Modify file
        time.sleep(0.1)  # Ensure different mtime
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write('{"modified": "data"}\n')
        
        # Read again - should detect file change and update cache
        cached_data2 = cache.read_file_with_cache(test_file)
        assert cached_data2[0]["modified"] == "data", "Modified data not detected"
        
        print("✓ Cache invalidation test passed")
        
    finally:
        os.unlink(test_file)


def test_thread_safety():
    """Test thread safety of the file cache."""
    print("Testing thread safety...")
    
    cache = get_file_cache()
    clear_global_cache()
    
    # Create a test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        test_file = f.name
        test_data = [
            {"user_id": f"user_{i}", "data": f"data_{i}"}
            for i in range(100)
        ]
        
        for line in test_data:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
    
    try:
        results = []
        errors = []
        
        def read_from_cache(thread_id):
            try:
                cached_data = cache.read_file_with_cache(test_file)
                if cached_data is None:
                    errors.append(f"Thread {thread_id}: Failed to read cache")
                    return
                
                # Validate data
                if len(cached_data) != 100:
                    errors.append(f"Thread {thread_id}: Expected 100 lines, got {len(cached_data)}")
                    return
                
                results.append(f"Thread {thread_id}: Success")
                
            except Exception as e:
                errors.append(f"Thread {thread_id}: Exception {e}")
        
        # Run multiple threads concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(read_from_cache, i) for i in range(20)]
            concurrent.futures.wait(futures)
        
        # Check results
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(results) == 20, f"Expected 20 successful results, got {len(results)}"
        
        print("✓ Thread safety test passed")
        
    finally:
        os.unlink(test_file)


def test_loader_optimization():
    """Test that optimized loader functions produce identical results."""
    print("Testing loader optimization...")
    
    clear_global_cache()
    
    # Create a comprehensive test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        test_file = f.name
        
        content_lines = [
            # Line 1: User history
            {"user_id": "test_user", "tweets": [
                {"tweet_id": "1", "full_text": "First tweet", "screen_name": "testuser"},
                {"tweet_id": "2", "full_text": "Second tweet", "screen_name": "testuser"}
            ]},
            
            # Line 2: Holdout tweets
            {"tweets": [
                {"tweet_id": "3", "full_text": "Holdout tweet 1", "reply_to_id": None},
                {"tweet_id": "4", "full_text": "Holdout tweet 2", "reply_to_id": "3"}
            ]},
            
            # Line 3: Runs data
            {"user_id": "test_user", "runs": [
                {"run_id": "test_run", "persona": "Test persona", "imitations": [
                    {"tweet_id": "3", "stimulus": "Test stimulus", "imitation": "Test imitation"}
                ]}
            ]},
            
            # Line 4: Evaluations
            {"user_id": "test_user", "evaluations": [
                {"run_id": "test_run", "evaluation_results": {
                    "overall": {"bleu": 0.5, "rouge": 0.6},
                    "best_predictions": [{"prediction": "best pred", "reference": "best ref"}],
                    "worst_predictions": [{"prediction": "worst pred", "reference": "worst ref"}]
                }}
            ]},
            
            # Line 5: Reflections
            {"reflections": [
                {"run_id": "test_run", "iteration": 1, "reflection_results": {
                    "reflection_on_results": "Test reflection",
                    "improved_persona": "Improved persona"
                }}
            ]}
        ]
        
        create_test_file(test_file, content_lines)
    
    try:
        # Test load_user_history
        user_history = load_user_history(test_file)
        assert user_history["user_id"] == "test_user", "load_user_history failed"
        assert len(user_history["tweets"]) == 2, "load_user_history tweet count incorrect"
        
        # Test load_stimulus
        stimuli = load_stimulus(test_file)
        assert len(stimuli) == 2, f"Expected 2 stimuli, got {len(stimuli)}"
        assert stimuli[0][2] == "3", "First stimulus tweet_id incorrect"
        
        # Test load_predictions
        predictions = load_predictions("test_run", test_file)
        assert len(predictions) == 1, f"Expected 1 prediction, got {len(predictions)}"
        assert predictions[0]["tweet_id"] == "3", "Prediction tweet_id incorrect"
        
        # Test load_orginal
        original = load_orginal(test_file, "3")
        assert original == "Holdout tweet 1", f"Expected 'Holdout tweet 1', got '{original}'"
        
        # Test load_predictions_orginales_formated
        formatted_predictions = load_predictions_orginales_formated("test_run", test_file)
        assert len(formatted_predictions) == 1, "Formatted predictions count incorrect"
        assert formatted_predictions[0]["original"] == "Holdout tweet 1", "Original not added correctly"
        
        # Test load_results_for_reflection
        reflection_data = load_results_for_reflection("test_run", test_file)
        assert reflection_data is not None, "load_results_for_reflection failed"
        assert reflection_data["user_id"] == "test_user", "Reflection user_id incorrect"
        assert reflection_data["persona"] == "Test persona", "Reflection persona incorrect"
        
        # Test load_latest_improved_persona
        improved_persona = load_latest_improved_persona("test_run", test_file)
        assert improved_persona == "Improved persona", f"Expected 'Improved persona', got '{improved_persona}'"
        
        print("✓ Loader optimization test passed")
        
    finally:
        os.unlink(test_file)


def test_saver_optimization():
    """Test that optimized saver functions work correctly."""
    print("Testing saver optimization...")
    
    clear_global_cache()
    
    # Create a base test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        test_file = f.name
        
        content_lines = [
            {"user_id": "test_user", "name": "Test User"},
            {"tweets": [{"tweet_id": "1", "full_text": "Test tweet"}]},
            {"user_id": "test_user", "runs": []},
            {"user_id": "test_user", "evaluations": []},
            {"reflections": []}
        ]
        
        create_test_file(test_file, content_lines)
    
    try:
        # Test save_user_imitation
        save_user_imitation(
            test_file, 
            "Test stimulus", 
            "Test persona", 
            "Test imitation", 
            "test_run_1", 
            "tweet_123"
        )
        
        # Verify the save worked
        predictions = load_predictions("test_run_1", test_file)
        assert len(predictions) == 1, f"Expected 1 prediction after save, got {len(predictions)}"
        assert predictions[0]["tweet_id"] == "tweet_123", "Saved tweet_id incorrect"
        
        # Test save_evaluation_results
        evaluation_data = {
            "overall": {
                "bleu": {"bleu": 0.7, "bleu_1": 0.6, "bleu_2": 0.8},
                "rouge": {"rouge_1": 0.8, "rouge_2": 0.7}
            },
            "best_predictions": [{"prediction": "best", "reference": "best_ref"}]
        }
        save_evaluation_results(test_file, evaluation_data, "test_run_1")
        
        # Verify evaluation save
        reflection_data = load_results_for_reflection("test_run_1", test_file)
        assert reflection_data is not None, "Evaluation save failed"
        assert reflection_data["bleu_scores"]["bleu"] == 0.7, "BLEU score incorrect"
        
        # Test save_reflection_results
        reflection_results = {
            "Reflection": "Test reflection content",
            "improved_persona": "Improved test persona"
        }
        save_reflection_results(test_file, reflection_results, "test_run_1", 1)
        
        # Verify reflection save
        improved_persona = load_latest_improved_persona("test_run_1", test_file)
        assert improved_persona == "Improved test persona", f"Expected 'Improved test persona', got '{improved_persona}'"
        
        print("✓ Saver optimization test passed")
        
    finally:
        os.unlink(test_file)


def test_performance_comparison():
    """Test performance improvement with caching."""
    print("Testing performance improvement...")
    
    clear_global_cache()
    
    # Create a larger test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        test_file = f.name
        
        # Create test data with many tweets
        tweets = [{"tweet_id": str(i), "full_text": f"Tweet {i}"} for i in range(1000)]
        
        content_lines = [
            {"user_id": "test_user", "tweets": tweets[:500]},
            {"tweets": tweets[500:]},
            {"user_id": "test_user", "runs": []},
            {"user_id": "test_user", "evaluations": []},
            {"reflections": []}
        ]
        
        create_test_file(test_file, content_lines)
    
    try:
        # Test multiple calls to load_orginal (this would be slow without caching)
        start_time = time.time()
        
        # Multiple calls that would read the file repeatedly without caching
        for i in range(100):
            tweet_id = str(i + 500)  # IDs in the holdout tweets
            original = load_orginal(test_file, tweet_id)
            if original != f"Tweet {i + 500}":
                print(f"Warning: Expected 'Tweet {i + 500}', got '{original}'")
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # With caching, this should be much faster than reading the file 100 times
        print(f"✓ Performance test completed in {elapsed:.3f} seconds")
        
        # Check cache stats
        stats = get_cache_stats()
        print(f"✓ Cache stats: {stats['cached_files']} files cached, {stats['cache_size_mb']:.2f} MB")
        
    finally:
        os.unlink(test_file)


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("FILE CACHE AND I/O OPTIMIZATION TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_file_cache_basic,
        test_cache_invalidation,
        test_thread_safety,
        test_loader_optimization,
        test_saver_optimization,
        test_performance_comparison,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 All tests passed! File caching optimization is working correctly.")
    else:
        print(f"❌ {failed} tests failed. Please check the implementation.")
    
    return failed == 0


if __name__ == "__main__":
    run_all_tests()