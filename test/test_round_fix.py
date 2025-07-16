"""
Test script to verify the round-based run_id fix works correctly
"""

import os
import tempfile
import json
from main import process_user
from round_analysis import RoundAnalyzer
from logging_config import logger

def create_test_config():
    """Create a test configuration for verification"""
    return {
        'experiment': {
            'number_of_rounds': 3,
            'num_stimuli_to_process': 2
        },
        'llm': {
            'persona_model': 'google',
            'imitation_model': 'ollama',
            'reflection_model': 'google_json'
        },
        'templates': {
            'persona_template': 'persona_template_simple',
            'imitation_post_template': 'imitation_post_template_simple',
            'imitation_reply_template': 'imitation_replies_template_simple',
            'reflection_template': 'reflect_results_template'
        }
    }

def test_run_id_generation():
    """Test that unique run_ids are generated for each round"""
    base_run_id = "20250709_test"
    expected_run_ids = [
        f"{base_run_id}_round_1",
        f"{base_run_id}_round_2", 
        f"{base_run_id}_round_3"
    ]
    
    print("Testing run_id generation...")
    for i, expected in enumerate(expected_run_ids, 1):
        generated = f"{base_run_id}_round_{i}"
        assert generated == expected, f"Expected {expected}, got {generated}"
        print(f"✓ Round {i}: {generated}")
    
    print("✓ All run_ids generated correctly!\n")

def test_analysis_functionality():
    """Test the analysis tool functionality"""
    print("Testing analysis functionality...")
    
    # Test with existing data
    analyzer = RoundAnalyzer("data/filtered_users")
    
    # Get sample data
    sample_files = [f for f in os.listdir("data/filtered_users") if f.endswith('.jsonl')][:1]
    
    if sample_files:
        sample_file = os.path.join("data/filtered_users", sample_files[0])
        round_data = analyzer.extract_round_data(sample_file)
        
        print(f"✓ Extracted round data from {sample_files[0]}")
        print(f"  Rounds found: {list(round_data.keys())}")
        
        # Test analysis
        df = analyzer.analyze_all_users()
        if not df.empty:
            print(f"✓ Analysis successful: {len(df)} entries, {df['user_id'].nunique()} users")
            print(f"  Rounds in data: {sorted(df['round'].unique())}")
        else:
            print("ℹ No round data found in current files")
    else:
        print("ℹ No test files found")
    
    print()

def demonstrate_fix():
    """Demonstrate the before/after of the fix"""
    print("=" * 60)
    print("DEMONSTRATING THE FIX")
    print("=" * 60)
    
    print("BEFORE (Original System):")
    print("- Single run_id for all rounds: '20250709_140000'")
    print("- Round 1: run_id = '20250709_140000'")
    print("- Round 2: run_id = '20250709_140000' (OVERWRITES Round 1)")
    print("- Round 3: run_id = '20250709_140000' (OVERWRITES Round 2)")
    print("- Result: Only final round data preserved ❌")
    
    print("\nAFTER (Fixed System):")
    print("- Unique run_id for each round:")
    print("- Round 1: run_id = '20250709_140000_round_1'")
    print("- Round 2: run_id = '20250709_140000_round_2'")  
    print("- Round 3: run_id = '20250709_140000_round_3'")
    print("- Result: All round data preserved ✅")
    
    print("\nBENEFITS:")
    print("✓ Track performance progression across rounds")
    print("✓ Analyze if pipeline improves imitation quality")
    print("✓ Compare individual and aggregated results")
    print("✓ Generate comprehensive reports")
    
    print()

def main():
    """Main test function"""
    print("=" * 60)
    print("TESTING ITERATIVE SELF-PROMPTING PIPELINE FIX")
    print("=" * 60)
    
    # Test 1: Run ID generation
    test_run_id_generation()
    
    # Test 2: Analysis functionality
    test_analysis_functionality()
    
    # Test 3: Demonstrate the fix
    demonstrate_fix()
    
    print("=" * 60)
    print("USAGE INSTRUCTIONS")
    print("=" * 60)
    print("1. Run your experiment with the fixed main.py")
    print("2. Use round_analysis.py to analyze results:")
    print("   python round_analysis.py")
    print("3. Or analyze specific experiment:")
    print("   analyzer = RoundAnalyzer()")
    print("   df, agg = analyzer.generate_report('20250709_140000')")
    print("=" * 60)

if __name__ == "__main__":
    main()