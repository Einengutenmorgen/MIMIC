#!/usr/bin/env python3
"""
Integration test to verify the fixed evaluate_with_individual_scores function.
"""

# Mock data for testing
test_data = [
    {'imitation': 'This is a great tweet!', 'original': 'This is an amazing tweet!', 'tweet_id': 'tweet_1'},
    {'imitation': 'Bad tweet here', 'original': 'This is an amazing tweet!', 'tweet_id': 'tweet_2'},
    {'imitation': 'Another decent tweet', 'original': 'This is an amazing tweet!', 'tweet_id': 'tweet_3'}
]

def test_with_real_function():
    """Test the actual function with 3 items to verify the fix."""
    # Import the function
    from eval import evaluate_with_individual_scores
    
    result = evaluate_with_individual_scores(test_data)
    
    print("Testing with 3 imitations:")
    print("=" * 40)
    
    best_predictions = result['best_predictions']
    worst_predictions = result['worst_predictions']
    
    print(f"Best predictions count: {len(best_predictions)}")
    print(f"Worst predictions count: {len(worst_predictions)}")
    
    # Extract indices for overlap check
    best_indices = [pred['index'] for pred in best_predictions]
    worst_indices = [pred['index'] for pred in worst_predictions]
    
    print(f"Best indices: {best_indices}")
    print(f"Worst indices: {worst_indices}")
    
    # Check for overlap
    overlap = set(best_indices) & set(worst_indices)
    print(f"Overlap: {overlap}")
    
    if len(overlap) == 0:
        print("✅ SUCCESS: No overlap between best and worst predictions!")
    else:
        print("❌ FAILURE: Overlap detected!")
        
    print("\nDetailed results:")
    print("Best predictions:")
    for i, pred in enumerate(best_predictions):
        print(f"  {i+1}. Index {pred['index']}: {pred['prediction'][:50]}... (score: {pred['combined_score']:.4f})")
    
    print("Worst predictions:")
    for i, pred in enumerate(worst_predictions):
        print(f"  {i+1}. Index {pred['index']}: {pred['prediction'][:50]}... (score: {pred['combined_score']:.4f})")

if __name__ == "__main__":
    test_with_real_function()