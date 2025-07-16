"""
Utility functions for data loading and analysis framework.

This module provides helper functions for common data operations,
filtering, sorting, and aggregation tasks.

Author: Assistant
Date: 2025-07-09
"""

import os
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from collections import defaultdict, Counter
import re
from datetime import datetime
import numpy as np

from data_framework import (
    UserData, Tweet, Imitation, EvaluationResult, 
    RunEvaluation, Run, DataLoader, RunAnalyzer
)


def find_run_files(data_dir: str, run_id: str) -> List[str]:
    """
    Find all files containing a specific run ID.
    
    Args:
        data_dir: Directory containing JSONL files
        run_id: Run ID to search for
    
    Returns:
        List of file paths containing the run ID
    """
    loader = DataLoader()
    files_with_run = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        return files_with_run
    
    for file_path in data_path.glob("*.jsonl"):
        try:
            data = loader.load_jsonl_file(str(file_path))
            
            # Check runs data (line 2)
            if 2 in data and 'runs' in data[2]:
                for run in data[2]['runs']:
                    if run.get('run_id') == run_id:
                        files_with_run.append(str(file_path))
                        break
        except Exception as e:
            continue
    
    return files_with_run


def get_available_run_ids(data_dir: str) -> List[str]:
    """
    Get all available run IDs from the data directory.
    
    Args:
        data_dir: Directory containing JSONL files
    
    Returns:
        List of unique run IDs
    """
    loader = DataLoader()
    run_ids = set()
    data_path = Path(data_dir)
    
    if not data_path.exists():
        return list(run_ids)
    
    for file_path in data_path.glob("*.jsonl"):
        try:
            data = loader.load_jsonl_file(str(file_path))
            
            # Check runs data (line 2)
            if 2 in data and 'runs' in data[2]:
                for run in data[2]['runs']:
                    run_id = run.get('run_id')
                    if run_id:
                        run_ids.add(run_id)
        except Exception as e:
            continue
    
    return sorted(list(run_ids))


def get_user_files(data_dir: str) -> List[str]:
    """
    Get all user JSONL files from the data directory.
    
    Args:
        data_dir: Directory containing JSONL files
    
    Returns:
        List of user file paths
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        return []
    
    return [str(f) for f in data_path.glob("*.jsonl")]


def filter_tweets_by_date(tweets: List[Tweet], start_date: str = None, end_date: str = None) -> List[Tweet]:
    """
    Filter tweets by date range.
    
    Args:
        tweets: List of Tweet objects
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)
    
    Returns:
        Filtered list of tweets
    """
    if not start_date and not end_date:
        return tweets
    
    filtered_tweets = []
    
    for tweet in tweets:
        try:
            # Parse the tweet date (assuming format: "2023-08-09 22:59:56+00:00")
            tweet_date = datetime.strptime(tweet.created_at[:10], "%Y-%m-%d")
            
            if start_date:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                if tweet_date < start_dt:
                    continue
            
            if end_date:
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                if tweet_date > end_dt:
                    continue
            
            filtered_tweets.append(tweet)
        except (ValueError, IndexError):
            # Skip tweets with invalid date format
            continue
    
    return filtered_tweets


def filter_tweets_by_type(tweets: List[Tweet], tweet_type: str) -> List[Tweet]:
    """
    Filter tweets by type (posts, replies, retweets).
    
    Args:
        tweets: List of Tweet objects
        tweet_type: Type of tweet ('posts', 'replies', 'retweets')
    
    Returns:
        Filtered list of tweets
    """
    filtered_tweets = []
    
    for tweet in tweets:
        if tweet_type == 'posts':
            # Original posts (not replies or retweets)
            if not tweet.reply_to_id and not tweet.retweeted_user_id:
                filtered_tweets.append(tweet)
        elif tweet_type == 'replies':
            # Replies to other tweets
            if tweet.reply_to_id:
                filtered_tweets.append(tweet)
        elif tweet_type == 'retweets':
            # Retweets
            if tweet.retweeted_user_id:
                filtered_tweets.append(tweet)
    
    return filtered_tweets


def aggregate_scores_by_user(user_data: Dict[str, UserData], run_id: str) -> Dict[str, Dict[str, float]]:
    """
    Aggregate evaluation scores by user for a specific run.
    
    Args:
        user_data: Dictionary of user data
        run_id: Run ID to aggregate for
    
    Returns:
        Dictionary mapping user_id to aggregated scores
    """
    user_scores = {}
    
    for user_id, data in user_data.items():
        # Get evaluations for this run
        run_evaluations = [eval for eval in data.evaluations if eval.run_id == run_id]
        
        if run_evaluations:
            # Calculate average scores with proper handling of nested structures
            total_bleu = 0.0
            total_rouge_1 = 0.0
            total_rouge_2 = 0.0
            total_rouge_l = 0.0
            
            for eval in run_evaluations:
                # Handle nested score structures
                bleu_score = eval.overall_scores.get('bleu', 0.0)
                if isinstance(bleu_score, dict):
                    bleu_score = bleu_score.get('score', 0.0)
                
                rouge_1_score = eval.overall_scores.get('rouge_1', 0.0)
                if isinstance(rouge_1_score, dict):
                    rouge_1_score = rouge_1_score.get('score', 0.0)
                
                rouge_2_score = eval.overall_scores.get('rouge_2', 0.0)
                if isinstance(rouge_2_score, dict):
                    rouge_2_score = rouge_2_score.get('score', 0.0)
                
                rouge_l_score = eval.overall_scores.get('rouge_l', 0.0)
                if isinstance(rouge_l_score, dict):
                    rouge_l_score = rouge_l_score.get('score', 0.0)
                
                total_bleu += float(bleu_score)
                total_rouge_1 += float(rouge_1_score)
                total_rouge_2 += float(rouge_2_score)
                total_rouge_l += float(rouge_l_score)
            
            count = len(run_evaluations)
            
            user_scores[user_id] = {
                'bleu': total_bleu / count,
                'rouge_1': total_rouge_1 / count,
                'rouge_2': total_rouge_2 / count,
                'rouge_l': total_rouge_l / count,
                'combined': (total_bleu + total_rouge_1 + total_rouge_2 + total_rouge_l) / (4 * count),
                'evaluation_count': count
            }
    
    return user_scores


def sort_users_by_performance(user_scores: Dict[str, Dict[str, float]], 
                             metric: str = 'combined', 
                             reverse: bool = True) -> List[Tuple[str, float]]:
    """
    Sort users by performance metric.
    
    Args:
        user_scores: Dictionary of user scores
        metric: Metric to sort by ('bleu', 'rouge_1', 'rouge_2', 'rouge_l', 'combined')
        reverse: Sort in descending order if True
    
    Returns:
        List of (user_id, score) tuples sorted by performance
    """
    user_performance = []
    
    for user_id, scores in user_scores.items():
        if metric in scores:
            user_performance.append((user_id, scores[metric]))
    
    return sorted(user_performance, key=lambda x: x[1], reverse=reverse)


def get_top_bottom_performers(user_scores: Dict[str, Dict[str, float]], 
                             metric: str = 'combined', 
                             n: int = 5) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """
    Get top and bottom performing users.
    
    Args:
        user_scores: Dictionary of user scores
        metric: Metric to sort by
        n: Number of top/bottom performers to return
    
    Returns:
        Tuple of (top_performers, bottom_performers)
    """
    sorted_users = sort_users_by_performance(user_scores, metric, reverse=True)
    
    top_performers = sorted_users[:n]
    bottom_performers = sorted_users[-n:] if len(sorted_users) > n else []
    
    return top_performers, bottom_performers


def analyze_tweet_patterns(tweets: List[Tweet]) -> Dict[str, Any]:
    """
    Analyze patterns in tweet data.
    
    Args:
        tweets: List of Tweet objects
    
    Returns:
        Dictionary containing pattern analysis
    """
    if not tweets:
        return {}
    
    # Count tweet types
    posts = filter_tweets_by_type(tweets, 'posts')
    replies = filter_tweets_by_type(tweets, 'replies')
    retweets = filter_tweets_by_type(tweets, 'retweets')
    
    # Analyze text lengths
    text_lengths = [len(tweet.full_text) for tweet in tweets]
    
    # Analyze posting times (hours)
    posting_hours = []
    for tweet in tweets:
        try:
            dt = datetime.strptime(tweet.created_at, "%Y-%m-%d %H:%M:%S+00:00")
            posting_hours.append(dt.hour)
        except ValueError:
            continue
    
    # Common words analysis
    all_text = ' '.join(tweet.full_text for tweet in tweets)
    words = re.findall(r'\b\w+\b', all_text.lower())
    common_words = Counter(words).most_common(20)
    
    # Hashtag and mention analysis
    hashtags = re.findall(r'#\w+', all_text)
    mentions = re.findall(r'@\w+', all_text)
    
    return {
        'total_tweets': len(tweets),
        'posts': len(posts),
        'replies': len(replies),
        'retweets': len(retweets),
        'avg_text_length': np.mean(text_lengths) if text_lengths else 0,
        'median_text_length': np.median(text_lengths) if text_lengths else 0,
        'posting_hours_distribution': Counter(posting_hours),
        'common_words': common_words,
        'hashtag_count': len(hashtags),
        'mention_count': len(mentions),
        'unique_hashtags': len(set(hashtags)),
        'unique_mentions': len(set(mentions))
    }


def create_performance_summary(user_data: Dict[str, UserData], run_id: str) -> Dict[str, Any]:
    """
    Create a comprehensive performance summary for a run.
    
    Args:
        user_data: Dictionary of user data
        run_id: Run ID to analyze
    
    Returns:
        Dictionary containing performance summary
    """
    # Aggregate scores
    user_scores = aggregate_scores_by_user(user_data, run_id)
    
    if not user_scores:
        return {'error': 'No evaluation data found for this run'}
    
    # Get top and bottom performers
    top_performers, bottom_performers = get_top_bottom_performers(user_scores)
    
    # Calculate overall statistics
    all_scores = list(user_scores.values())
    
    summary = {
        'run_id': run_id,
        'total_users': len(user_scores),
        'top_performers': top_performers,
        'bottom_performers': bottom_performers,
        'overall_stats': {
            'avg_bleu': np.mean([s['bleu'] for s in all_scores]),
            'avg_rouge_1': np.mean([s['rouge_1'] for s in all_scores]),
            'avg_rouge_2': np.mean([s['rouge_2'] for s in all_scores]),
            'avg_rouge_l': np.mean([s['rouge_l'] for s in all_scores]),
            'avg_combined': np.mean([s['combined'] for s in all_scores]),
            'std_combined': np.std([s['combined'] for s in all_scores]),
            'total_evaluations': sum(s['evaluation_count'] for s in all_scores)
        },
        'score_distribution': {
            'combined_scores': [s['combined'] for s in all_scores],
            'bleu_scores': [s['bleu'] for s in all_scores],
            'rouge_1_scores': [s['rouge_1'] for s in all_scores]
        }
    }
    
    return summary


def export_to_csv(user_data: Dict[str, UserData], run_id: str, output_path: str) -> None:
    """
    Export user performance data to CSV.
    
    Args:
        user_data: Dictionary of user data
        run_id: Run ID to export
        output_path: Path for output CSV file
    """
    user_scores = aggregate_scores_by_user(user_data, run_id)
    
    # Convert to DataFrame
    df_data = []
    for user_id, scores in user_scores.items():
        row = {'user_id': user_id}
        row.update(scores)
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    df.to_csv(output_path, index=False)


def validate_data_integrity(user_data: UserData) -> Dict[str, Any]:
    """
    Validate data integrity for a user.
    
    Args:
        user_data: UserData object to validate
    
    Returns:
        Dictionary containing validation results
    """
    issues = []
    
    # Check for missing user_id
    if not user_data.user_id:
        issues.append("Missing user_id")
    
    # Check for empty tweet lists
    if not user_data.historical_tweets:
        issues.append("No historical tweets")
    
    if not user_data.holdout_tweets:
        issues.append("No holdout tweets")
    
    # Check for runs without evaluations
    run_ids = {run.run_id for run in user_data.runs}
    eval_run_ids = {eval.run_id for eval in user_data.evaluations}
    
    runs_without_evals = run_ids - eval_run_ids
    if runs_without_evals:
        issues.append(f"Runs without evaluations: {runs_without_evals}")
    
    # Check for duplicate tweet IDs
    all_tweet_ids = []
    for tweet in user_data.historical_tweets + user_data.holdout_tweets:
        all_tweet_ids.append(tweet.tweet_id)
    
    duplicate_tweets = len(all_tweet_ids) - len(set(all_tweet_ids))
    if duplicate_tweets > 0:
        issues.append(f"Duplicate tweet IDs: {duplicate_tweets}")
    
    # Check for empty personas
    empty_personas = sum(1 for run in user_data.runs if not run.persona.strip())
    if empty_personas > 0:
        issues.append(f"Empty personas: {empty_personas}")
    
    return {
        'user_id': user_data.user_id,
        'is_valid': len(issues) == 0,
        'issues': issues,
        'historical_tweets': len(user_data.historical_tweets),
        'holdout_tweets': len(user_data.holdout_tweets),
        'runs': len(user_data.runs),
        'evaluations': len(user_data.evaluations),
        'reflections': len(user_data.reflections)
    }


def generate_data_report(data_dir: str, run_id: str = None) -> Dict[str, Any]:
    """
    Generate a comprehensive data report.
    
    Args:
        data_dir: Directory containing JSONL files
        run_id: Optional run ID to focus on
    
    Returns:
        Dictionary containing data report
    """
    analyzer = RunAnalyzer(data_dir)
    loader = DataLoader()
    
    # Get all available run IDs if none specified
    if not run_id:
        available_runs = get_available_run_ids(data_dir)
        if available_runs:
            run_id = available_runs[-1]  # Use the latest run
        else:
            return {'error': 'No runs found in data directory'}
    
    # Load data for the specified run
    user_data = analyzer.load_run_data(run_id)
    
    if not user_data:
        return {'error': f'No data found for run {run_id}'}
    
    # Validate data integrity
    validation_results = []
    for user_id, data in user_data.items():
        validation_results.append(validate_data_integrity(data))
    
    # Generate performance summary
    performance_summary = create_performance_summary(user_data, run_id)
    
    # Analyze tweet patterns for a sample of users
    sample_users = list(user_data.keys())[:5]  # Take first 5 users
    tweet_patterns = {}
    
    for user_id in sample_users:
        data = user_data[user_id]
        all_tweets = data.historical_tweets + data.holdout_tweets
        tweet_patterns[user_id] = analyze_tweet_patterns(all_tweets)
    
    report = {
        'run_id': run_id,
        'data_directory': data_dir,
        'timestamp': datetime.now().isoformat(),
        'data_overview': {
            'total_users': len(user_data),
            'total_files': len(get_user_files(data_dir)),
            'available_runs': get_available_run_ids(data_dir)
        },
        'validation_results': validation_results,
        'performance_summary': performance_summary,
        'tweet_patterns_sample': tweet_patterns,
        'data_quality': {
            'valid_users': sum(1 for v in validation_results if v['is_valid']),
            'invalid_users': sum(1 for v in validation_results if not v['is_valid']),
            'common_issues': Counter(
                issue for v in validation_results 
                for issue in v['issues']
            ).most_common(10)
        }
    }
    
    return report


if __name__ == "__main__":
    # Example usage
    data_dir = "data/users"
    run_id = "20250702_114023"
    
    # Generate comprehensive report
    report = generate_data_report(data_dir, run_id)
    
    print(f"Data Report for {run_id}")
    print("=" * 50)
    print(f"Total users: {report['data_overview']['total_users']}")
    print(f"Available runs: {report['data_overview']['available_runs']}")
    print(f"Valid users: {report['data_quality']['valid_users']}")
    print(f"Invalid users: {report['data_quality']['invalid_users']}")
    
    if 'performance_summary' in report:
        perf = report['performance_summary']
        if 'overall_stats' in perf:
            print(f"Average combined score: {perf['overall_stats']['avg_combined']:.3f}")
            print(f"Total evaluations: {perf['overall_stats']['total_evaluations']}")