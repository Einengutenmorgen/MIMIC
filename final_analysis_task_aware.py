# final_analysis_task_aware.py

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from round_analysis import RoundAnalyzer  # Note: Requires external module 'round_analysis'
from scipy import stats
import warnings
import re

warnings.filterwarnings('ignore')

# --- Configuration ---
RUN_PREFIXS = ["run_r50_rouge_1"]

# Assuming __file__ exists. If running interactively, define script_dir manually.
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

USERS_DIR = os.path.join(script_dir, "data/restore")

# --- Helper Functions ---

def _sanitize_filename(title):
    """Sanitizes a string to be a valid filename."""
    title = title.replace(' ', '_').replace(':', '_')
    return re.sub(r'[^a-zA-Z0-9_.-]', '', title)

def find_run_ids(users_directory, prefix):
    """Find all run_ids in the user files that start with a given prefix.

    Updated to check all lines and all possible data structures.
    """
    run_ids = set()
    if not os.path.isdir(users_directory):
        print(f"Error: Directory not found at {users_directory}")
        return []

    user_files = [f for f in os.listdir(users_directory) if f.endswith('.jsonl')]

    for user_file in user_files:
        user_file_path = os.path.join(users_directory, user_file)
        try:
            with open(user_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

                # Check all lines, not just line 3
                for line_num, line in enumerate(lines):
                    try:
                        data = json.loads(line)

                        # 1. Original structure: 'runs' key (line 3)
                        if 'runs' in data:
                            for run in data.get('runs', []):
                                run_id = run.get('run_id')
                                if run_id and run_id.startswith(prefix):
                                    run_ids.add(run_id)

                        # 2. New structure: 'evaluations' key (line 4)
                        if 'evaluations' in data:
                            for eval_item in data.get('evaluations', []):
                                # The run_id might be directly in the evaluation item
                                # or nested within it
                                run_id = eval_item.get('run_id')
                                if run_id and run_id.startswith(prefix):
                                    run_ids.add(run_id)

                                # Check if run_id is nested deeper
                                if isinstance(eval_item, dict):
                                    for key, value in eval_item.items():
                                        if key == 'run_id' and isinstance(value, str) and value.startswith(prefix):
                                            run_ids.add(value)

                        # 3. Check any other list structures that might contain run_ids
                        for key, value in data.items():
                            if isinstance(value, list):
                                for item in value:
                                    if isinstance(item, dict) and 'run_id' in item:
                                        run_id = item.get('run_id')
                                        if run_id and run_id.startswith(prefix):
                                            run_ids.add(run_id)

                    except json.JSONDecodeError:
                        continue

        except (FileNotFoundError, IOError):
            continue

    return sorted(list(run_ids))

def load_tweets_data_map(users_dir):
    """
    Load tweets data for all users to enable task type detection.

    :param users_dir: Directory containing user JSONL files
    :return: Dictionary mapping user_id to tweets data
    """
    tweets_map = {}
    if not os.path.isdir(users_dir):
        print(f"Warning: Tweets data directory not found at {users_dir}")
        return tweets_map

    user_files = [f for f in os.listdir(users_dir) if f.endswith('.jsonl')]

    for user_file in user_files:
        user_file_path = os.path.join(users_dir, user_file)
        try:
            with open(user_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if len(lines) > 1:  # Tweets are typically on line 2 (index 1)
                    data = json.loads(lines[1])
                    user_id = data.get('user_id')
                    tweets = data.get('tweets', [])
                    if user_id:
                        tweets_map[user_id] = tweets
        except (json.JSONDecodeError, FileNotFoundError, IndexError):
            continue

    return tweets_map

def determine_task_type(tweet_id, tweets_data):
    """
    Determine if a tweet is a post or reply based on reply_to_id.

    :param tweet_id: ID of the tweet to check
    :param tweets_data: List of tweet dictionaries from the data
    :return: 'post' or 'reply'
    """
    for tweet in tweets_data:
        if tweet.get('tweet_id') == tweet_id:
            reply_to_id = tweet.get('reply_to_id')
            return 'post' if reply_to_id is None else 'reply'
    return 'unknown'  # Fallback

def extract_masked_tokens(text, prediction, reference):
    """
    Extract only the tokens that were filled in for [MASKED] positions.
    This implementation cleans the text but returns full strings for now.

    :param text: Original masked text with [MASKED] tokens
    :param prediction: Model's prediction
    :param reference: Reference answer
    :return: Tuple of (predicted_tokens, reference_tokens)
    """
    # Remove common prefixes/suffixes that models add
    pred_clean = prediction.replace('Post:', '').replace('Completed Tweet:', '').strip()
    pred_clean = re.sub(r'\*\*([^*]+)\*\*', r'\1', pred_clean)  # Remove **bold**

    # For now, return the full cleaned prediction vs reference for semantic comparison
    return pred_clean.strip(), reference.strip()

def calculate_semantic_similarity(text1, text2):
    """
    Calculate semantic similarity between two texts.
    Simple implementation using word overlap (Jaccard similarity).

    :param text1: First text
    :param text2: Second text
    :return: Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0

    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if len(words1) == 0 and len(words2) == 0:
        return 1.0

    intersection = words1.intersection(words2)
    union = words1.union(words2)

    return len(intersection) / len(union) if len(union) > 0 else 0.0

def calculate_masked_token_metrics(individual_scores, tweets_data):
    """
    Calculate sophisticated metrics for posts (masked token filling).

    :param individual_scores: List of individual score dictionaries
    :param tweets_data: List of tweet data to get masked text
    :return: Dictionary with various metrics
    """
    if not individual_scores:
        return {
            'exact_match_rate': 0,
            'semantic_similarity_mean': 0,
            'content_accuracy': 0,
            'format_compliance': 0,
            'total_samples': 0
        }

    exact_matches = 0
    semantic_scores = []
    content_scores = []
    format_scores = []
    total_samples = len(individual_scores)

    # Create tweet lookup for masked text
    tweet_lookup = {tweet.get('tweet_id'): tweet for tweet in tweets_data}

    for score in individual_scores:
        prediction = score.get('prediction', '').strip()
        reference = score.get('reference', '').strip()
        tweet_id = score.get('tweet_id')

        # Get original masked text
        tweet_data = tweet_lookup.get(tweet_id, {})
        masked_text = tweet_data.get('masked_text', '')

        # Extract filled tokens/cleaned strings
        pred_tokens, ref_tokens = extract_masked_tokens(masked_text, prediction, reference)

        # 1. Exact match (strict on cleaned prediction vs reference)
        if pred_tokens.lower() == ref_tokens.lower():
            exact_matches += 1

        # 2. Semantic similarity on extracted tokens/strings
        semantic_sim = calculate_semantic_similarity(pred_tokens, ref_tokens)
        semantic_scores.append(semantic_sim)

        # 3. Content accuracy (high semantic similarity threshold)
        content_score = 1.0 if semantic_sim > 0.7 else (semantic_sim if semantic_sim > 0.3 else 0.0)
        content_scores.append(content_score)

        # 4. Format compliance (does it look like expected output?)
        format_score = 1.0

        # Penalize if prediction has formatting artifacts that extract_masked_tokens missed or if it contains common failure modes
        if any(artifact in prediction for artifact in ['Post:', 'Completed Tweet:', '**', 'flexed!']):
            format_score *= 0.5

        # Penalize if wildly different length (simple heuristic)
        if ref_tokens:
            len_ratio = len(pred_tokens) / max(len(ref_tokens), 1)
            if len_ratio > 3 or len_ratio < 0.3:  # Too long or too short
                format_score *= 0.7

        format_scores.append(format_score)

    return {
        'exact_match_rate': exact_matches / total_samples if total_samples > 0 else 0,
        'semantic_similarity_mean': np.mean(semantic_scores) if semantic_scores else 0,
        'content_accuracy': np.mean(content_scores) if content_scores else 0,
        'format_compliance': np.mean(format_scores) if format_scores else 0,
        'exact_matches': exact_matches,
        'total_samples': total_samples,
        'semantic_similarity_std': np.std(semantic_scores) if semantic_scores else 0
    }

def compute_missing_reply_metrics(individual_scores, user_tweets, run_id):
    """
    Compute missing metrics for reply tweets using existing evaluation functions.
    Only computes metrics for replies, not posts.

    :param individual_scores: List of individual score dictionaries
    :param user_tweets: List of tweet data for task type detection
    :param run_id: Run ID to determine which metrics to compute
    :return: Dictionary with computed metrics for each item
    """
    # Determine which metrics are missing based on run_id
    if "bleu_only" in run_id:
        missing_metrics = ["rouge", "bertscore", "perplexity"]
    elif "perplexity_only" in run_id:
        missing_metrics = ["rouge", "bleu", "bertscore"]
    elif "bertscore" in run_id:
        missing_metrics = ["rouge", "bleu", "perplexity"]
    elif "rouge_1" in run_id:
        missing_metrics = ["bleu", "bertscore", "perplexity"]
    else:
        return {}  # all_metrics has everything, no computation needed

    print(f"Computing missing metrics {missing_metrics} for run_id: {run_id}")

    # Import evaluation functions dynamically
    try:
        from evaluate import load
    except ImportError as e:
        print(f"Error importing 'evaluate' library. Please install it: pip install evaluate. Error: {e}")
        return {}

    # Load the required metrics once
    rouge = load("rouge") if "rouge" in missing_metrics else None
    bleu = load("bleu") if "bleu" in missing_metrics else None
    bertscore = load("bertscore") if "bertscore" in missing_metrics else None
    perplexity = load("perplexity", module_type="metric") if "perplexity" in missing_metrics else None

    computed_results = {}

    # Process each individual score
    for i, score in enumerate(individual_scores):
        tweet_id = score.get('tweet_id')

        # Check if this is a reply (skip posts)
        task_type = determine_task_type(tweet_id, user_tweets)
        if task_type != 'reply':
            continue

        prediction = score.get('prediction', '')
        reference = score.get('reference', '')

        if not prediction or not reference:
            continue

        computed_metrics = {}

        try:
            # Compute ROUGE if missing
            if rouge is not None:
                rouge_result = rouge.compute(
                    predictions=[prediction],
                    references=[reference],
                    use_stemmer=False
                )
                computed_metrics['computed_rouge'] = rouge_result

            # Compute BLEU if missing
            if bleu is not None:
                bleu_result = bleu.compute(
                    predictions=[prediction],
                    references=[[reference]]
                )
                computed_metrics['computed_bleu'] = bleu_result

            # Compute BERTScore if missing
            if bertscore is not None:
                bertscore_result = bertscore.compute(
                    predictions=[prediction],
                    references=[reference],
                    lang="en"
                )
                computed_metrics['computed_bertscore'] = {
                    'precision': bertscore_result['precision'][0],
                    'recall': bertscore_result['recall'][0],
                    'f1': bertscore_result['f1'][0],
                }

            # Compute Perplexity if missing
            if perplexity is not None:
                try:
                    perplexity_result = perplexity.compute(
                        predictions=[prediction],
                        model_id='gpt2'
                    )
                    computed_metrics['computed_perplexity'] = perplexity_result['mean_perplexity']
                except Exception as pe:
                    # Catch errors during perplexity calculation (e.g., empty prediction)
                    print(f"Warning: Could not calculate perplexity for item {i}: {pe}")
                    computed_metrics['computed_perplexity'] = np.nan

        except Exception as e:
            print(f"Error computing metrics for item {i}: {e}")
            continue

        computed_results[i] = computed_metrics

    return computed_results

def extract_metrics_task_aware(df, tweets_data_map):
    """
    Extract metrics with task-aware evaluation (posts vs replies).
    Now includes computation of missing metrics for limited-metric runs.

    :param df: DataFrame with evaluation data
    :param tweets_data_map: Dictionary mapping user_id to tweets data
    :return: DataFrame with task-aware metrics including computed metrics
    """
    extracted_rows = []
    print(f"Processing {len(df)} rows for task-aware metric extraction...")

    for idx, row in df.iterrows():
        try:
            base_row = {
                'user_id': row['user_id'], 'round': row['round'],
                'run_id': row['run_id'], 'timestamp': row['timestamp']
            }

            # Get tweets data for this user
            user_tweets = tweets_data_map.get(row['user_id'], [])
            if not user_tweets:
                print(f"Warning: No tweet data found for user {row['user_id']}")

            # Separate individual scores by task type
            if isinstance(row['individual_scores'], list):
                posts_scores = []
                replies_scores = []

                for score in row['individual_scores']:
                    if isinstance(score, dict) and 'tweet_id' in score:
                        tweet_id = score['tweet_id']
                        task_type = determine_task_type(tweet_id, user_tweets)

                        if task_type == 'post':
                            posts_scores.append(score)
                        elif task_type == 'reply':
                            replies_scores.append(score)

                # Calculate metrics for posts (enhanced masked token evaluation)
                if posts_scores:
                    post_metrics = calculate_masked_token_metrics(posts_scores, user_tweets)
                    base_row.update({
                        'posts_exact_match_rate': post_metrics['exact_match_rate'],
                        'posts_semantic_similarity': post_metrics['semantic_similarity_mean'],
                        'posts_content_accuracy': post_metrics['content_accuracy'],
                        'posts_format_compliance': post_metrics['format_compliance'],
                        'posts_exact_matches': post_metrics['exact_matches'],
                        'posts_total_samples': post_metrics['total_samples']
                    })

                # Calculate metrics for replies (traditional NLG metrics + computed missing metrics)
                if replies_scores:
                    # Compute missing metrics for limited-metric runs
                    computed_metrics = compute_missing_reply_metrics(
                        replies_scores, user_tweets, row['run_id']
                    )

                    # --- Data extraction loop ---
                    # Collect original metrics from score data structure
                    orig_rouge_scores = []
                    orig_bleu_scores = []
                    orig_combined_scores = []
                    orig_perp_scores = []
                    orig_bertscore_scores = [] # Assuming bertscore might be present in 'all_metrics' runs

                    # Collect newly computed metrics separately
                    computed_rouge_scores = []
                    computed_bleu_scores = []
                    computed_perp_scores = []
                    computed_bertscore_scores = []

                    for i, score in enumerate(replies_scores):
                        # Original metrics from 'individual_scores' field
                        if 'rouge' in score and isinstance(score['rouge'], dict):
                            orig_rouge_scores.append(score['rouge'].get('rouge1', np.nan))
                        if 'bleu' in score and isinstance(score['bleu'], dict):
                            orig_bleu_scores.append(score['bleu'].get('bleu', np.nan))
                        if 'combined_score' in score:
                            orig_combined_scores.append(score.get('combined_score', np.nan))
                        if 'perplexity' in score and isinstance(score['perplexity'], dict):
                            orig_perp_scores.append(score['perplexity'].get('mean_perplexity', np.nan))
                        if 'bertscore' in score and isinstance(score['bertscore'], dict):
                            orig_bertscore_scores.append(score['bertscore'].get('f1', np.nan))

                        # Computed metrics (fill in missing ones)
                        if i in computed_metrics:
                            comp_metrics = computed_metrics[i]
                            if 'computed_rouge' in comp_metrics:
                                computed_rouge_scores.append(comp_metrics['computed_rouge'].get('rouge1', np.nan))
                            if 'computed_bleu' in comp_metrics:
                                computed_bleu_scores.append(comp_metrics['computed_bleu'].get('bleu', np.nan))
                            if 'computed_perplexity' in comp_metrics:
                                computed_perp_scores.append(comp_metrics.get('computed_perplexity', np.nan))
                            if 'computed_bertscore' in comp_metrics:
                                computed_bertscore_scores.append(comp_metrics['computed_bertscore'].get('f1', np.nan))

                    # --- Consolidation logic ---
                    # Use original metrics if available, otherwise use computed metrics.
                    # This ensures runs like "run_r50_all_metrics" use their original data,
                    # while runs like "run_r50_bleu_only" get populated with computed ROUGE, etc.
                    final_rouge_scores = orig_rouge_scores if orig_rouge_scores else computed_rouge_scores
                    final_bleu_scores = orig_bleu_scores if orig_bleu_scores else computed_bleu_scores
                    final_perp_scores = orig_perp_scores if orig_perp_scores else computed_perp_scores
                    final_bertscore_scores = orig_bertscore_scores if orig_bertscore_scores else computed_bertscore_scores

                    # Add reply metrics to base_row, checking if list is non-empty before calculating mean
                    if final_rouge_scores:
                        base_row['replies_mean_rouge1'] = np.nanmean(final_rouge_scores)
                    if final_bleu_scores:
                        base_row['replies_mean_bleu'] = np.nanmean(final_bleu_scores)
                    if final_perp_scores:
                        base_row['replies_mean_perplexity'] = np.nanmean(final_perp_scores)
                    if final_bertscore_scores:
                        base_row['replies_mean_bertscore_f1'] = np.nanmean(final_bertscore_scores)

                    # Combined scores logic
                    if orig_combined_scores:
                        base_row['replies_mean_combined'] = np.nanmean(orig_combined_scores)
                        base_row['replies_std_combined'] = np.nanstd(orig_combined_scores)
                        base_row['replies_total_samples'] = len(orig_combined_scores)
                    elif final_rouge_scores and final_bleu_scores:
                        # Re-calculate combined score if original wasn't present but components are
                        # Ensure lists have the same length for zip operation
                        min_len = min(len(final_rouge_scores), len(final_bleu_scores))
                        computed_combined = [
                            (r * 0.5 + b * 0.5) for r, b in zip(final_rouge_scores[:min_len], final_bleu_scores[:min_len])
                            if not np.isnan(r) and not np.isnan(b)
                        ]
                        if computed_combined:
                            base_row['replies_mean_combined'] = np.mean(computed_combined)
                            base_row['replies_std_combined'] = np.std(computed_combined)
                            base_row['replies_total_samples'] = len(computed_combined)

            # Add overall statistics (keeping existing for compatibility)
            if isinstance(row['statistics'], dict):
                stats_dict = row['statistics']
                base_row.update({
                    'overall_mean_combined_score': stats_dict.get('mean_combined_score', np.nan),
                    'overall_std_combined_score': stats_dict.get('std_combined_score', np.nan),
                    'overall_best_score': stats_dict.get('best_score', np.nan),
                    'overall_worst_score': stats_dict.get('worst_score', np.nan)
                })

            extracted_rows.append(base_row)

        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            # Optionally re-raise to stop execution: raise e

    print(f"Successfully extracted task-aware metrics from {len(extracted_rows)} rows.")
    return pd.DataFrame(extracted_rows)

def create_task_aware_plots(df_metrics, save_dir=None):
    """
    Create separate plots for posts and replies with appropriate metrics.
    Layout: 2x2 grid with posts metrics on top, replies metrics on bottom.
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Task-Aware Performance Analysis: Posts vs Replies', fontsize=20, fontweight='bold')

    rounds = sorted(df_metrics['round'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(rounds)))

    # --- TOP ROW: POSTS METRICS ---

    # Plot 1 (Top Left): Posts - Multiple Metrics Distribution (Box Plot)
    ax1 = axes[0, 0]
    post_metrics = ['posts_exact_match_rate', 'posts_semantic_similarity', 'posts_content_accuracy', 'posts_format_compliance']
    post_labels = ['Exact Match', 'Semantic Sim.', 'Content Acc.', 'Format Compl.']
    post_colors = ['red', 'blue', 'green', 'orange']

    box_data_all = []
    box_labels_used = []

    for metric, label in zip(post_metrics, post_labels):
        if metric in df_metrics.columns:
            metric_data = df_metrics[metric].dropna()
            if not metric_data.empty:
                box_data_all.append(metric_data)
                box_labels_used.append(label)

    if box_data_all:
        bp = ax1.boxplot(box_data_all, patch_artist=True, medianprops=dict(color='black'))
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(post_colors[i % len(post_colors)])
            patch.set_alpha(0.7)
        ax1.set_xticks(range(1, len(box_labels_used) + 1))
        ax1.set_xticklabels(box_labels_used, rotation=45)

    ax1.set_title('Posts: Multiple Metrics Distribution (All Rounds)', fontweight='bold')
    ax1.set_xlabel('Metric Type')
    ax1.set_ylabel('Score')
    ax1.grid(True, alpha=0.3)

    # Plot 2 (Top Right): Posts - Multi-Metric Progression Over Time (Line Plot)
    ax2 = axes[0, 1]
    for metric, color, label in zip(post_metrics, post_colors, post_labels):
        if metric in df_metrics.columns:
            post_data = df_metrics[df_metrics[metric].notna()]
            if not post_data.empty:
                round_means = post_data.groupby('round')[metric].mean()
                if not round_means.empty:
                    ax2.plot(round_means.index, round_means.values, color=color,
                             linewidth=2, marker='o', label=label, markersize=4)

    ax2.set_title('Posts: Multi-Metric Progression', fontweight='bold')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # --- BOTTOM ROW: REPLIES METRICS ---

    # Plot 3 (Bottom Left): Replies - Combined Score Distribution by Round (Box Plot)
    ax3 = axes[1, 0]
    if 'replies_mean_combined' in df_metrics.columns:
        reply_data = df_metrics[df_metrics['replies_mean_combined'].notna()]
        if not reply_data.empty:
            box_data = []
            valid_rounds = []
            for r in rounds:
                round_data = reply_data[reply_data['round'] == r]['replies_mean_combined'].dropna()
                if not round_data.empty:
                    box_data.append(round_data)
                    valid_rounds.append(r)

            if box_data:
                bp = ax3.boxplot(box_data, patch_artist=True, medianprops=dict(color='black'))
                for i, patch in enumerate(bp['boxes']):
                    patch.set_facecolor(colors[rounds.index(valid_rounds[i])])
                    patch.set_alpha(0.7)

                ax3.set_xticks(range(1, len(valid_rounds) + 1))
                ax3.set_xticklabels([f'R{r}' for r in valid_rounds], rotation=45)

    ax3.set_title('Replies: Combined Score Distribution by Round', fontweight='bold')
    ax3.set_xlabel('Round')
    ax3.set_ylabel('Combined Score')
    ax3.grid(True, alpha=0.3)

    # Plot 4 (Bottom Right): Replies - Multi-Metric Progression Over Time (Line Plot)
    ax4 = axes[1, 1]
    reply_metrics = ['replies_mean_combined', 'replies_mean_rouge1', 'replies_mean_bleu', 'replies_mean_perplexity', 'replies_mean_bertscore_f1']
    metric_colors = ['red', 'blue', 'green', 'purple', 'orange']
    metric_labels = ['Combined Score', 'ROUGE-1', 'BLEU', 'Perplexity', 'BERTScore F1']

    for metric, color, label in zip(reply_metrics, metric_colors, metric_labels):
        if metric in df_metrics.columns:
            reply_data = df_metrics[df_metrics[metric].notna()]
            if not reply_data.empty:
                round_means = reply_data.groupby('round')[metric].mean()
                if not round_means.empty:
                    ax4.plot(round_means.index, round_means.values, color=color,
                             linewidth=2, marker='o', label=label, markersize=4)

    ax4.set_title('Replies: Multi-Metric Progression (Original + Computed)', fontweight='bold')
    ax4.set_xlabel('Round')
    ax4.set_ylabel('Score')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_dir:
        filepath = os.path.join(save_dir, "Task_Aware_Analysis_Posts_vs_Replies.png")
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"✅ Task-aware plot saved to {filepath}")

    plt.close(fig)

def create_comprehensive_metric_plots(df_metrics, metric_name, save_dir=None):
    """Create and save a 4-panel plot for a single metric with fixed formatting."""
    if metric_name not in df_metrics.columns or df_metrics[metric_name].dropna().empty:
        print(f"Skipping plot for '{metric_name}': No data available.")
        return

    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    title_text = f'Comprehensive Analysis: {metric_name.replace("_", " ").title()}'
    fig.suptitle(title_text, fontsize=20, fontweight='bold')

    rounds = sorted(df_metrics['round'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(rounds)))

    # --- Plot 1 (Top Left): Box plot by Round ---
    ax1 = axes[0, 0]
    box_data = []
    valid_rounds_box = []
    for r in rounds:
        round_data = df_metrics[df_metrics['round'] == r][metric_name].dropna()
        if not round_data.empty:
            box_data.append(round_data)
            valid_rounds_box.append(r)

    if box_data:
        bp = ax1.boxplot(box_data, patch_artist=True, medianprops=dict(color='black'))
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(colors[rounds.index(valid_rounds_box[i])])
            patch.set_alpha(0.7)

        ax1.set_xticks(range(1, len(valid_rounds_box) + 1))
        ax1.set_xticklabels([f'R{r}' for r in valid_rounds_box], rotation=45)

    ax1.set_title('Distribution by Round', fontweight='bold')
    ax1.set_xlabel('Round')
    ax1.set_ylabel(metric_name.replace("_", " ").title())
    ax1.grid(True, alpha=0.3)

    # --- Plot 2 (Top Right): Line plot of progression over rounds ---
    ax2 = axes[0, 1]
    for user_id in df_metrics['user_id'].unique():
        user_data = df_metrics[df_metrics['user_id'] == user_id].sort_values('round')
        if len(user_data) > 1:
            ax2.plot(user_data['round'], user_data[metric_name], alpha=0.3, color='lightblue', marker='o', markersize=3)

    round_stats = df_metrics.groupby('round')[metric_name].agg(['mean', 'std', 'count'])
    if not round_stats.empty:
        ax2.plot(round_stats.index, round_stats['mean'], color='red', linewidth=3, marker='o', label='Mean')

    ax2.set_title('Progression Over Rounds', fontweight='bold')
    ax2.set_xlabel('Round')
    ax2.set_ylabel(metric_name.replace("_", " ").title())
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # --- Plot 3 (Bottom Left): KDE plot for distribution shape ---
    ax3 = axes[1, 0]
    legend_rounds = []
    if len(rounds) > 10:
        legend_rounds = [rounds[0]] + rounds[4::5] + [rounds[-1]]
        legend_rounds = sorted(list(set(legend_rounds)))
    else:
        legend_rounds = rounds

    for i, round_num in enumerate(rounds):
        round_data = df_metrics[df_metrics['round'] == round_num][metric_name].dropna()
        if not round_data.empty:
            label = f'Round {round_num}' if round_num in legend_rounds else None
            sns.kdeplot(round_data, ax=ax3, label=label, color=colors[i], fill=True, alpha=0.3)

    ax3.set_title('Distribution Shape by Round (KDE)', fontweight='bold')
    ax3.set_xlabel(metric_name.replace("_", " ").title())
    ax3.set_ylabel('Density')
    if legend_rounds and any(label is not None for label in ax3.get_legend_handles_labels()[1]):
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
    ax3.grid(True, alpha=0.3)

    # --- Plot 4 (Bottom Right): Bar plot of mean and std dev ---
    ax4 = axes[1, 1]
    if not round_stats.empty:
        x_pos = np.arange(len(round_stats))
        ax4.bar(x_pos, round_stats['mean'], yerr=round_stats['std'], capsize=5, alpha=0.7, color=[colors[rounds.index(r)] for r in round_stats.index])
        ax4.set_title('Mean ± Standard Deviation by Round', fontweight='bold')
        ax4.set_xlabel('Round')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([f'R{r}' for r in round_stats.index], rotation=45)
        ax4.set_ylabel(metric_name.replace("_", " ").title())
        ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_dir:
        filename = _sanitize_filename(title_text)
        filepath = os.path.join(save_dir, f"{filename}.png")
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"✅ Plot saved to {filepath}")
    plt.close(fig)

def analyze_all_metrics_individually(df_metrics, save_dir=None):
    """Run the comprehensive plot generation for all numeric metrics."""
    metric_cols = df_metrics.select_dtypes(include=np.number).columns.tolist()
    # Exclude non-score columns
    metric_cols = [col for col in metric_cols if col not in ['round', 'user_id', 'timestamp', 'posts_exact_matches', 'posts_total_samples', 'replies_total_samples']]

    print(f"\n=== Analyzing {len(metric_cols)} metrics individually ===")
    for metric in metric_cols:
        print(f"\n--- Analyzing: {metric} ---")
        create_comprehensive_metric_plots(df_metrics, metric, save_dir=save_dir)

def process_single_prefix(run_prefix, users_dir, script_dir):
    """Process a single run prefix and generate all analysis outputs."""
    print(f"\n{'='*60}")
    print(f"Starting analysis for runs with prefix: '{run_prefix}'")
    print(f"{'='*60}")

    output_dir = os.path.join(script_dir, f"results/{run_prefix}_final_analysis/")
    os.makedirs(output_dir, exist_ok=True)

    # Load tweets data for task type detection
    print("Loading tweets data for task type detection...")
    tweets_data_map = load_tweets_data_map(users_dir)
    print(f"Loaded tweets data for {len(tweets_data_map)} users")

    # Load and filter data using RoundAnalyzer
    # Ensure RoundAnalyzer is imported or defined, and it has analyze_all_users method.
    try:
        analyzer = RoundAnalyzer(users_directory=users_dir)
        df_raw = analyzer.analyze_all_users()
    except Exception as e:
        print(f"Error initializing or running RoundAnalyzer: {e}")
        print("Please ensure 'round_analysis.py' exists and is correctly implemented.")
        return

    if df_raw.empty:
        print(f"No data loaded from RoundAnalyzer for prefix '{run_prefix}'. Skipping.")
        return

    matching_run_ids = find_run_ids(users_dir, run_prefix)
    if not matching_run_ids:
        print(f"No matching run IDs found in user files for prefix '{run_prefix}'. Checking raw data anyway.")
        # Fallback check in case find_run_ids logic differs from RoundAnalyzer's scope
        df_filtered = df_raw[df_raw['run_id'].str.startswith(run_prefix, na=False)]
    else:
        print(f"Found {len(matching_run_ids)} matching run IDs for prefix '{run_prefix}'. Filtering data.")
        df_filtered = df_raw[df_raw['run_id'].isin(matching_run_ids)]

    if df_filtered.empty:
        print(f"No data rows match the specified prefix '{run_prefix}'. Skipping.")
        return

    # Extract task-aware metrics
    df_metrics = extract_metrics_task_aware(df_filtered, tweets_data_map)
    if df_metrics.empty:
        print(f"Metric extraction resulted in an empty DataFrame for prefix '{run_prefix}'. Skipping further analysis.")
        return

    df_metrics_path = os.path.join(output_dir, 'extracted_task_aware_metrics.csv')
    df_metrics.to_csv(df_metrics_path, index=False)
    print(f"Saved task-aware metrics to {df_metrics_path}")

    # Generate and save tables
    numeric_df = df_metrics.select_dtypes(include=np.number)
    if not numeric_df.empty and 'round' in numeric_df.columns:
        aggregated_results = numeric_df.groupby('round').agg(['mean', 'std', 'count']).round(4)
        agg_path = os.path.join(output_dir, 'aggregated_task_aware_statistics.csv')
        aggregated_results.to_csv(agg_path)
        print(f"Saved aggregated statistics to {agg_path}")
    else:
        print("Skipping aggregated statistics generation: no numeric data or 'round' column found.")

    # Create task-aware visualization
    print("\nGenerating task-aware plots...")
    create_task_aware_plots(df_metrics, save_dir=output_dir)

    # Run the comprehensive individual analysis for all metrics
    print("\nGenerating individual metric analyses...")
    analyze_all_metrics_individually(df_metrics, save_dir=output_dir)

    print(f"\n✅ Analysis complete for prefix '{run_prefix}'. Results are saved in '{output_dir}'")

# --- Main Execution ---

def main():
    """Main function to run the complete analysis for all prefixes and save all artifacts."""

    print(f"Starting comprehensive task-aware analysis for {len(RUN_PREFIXS)} run prefixes:")
    for prefix in RUN_PREFIXS:
        print(f"  - {prefix}")

    # Process each prefix individually
    for run_prefix in RUN_PREFIXS:
        try:
            process_single_prefix(run_prefix, USERS_DIR, script_dir)
        except Exception as e:
            print(f"❌ Error processing prefix '{run_prefix}': {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n🎉 All analyses complete! Processed {len(RUN_PREFIXS)} run prefixes.")
    print("Results are saved in their respective output directories under 'results/'")

if __name__ == "__main__":
    main()