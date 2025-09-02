# final_analysis.py

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from round_analysis import RoundAnalyzer
from scipy import stats
import warnings
import re

warnings.filterwarnings('ignore')

# --- Configuration ---
RUN_PREFIX = "test_run_01"
script_dir = os.path.dirname(os.path.abspath(__file__))
# Erstellen Sie den absoluten Pfad zum Benutzerverzeichnis
USERS_DIR = os.path.join(script_dir, "data/filtered_users")

OUTPUT_DIR = os.path.join(script_dir, f"output/{RUN_PREFIX}_final_analysis/")
# --- Helper Functions ---

def _sanitize_filename(title):
    """Sanitizes a string to be a valid filename."""
    title = title.replace(' ', '_').replace(':', '_')
    return re.sub(r'[^a-zA-Z0-9_.-]', '', title)

def find_run_ids(users_directory, prefix):
    """Find all run_ids in the user files that start with a given prefix."""
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
                if len(lines) > 2:
                    data = json.loads(lines[2])
                    if 'runs' in data:
                        for run in data.get('runs', []):
                            run_id = run.get('run_id')
                            if run_id and run_id.startswith(prefix):
                                run_ids.add(run_id)
        except (json.JSONDecodeError, IndexError, FileNotFoundError):
            continue
    return sorted(list(run_ids))

def extract_metrics(df):
    """Extract comprehensive evaluation metrics from nested data structure."""
    extracted_rows = []
    print(f"Processing {len(df)} rows for metric extraction...")
    for idx, row in df.iterrows():
        try:
            base_row = {
                'user_id': row['user_id'], 'round': row['round'],
                'run_id': row['run_id'], 'timestamp': row['timestamp']
            }
            if isinstance(row['statistics'], dict):
                stats_dict = row['statistics']
                base_row.update({
                    'mean_combined_score': stats_dict.get('mean_combined_score', 0),
                    'std_combined_score': stats_dict.get('std_combined_score', 0),
                    'best_score': stats_dict.get('best_score', 0),
                    'worst_score': stats_dict.get('worst_score', 0)
                })
            if isinstance(row['overall'], dict):
                overall_dict = row['overall']
                if 'rouge' in overall_dict and isinstance(overall_dict['rouge'], dict):
                    base_row.update({
                        'overall_rouge1': overall_dict['rouge'].get('rouge1', 0),
                        'overall_rouge2': overall_dict['rouge'].get('rouge2', 0),
                        'overall_rougeL': overall_dict['rouge'].get('rougeL', 0)
                    })
                if 'bleu' in overall_dict and isinstance(overall_dict['bleu'], dict):
                    base_row['overall_bleu'] = overall_dict['bleu'].get('bleu', 0)

                # Extract new metrics: perplexity, bertscore, llm_evaluation
                if 'perplexity' in overall_dict and isinstance(overall_dict['perplexity'], dict):
                    base_row['perplexity'] = overall_dict['perplexity'].get('mean_perplexity', 0)
                if 'bertscore' in overall_dict and isinstance(overall_dict['bertscore'], dict):
                    base_row['bertscore_precision'] = np.mean(overall_dict['bertscore'].get('precision', [0]))
                    base_row['bertscore_recall'] = np.mean(overall_dict['bertscore'].get('recall', [0]))
                    base_row['bertscore_f1'] = np.mean(overall_dict['bertscore'].get('f1', [0]))
                if 'llm_evaluation' in overall_dict:
                    base_row['llm_evaluation'] = overall_dict.get('llm_evaluation', 0)

            if isinstance(row['individual_scores'], list):
                scores = {
                    'combined': [item.get('combined_score') for item in row['individual_scores'] if isinstance(item, dict) and 'combined_score' in item],
                    'rouge1': [item['rouge'].get('rouge1', 0) for item in row['individual_scores'] if isinstance(item, dict) and 'rouge' in item],
                    'bleu': [item['bleu'].get('bleu', 0) for item in row['individual_scores'] if isinstance(item, dict) and 'bleu' in item],
                    'perplexity': [item['perplexity'].get('mean_perplexity', 0) for item in row['individual_scores'] if isinstance(item, dict) and 'perplexity' in item],
                    'bertscore_f1': [np.mean(item['bertscore'].get('f1', [0])) for item in row['individual_scores'] if isinstance(item, dict) and 'bertscore' in item],
                    'llm_evaluation': [item.get('llm_evaluation', 0) for item in row['individual_scores'] if isinstance(item, dict) and 'llm_evaluation' in item]
                }
                if scores['combined']:
                    base_row.update({
                        'individual_mean_combined': np.mean(scores['combined']),
                        'individual_std_combined': np.std(scores['combined']),
                        'individual_max_combined': np.max(scores['combined']),
                        'individual_min_combined': np.min(scores['combined'])
                    })
                if scores['rouge1']:
                    base_row['individual_mean_rouge1'] = np.mean(scores['rouge1'])
                if scores['bleu']:
                    base_row['individual_mean_bleu'] = np.mean(scores['bleu'])
                if scores['perplexity']:
                    base_row['individual_mean_perplexity'] = np.mean(scores['perplexity'])
                if scores['bertscore_f1']:
                    base_row['individual_mean_bertscore_f1'] = np.mean(scores['bertscore_f1'])
                if scores['llm_evaluation']:
                    base_row['individual_mean_llm_evaluation'] = np.mean(scores['llm_evaluation'])
            extracted_rows.append(base_row)
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
    print(f"Successfully extracted metrics from {len(extracted_rows)} rows.")
    return pd.DataFrame(extracted_rows)

# --- Plotting and Analysis Functions from Notebook ---

def create_comprehensive_metric_plots(df_metrics, metric_name, save_dir=None):
    """Create and save a 4-panel plot for a single metric."""
    if metric_name not in df_metrics.columns:
        print(f"Metric '{metric_name}' not found.")
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    title_text = f'Comprehensive Analysis: {metric_name.replace("_", " ").title()}'
    fig.suptitle(title_text, fontsize=20, fontweight='bold')

    rounds = sorted(df_metrics['round'].unique())
    
    # 1. Box plot
    ax1 = axes[0, 0]
    box_data = [df_metrics[df_metrics['round'] == r][metric_name].dropna() for r in rounds]
    bp = ax1.boxplot(box_data, labels=rounds, patch_artist=True, medianprops=dict(color='black'))
    colors = plt.cm.viridis(np.linspace(0, 1, len(rounds)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax1.set_title('Distribution by Round', fontweight='bold')
    ax1.set_xlabel('Round')
    ax1.set_ylabel(metric_name.replace("_", " ").title())
    ax1.grid(True, alpha=0.3)

    # 2. Line plot
    ax2 = axes[0, 1]
    for user_id in df_metrics['user_id'].unique():
        user_data = df_metrics[df_metrics['user_id'] == user_id].sort_values('round')
        if len(user_data) > 1:
            ax2.plot(user_data['round'], user_data[metric_name], alpha=0.3, color='lightblue', marker='o', markersize=3)
    round_stats = df_metrics.groupby('round')[metric_name].agg(['mean', 'std', 'count'])
    ax2.plot(round_stats.index, round_stats['mean'], color='red', linewidth=3, marker='o', label='Mean')
    ax2.set_title('Progression Over Rounds', fontweight='bold')
    ax2.set_xlabel('Round')
    ax2.set_ylabel(metric_name.replace("_", " ").title())
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. KDE plot
    ax3 = axes[1, 0]
    for i, round_num in enumerate(rounds):
        round_data = df_metrics[df_metrics['round'] == round_num][metric_name].dropna()
        if not round_data.empty:
            sns.kdeplot(round_data, ax=ax3, label=f'Round {round_num}', color=colors[i], fill=True, alpha=0.3)
    ax3.set_title('Distribution Shape by Round (KDE)', fontweight='bold')
    ax3.set_xlabel(metric_name.replace("_", " ").title())
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Bar plot with error bars
    ax4 = axes[1, 1]
    x_pos = np.arange(len(round_stats))
    ax4.bar(x_pos, round_stats['mean'], yerr=round_stats['std'], capsize=5, alpha=0.7, color=colors)
    ax4.set_title('Mean ± Standard Deviation by Round', fontweight='bold')
    ax4.set_xlabel('Round')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f'R{r}' for r in round_stats.index])
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
    metric_cols = [col for col in metric_cols if col not in ['round', 'user_id', 'timestamp']]
    
    print(f"\n=== Analyzing {len(metric_cols)} metrics individually ===")
    for metric in metric_cols:
        print(f"\n--- Analyzing: {metric} ---")
        create_comprehensive_metric_plots(df_metrics, metric, save_dir=save_dir)

# --- Main Execution ---

def main():
    """Main function to run the complete analysis and save all artifacts."""
    print(f"Starting analysis for runs with prefix: '{RUN_PREFIX}'")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load and filter data
    analyzer = RoundAnalyzer(users_directory=USERS_DIR)
    df_raw = analyzer.analyze_all_users()
    if df_raw.empty:
        print("No data loaded. Exiting.")
        return

    matching_run_ids = find_run_ids(USERS_DIR, RUN_PREFIX)
    if not matching_run_ids:
        print("No matching run IDs found. Exiting.")
        return
    print(f"Found {len(matching_run_ids)} matching runs.")
    
    df_filtered = df_raw[df_raw['run_id'].isin(matching_run_ids)]
    if df_filtered.empty:
        print("No data for the specified run IDs. Exiting.")
        return

    # Extract metrics
    df_metrics = extract_metrics(df_filtered)
    df_metrics_path = os.path.join(OUTPUT_DIR, 'extracted_metrics.csv')
    df_metrics.to_csv(df_metrics_path, index=False)
    print(f"Saved extracted metrics to {df_metrics_path}")

    # Generate and save tables
    numeric_df = df_metrics.select_dtypes(include=np.number)
    aggregated_results = numeric_df.groupby('round').agg(['mean', 'std', 'count']).round(4)
    agg_path = os.path.join(OUTPUT_DIR, 'aggregated_round_statistics.csv')
    aggregated_results.to_csv(agg_path)
    print(f"Saved aggregated statistics to {agg_path}")

    # Run the comprehensive individual analysis
    analyze_all_metrics_individually(df_metrics, save_dir=OUTPUT_DIR)
    
    print(f"\n✅ All analyses complete. Results are saved in the '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    main()