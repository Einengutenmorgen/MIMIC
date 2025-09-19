#!/usr/bin/env python3
"""
Baseline Experiment Analysis Script (V3 - Task-Aware)

Analyzes results from baseline experiments comparing:
1. No-Persona: Task performance without persona conditioning  
2. Generic-Persona: Minimal universal social media user persona
3. History-Only: Raw user data without LLM abstraction
4. Best-Persona: Optimized persona from iterative rounds

This version includes a task-aware analysis, differentiating between
'post_completion' and 'reply_generation' to test for interaction effects.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
from scipy.stats import ttest_rel
import argparse
import pingouin as pg
from file_cache import get_file_cache
from logging_config import logger
from loader import load_stimulus 


# MODIFIED: Add 'task_type' to the result dataclass
@dataclass
class BaselineResult:
    """Individual user's baseline condition result for a specific task."""
    user_id: str
    condition: str
    task_type: str  # NEW: 'Post Completion', 'Reply Generation', or 'Overall'
    run_id: str
    bertscore_f1: float
    rouge_l: float
    bleu: float
    num_samples: int

@dataclass
class StatisticalTest:
    """Statistical test result."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    interpretation: str

class BaselineAnalyzer:
    """
    Comprehensive analysis of baseline experiment results.
    CORRECTED VERSION: Aligned with the actual data structure, this class processes
    the 'individual_scores' list to manually aggregate metrics by task type
    ('Post Completion' and 'Reply Generation').
    """
    def __init__(self, data_directory: str):
        self.data_dir = Path(data_directory)
        self.cache = get_file_cache()
        self.results: List[BaselineResult] = []
        self.condition_mapping = {
            'no_persona': 'No-Persona',
            'generic': 'Generic-Persona',
            'history_only': 'History-Only',
            'best_persona': 'Best-Persona'
        }

    # FINAL VERSION: Correctly handles nested metric dictionaries using json_normalize
    def _calculate_mean_metrics(self, scores_list: list) -> dict:
        """
        Calculates mean metrics from a list of individual score items.
        This version correctly handles the nested dictionary structure for all metrics.
        """
        if not scores_list:
            return {}

        df = pd.DataFrame(scores_list)
        metrics = {'num_samples': len(df)}

        # Safely calculate mean BERTScore F1
        if 'bertscore' in df.columns:
            try:
                bert_df = pd.json_normalize(df['bertscore'].dropna())
                if 'f1' in bert_df.columns:
                    metrics['bertscore_f1'] = bert_df['f1'].mean()
            except Exception as e:
                logger.warning(f"Could not process 'bertscore' column: {e}")

        # Safely calculate mean ROUGE-L from the nested 'rouge' dictionary
        if 'rouge' in df.columns:
            try:
                rouge_df = pd.json_normalize(df['rouge'].dropna())
                if 'rougeL' in rouge_df.columns:
                    metrics['rouge_l'] = rouge_df['rougeL'].mean()
            except Exception as e:
                logger.warning(f"Could not process 'rouge' column: {e}")
        
        # Safely calculate mean BLEU from the nested 'bleu' dictionary
        if 'bleu' in df.columns:
            try:
                bleu_df = pd.json_normalize(df['bleu'].dropna())
                if 'bleu' in bleu_df.columns:
                    metrics['bleu'] = bleu_df['bleu'].mean()
            except Exception as e:
                logger.warning(f"Could not process 'bleu' column: {e}")
        
        return metrics

    # MODIFIED: Added diagnostic logging to debug data extraction
    def extract_baseline_results(self, run_ids: List[str]) -> None:
        """
        Extracts results by processing the 'individual_scores' list.
        Includes diagnostic logging to help identify data structure issues.
        """
        logger.info(f"Extracting results for {len(run_ids)} baseline runs. Searching for these specific run IDs.")
        run_id_mapping = {run_id: self._get_condition_from_run_id(run_id) for run_id in run_ids}

        for jsonl_file in self.data_dir.glob("*.jsonl"):
            user_id = jsonl_file.stem
            try:
                # 1. Create a lookup map from tweet_id to task type (is_post)
                all_stimuli = load_stimulus(str(jsonl_file))
                task_type_map = {str(tweet_id): is_post for _, is_post, tweet_id in all_stimuli}

                # 2. Load cached evaluation data
                cached_data = self.cache.read_file_with_cache(str(jsonl_file))
                if not cached_data or 3 not in cached_data:
                    continue
                
                evaluations = cached_data[3].get('evaluations', [])
                for evaluation in evaluations:
                    eval_run_id = evaluation.get('run_id', '')

                    # --- DIAGNOSTIC CHECK 1: See if run_id matches ---
                    if eval_run_id in run_id_mapping:
                        logger.info(f"Found matching run_id '{eval_run_id}' in file '{jsonl_file.name}'.")
                        condition = run_id_mapping[eval_run_id]
                        eval_results = evaluation.get('evaluation_results', {})

                        # --- DIAGNOSTIC CHECK 2: See what keys are available ---
                        logger.info(f"  Available keys in evaluation_results: {list(eval_results.keys())}")
                        
                        individual_scores = eval_results.get('individual_scores')

                        if not individual_scores:
                            logger.warning(f"  'individual_scores' key is missing or empty. Cannot process task-specific results for this entry.")
                            continue

                        # 3. Sort individual scores into post/reply groups
                        post_scores = []
                        reply_scores = []
                        for score_item in individual_scores:
                            tweet_id = str(score_item.get('tweet_id'))
                            is_post = task_type_map.get(tweet_id)
                            if is_post is True:
                                post_scores.append(score_item)
                            elif is_post is False:
                                reply_scores.append(score_item)
                        
                        # 4. Calculate mean metrics for each group
                        post_metrics = self._calculate_mean_metrics(post_scores)
                        reply_metrics = self._calculate_mean_metrics(reply_scores)

                        # 5. Append the aggregated, task-specific results
                        if post_metrics:
                            self._append_result(user_id, condition, "Post Completion", eval_run_id, post_metrics)
                        if reply_metrics:
                            self._append_result(user_id, condition, "Reply Generation", eval_run_id, reply_metrics)

            except Exception as e:
                logger.error(f"Error processing {jsonl_file.name}: {e}", exc_info=True)
        
        logger.info(f"Extracted {len(self.results)} aggregated task-specific results.")
        self._validate_results()

    # MODIFIED: Simplified to accept a flat metrics dictionary
    def _append_result(self, user_id, condition, task_type, run_id, metrics):
        """Appends a result from a pre-aggregated metrics dictionary."""
        if metrics.get('num_samples', 0) > 0:
            self.results.append(BaselineResult(
                user_id=user_id,
                condition=condition,
                task_type=task_type,
                run_id=run_id,
                bertscore_f1=metrics.get('bertscore_f1', 0.0),
                rouge_l=metrics.get('rouge_l', 0.0),
                bleu=metrics.get('bleu', 0.0),
                num_samples=metrics.get('num_samples', 0)
            ))
            
    # --- NO CHANGES NEEDED FOR THE FUNCTIONS BELOW THIS LINE ---
    # The rest of the class (statistical tests, plotting, reporting)
    # will now work correctly because the data is being fed to them
    # in the format they expect.

    def _get_condition_from_run_id(self, run_id: str) -> Optional[str]:
        for key in self.condition_mapping.keys():
            if key in run_id:
                return key
        logger.warning(f"Unknown condition for run_id: {run_id}")
        return None

    def _validate_results(self) -> None:
        if not self.results:
            raise ValueError("No baseline results were found for the given run IDs.")
        df = self.create_results_dataframe()
        if len(df['user_id'].unique()) < 2:
            logger.warning("Fewer than 2 users found. Statistical tests may not be meaningful.")

    def create_results_dataframe(self) -> pd.DataFrame:
        data = [{
            'user_id': r.user_id,
            'condition': self.condition_mapping.get(r.condition, r.condition),
            'task_type': r.task_type,
            'bertscore_f1': r.bertscore_f1,
            'rouge_l': r.rouge_l,
            'bleu': r.bleu,
        } for r in self.results]
        return pd.DataFrame(data)

    def calculate_descriptive_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        metrics = ['bertscore_f1', 'rouge_l', 'bleu']
        desc_stats = df.groupby(['condition', 'task_type'])[metrics].agg(
            ['count', 'mean', 'std', 'median']
        ).reset_index()
        desc_stats.columns = ['_'.join(col).strip('_') for col in desc_stats.columns.values]
        return desc_stats

    def perform_statistical_tests(self, df: pd.DataFrame) -> List[StatisticalTest]:
        tests = []
        metrics = ['bertscore_f1', 'rouge_l', 'bleu']
        num_conditions = len(df['condition'].unique())
        num_task_types = len(df['task_type'].unique())
        required_rows = num_conditions * num_task_types
        user_counts = df.groupby('user_id').size()
        complete_users = user_counts[user_counts == required_rows].index
        
        if len(complete_users) < 2:
            logger.warning(f"Not enough users ({len(complete_users)}) with complete data. Skipping inferential tests.")
            return tests
            
        df_complete = df[df['user_id'].isin(complete_users)]
        
        if num_task_types > 1:
            logger.info(f"Found {num_task_types} task types. Running Two-Way Repeated Measures ANOVA on {len(complete_users)} users.")
            for metric in metrics:
                try:
                    aov = pg.rm_anova(data=df_complete, dv=metric, within=['condition', 'task_type'], subject='user_id', detailed=True)
                    for _, row in aov.iterrows():
                        source = row['Source']
                        p_val = row.get('p-GG-corr') if 'spher' in row and not row['spher'] else row.get('p-unc')
                        interpretation = "Significant effect" if p_val < 0.05 else "Not a significant effect"
                        tests.append(StatisticalTest(test_name=f"2-Way ANOVA ({source}) - {metric}", statistic=row.get('F', np.nan), p_value=p_val, effect_size=row.get('ges', np.nan), interpretation=interpretation))
                except Exception as e:
                    logger.error(f"Could not perform Two-Way ANOVA for {metric}: {e}")
        else:
            logger.warning(f"Only 1 task type found. Falling back to One-Way Repeated Measures ANOVA.")
            # Fallback logic here if needed...
            
        logger.info("Running post-hoc pairwise t-tests for conditions.")
        for metric in metrics:
            try:
                df_agg = df_complete.groupby(['user_id', 'condition'])[metric].mean().reset_index()
                posthocs = pg.pairwise_tests(data=df_agg, dv=metric, within='condition', subject='user_id', padjust='bonf')
                for _, row in posthocs.iterrows():
                    effect_size = row.get('cohen-d', np.nan)
                    p_corrected = row.get('p-corr', np.nan)
                    tests.append(StatisticalTest(test_name=f"Paired t-test ({row['A']} vs {row['B']}) - {metric}", statistic=row.get('T', np.nan), p_value=p_corrected, effect_size=effect_size, interpretation=f"Significant after Bonferroni correction" if p_corrected < 0.05 else "Not significant"))
            except Exception as e:
                logger.error(f"Could not perform post-hoc tests for {metric}: {e}")
        return tests

    def create_visualizations(self, df: pd.DataFrame, output_dir: str = "baseline_plots") -> None:
        """Creates task-aware visualizations showing grouped results."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        sns.set_theme(style="whitegrid", palette="viridis")
        metrics = {'bertscore_f1': 'BERTScore F1', 'rouge_l': 'ROUGE-L', 'bleu': 'BLEU'}
        
        order = ['No-Persona', 'Generic-Persona', 'History-Only', 'Best-Persona']

        # Grouped Box plots by condition, faceted by metric
        for metric, label in metrics.items():
            plt.figure(figsize=(12, 7))
            sns.boxplot(data=df, x='condition', y=metric, hue='task_type', order=order)
            plt.title(f'Distribution of {label} by Condition and Task Type', fontsize=16, pad=20)
            plt.xlabel('Condition', fontsize=12)
            plt.ylabel(label, fontsize=12)
            plt.xticks(rotation=15)
            plt.legend(title='Task Type')
            plt.tight_layout()
            plt.savefig(output_path / f'boxplot_{metric}_by_task.png', dpi=300)
            plt.close()

        # Grouped Bar plot for primary metric (BERTScore F1)
        plt.figure(figsize=(12, 8))
        sns.barplot(data=df, x='condition', y='bertscore_f1', hue='task_type', order=order,
                    errorbar='se', capsize=.05)
        plt.title('Mean BERTScore F1 by Condition and Task Type', fontsize=16, pad=20)
        plt.xlabel('Baseline Condition', fontsize=12)
        plt.ylabel('BERTScore F1 (with SE)', fontsize=12)
        plt.xticks(rotation=15)
        plt.legend(title='Task Type')
        plt.tight_layout()
        plt.savefig(output_path / 'barplot_bertscore_by_task.png', dpi=300)
        plt.close()

        logger.info(f"Task-aware visualizations saved to {output_path}")

    def generate_report(self, stats_tests: List[StatisticalTest], descriptive_stats: pd.DataFrame, output_file: str) -> None:
        """Generates a comprehensive, task-aware analysis report."""
        with open(output_file, 'w') as f:
            f.write("TASK-AWARE BASELINE EXPERIMENT ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")

            f.write("### 1. Research Questions ###\n")
            f.write("1. Is there a main effect of persona condition on performance?\n")
            f.write("2. Is there a main effect of task type (e.g., is one task harder)?\n")
            f.write("3. Is there an interaction effect? (i.e., Does the best persona depend on the task?)\n\n")

            f.write("### 2. Descriptive Statistics by Task ###\n")
            f.write(descriptive_stats.to_string())
            
            f.write("\n\n### 3. Inferential Statistics (Two-Way Repeated Measures ANOVA) ###\n")
            for test in stats_tests:
                if "ANOVA" in test.test_name:
                    f.write(f"\n--- {test.test_name} ---\n")
                    f.write(f"  F-statistic: {test.statistic:.4f}\n")
                    f.write(f"  p-value: {test.p_value:.4f}\n")
                    f.write(f"  Effect size (ges): {test.effect_size:.4f}\n")
                    f.write(f"  Interpretation: {test.interpretation}\n")

            f.write("\n\n### 4. Post-Hoc Pairwise Comparisons (Bonferroni Corrected) ###\n")
            for test in stats_tests:
                if "t-test" in test.test_name:
                    f.write(f"\n--- {test.test_name} ---\n")
                    f.write(f"  t-statistic: {test.statistic:.4f}\n")
                    f.write(f"  p-value (corrected): {test.p_value:.4f}\n")
                    f.write(f"  Effect size (Cohen's d): {test.effect_size:.4f}\n")
                    f.write(f"  Interpretation: {test.interpretation}\n")
            
            f.write("\n\n### 5. Key Findings & Interpretation ###\n")
            interaction_test = next((t for t in stats_tests if 'condition * task_type' in t.test_name and 'bertscore_f1' in t.test_name), None)
            if interaction_test:
                f.write("Primary Metric: BERTScore F1\n")
                if interaction_test.p_value < 0.05:
                    f.write(f"  - CRITICAL FINDING: A significant interaction effect was found (p={interaction_test.p_value:.4f}).\n")
                    f.write("    This means the effectiveness of the persona conditions DEPENDS on the task.\n")
                    f.write("    For example, 'Best-Persona' might be significantly better for replies but not for posts.\n")
                    f.write("    You must analyze the simple effects within each task to understand this relationship fully.\n")
                else:
                    f.write(f"  - KEY FINDING: No significant interaction effect was found (p={interaction_test.p_value:.4f}).\n")
                    f.write("    This suggests that the effect of the persona conditions is consistent across both Post Completion and Reply Generation tasks.\n")
                    f.write("    If there is a main effect of condition, the ranking of personas (e.g., Best > Generic > None) holds for both tasks.\n")

        logger.info(f"Task-aware analysis report saved to {output_file}")
    
    def run_complete_analysis(self, run_ids: List[str], output_dir: str = "baseline_analysis") -> None:
        """Runs the complete task-aware analysis pipeline with pre-test validation."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        logger.info("Starting comprehensive task-aware baseline analysis")
        self.extract_baseline_results(run_ids)
        if not self.results:
            logger.error("Analysis halted: No results found for the specified run IDs.")
            return
        
        df = self.create_results_dataframe()

        # --- PRE-TEST VALIDATION ---
        unique_tasks = df['task_type'].unique()
        if len(unique_tasks) <= 1 and "Overall" in unique_tasks:
            logger.error("="*80)
            logger.error("ANALYSIS HALTED: Task-specific data could not be found.")
            logger.error("All results were loaded as 'Overall'. This prevents a meaningful task-aware analysis.")
            logger.error("Please check the keys in your evaluation data (e.g., ensure 'post_completion' exists).")
            logger.error("="*80)
            # Save the limited data for debugging
            df.to_csv(output_path / "debug_overall_results.csv", index=False)
            return
        # --- END OF VALIDATION ---

        descriptive_stats = self.calculate_descriptive_statistics(df)
        statistical_tests = self.perform_statistical_tests(df)
        
        self.create_visualizations(df, str(output_path / "plots"))
        self.generate_report(statistical_tests, descriptive_stats, str(output_path / "analysis_report.txt"))
        
        df.to_csv(output_path / "baseline_results_by_task.csv", index=False)
        descriptive_stats.to_csv(output_path / "descriptive_statistics_by_task.csv", index=False)
        
        logger.info(f"Complete analysis saved to {output_path}")
        
        print("\n--- BASELINE ANALYSIS SUMMARY (BERTScore F1) ---")
        summary = df.groupby(['condition', 'task_type'])['bertscore_f1'].mean().unstack()
        print(summary.to_string(float_format="%.4f"))
        
def main():
    """Main function for running baseline analysis."""
    parser = argparse.ArgumentParser(description="Analyze baseline experiment results with task-aware methods.")
    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing user JSONL result files')
    parser.add_argument('--run-ids', nargs='+', required=True, help='List of baseline run IDs to analyze')
    parser.add_argument('--output', type=str, default='baseline_analysis_v3', help='Output directory for analysis results')
    args = parser.parse_args()
    
    analyzer = BaselineAnalyzer(args.data_dir)
    analyzer.run_complete_analysis(args.run_ids, args.output)

if __name__ == "__main__":
    # Example usage from command line:
    # python your_script_name.py --data-dir /path/to/data \
    # --run-ids run_no_persona_123 run_generic_123 run_history_only_123 run_best_persona_123 \
    # --output my_analysis_results
    main()