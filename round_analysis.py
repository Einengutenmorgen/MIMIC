import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from loader import get_file_cache
from logging_config import logger


class RoundAnalyzer:
    """
    Analyzes performance progression across rounds in the iterative self-prompting pipeline
    """
    
    def __init__(self, users_directory: str = "data/filtered_users"):
        self.users_directory = users_directory
        self.cache = get_file_cache()
        
    def extract_round_data(self, file_path: str) -> Dict:
        """
        Extract all round data from a user file
        
        :param file_path: Path to user JSONL file
        :return: Dictionary with round data
        """
        try:
            cached_data = self.cache.read_file_with_cache(file_path)
            if cached_data is None:
                logger.error(f"Failed to read file {file_path}")
                return {}
            
            # Get user_id from first line
            user_id = cached_data[0].get('user_id') if 0 in cached_data else None
            if not user_id:
                logger.error(f"No user_id found in {file_path}")
                return {}
            
            # Extract runs from line 3 (index 2) and evaluations from line 4 (index 3)
            runs_data = cached_data.get(2, {}).get('runs', [])
            evaluations_data = cached_data.get(3, {}).get('evaluations', [])

            # Create a dictionary to hold evaluation results keyed by run_id
            eval_results_map = {
                eval_entry.get('run_id'): eval_entry.get('evaluation_results', {})
                for eval_entry in evaluations_data
            }

            # Group by rounds, merging run and evaluation data
            round_data = {}
            for run_entry in runs_data:
                run_id = run_entry.get('run_id', '')
                if '_round_' in run_id:
                    try:
                        round_num = int(run_id.split('_round_')[1])
                        # Merge run info with corresponding evaluation results
                        if run_id in eval_results_map:
                            round_data[round_num] = {
                                'user_id': user_id,
                                'round': round_num,
                                'run_id': run_id,
                                'timestamp': run_entry.get('timestamp'),
                                'evaluation_results': eval_results_map[run_id]
                            }
                        else:
                            logger.warning(f"No evaluation found for run_id: {run_id}")
                    except (ValueError, IndexError):
                        logger.warning(f"Could not parse round number from run_id: {run_id}")
                        continue
            
            return round_data
            
        except Exception as e:
            logger.error(f"Error extracting round data from {file_path}: {e}")
            return {}
    
    def analyze_all_users(self, base_run_id: Optional[str] = None) -> pd.DataFrame:
        """
        Analyze performance progression for all users
        
        :param base_run_id: Optional base run_id to filter results
        :return: DataFrame with all round results
        """
        all_results = []
        
        # Get all user files
        user_files = [f for f in os.listdir(self.users_directory) 
                     if f.endswith('.jsonl')]
        
        for user_file in user_files:
            file_path = os.path.join(self.users_directory, user_file)
            round_data = self.extract_round_data(file_path)
            
            for round_num, data in round_data.items():
                # Filter by base_run_id if provided
                if base_run_id and not data['run_id'].startswith(base_run_id):
                    continue
                
                # Restructure the data to match the expected format in final_analysis.py
                eval_results = data.get('evaluation_results', {})
                
                # The structure in final_analysis.py expects nested dictionaries
                # for 'statistics', 'overall', and 'individual_scores'.
                # We will reconstruct this structure here.
                
                result_row = {
                    'user_id': data['user_id'],
                    'round': round_num,
                    'run_id': data['run_id'],
                    'timestamp': data['timestamp'],
                    'statistics': eval_results.get('statistics', {}),
                    'overall': eval_results.get('overall', {}),
                    'individual_scores': eval_results.get('individual_scores', [])
                }
                
                all_results.append(result_row)
        
        return pd.DataFrame(all_results)
    
    def get_aggregated_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get aggregated results across all users for each round
        
        :param df: DataFrame with individual results
        :return: DataFrame with aggregated results
        """
        # Get numeric columns (metrics)
        numeric_cols = df.select_dtypes(include=['number']).columns
        metric_cols = [col for col in numeric_cols if col != 'round']
        
        # Group by round and calculate statistics
        aggregated = df.groupby('round')[metric_cols].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).round(4)
        
        # Flatten column names
        aggregated.columns = [f"{metric}_{stat}" for metric, stat in aggregated.columns]
        
        return aggregated.reset_index()
    
    def plot_progression(self, df: pd.DataFrame, metrics: List[str] = None):
        """
        Plot performance progression across rounds
        
        :param df: DataFrame with results
        :param metrics: List of metrics to plot (default: all numeric metrics)
        """
        if df.empty:
            print("No data to plot")
            return
        
        # Get metrics to plot
        if metrics is None:
            numeric_cols = df.select_dtypes(include=['number']).columns
            metrics = [col for col in numeric_cols if col != 'round']
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics[:4]):  # Plot first 4 metrics
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Individual user lines (light)
            for user_id in df['user_id'].unique():
                user_data = df[df['user_id'] == user_id].sort_values('round')
                ax.plot(user_data['round'], user_data[metric], 
                       alpha=0.3, color='gray', linewidth=0.5)
            
            # Aggregated mean line (bold)
            round_means = df.groupby('round')[metric].mean()
            ax.plot(round_means.index, round_means.values, 
                   color='red', linewidth=2, marker='o', label='Mean')
            
            ax.set_title(f'{metric.upper()} Progression')
            ax.set_xlabel('Round')
            ax.set_ylabel(metric.upper())
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(len(metrics), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, base_run_id: Optional[str] = None):
        """
        Generate a comprehensive report of round progression
        
        :param base_run_id: Optional base run_id to filter results
        """
        print("=" * 60)
        print("ITERATIVE SELF-PROMPTING PIPELINE ANALYSIS")
        print("=" * 60)
        
        # Get all results
        df = self.analyze_all_users(base_run_id)
        
        if df.empty:
            print("No round data found!")
            return
        
        print(f"\nFound data for {df['user_id'].nunique()} users across {df['round'].nunique()} rounds")
        print(f"Total evaluations: {len(df)}")
        
        # Show rounds available
        rounds = sorted(df['round'].unique())
        print(f"Rounds available: {rounds}")
        
        # Get aggregated results
        aggregated = self.get_aggregated_results(df)
        
        print("\n" + "=" * 60)
        print("AGGREGATED RESULTS BY ROUND")
        print("=" * 60)
        print(aggregated.to_string(index=False))
        
        # Show improvement trends
        print("\n" + "=" * 60)
        print("IMPROVEMENT TRENDS")
        print("=" * 60)
        
        # Get numeric metrics
        numeric_cols = df.select_dtypes(include=['number']).columns
        metrics = [col for col in numeric_cols if col != 'round']
        
        for metric in metrics:
            round_means = df.groupby('round')[metric].mean()
            if len(round_means) > 1:
                first_round = round_means.iloc[0]
                last_round = round_means.iloc[-1]
                change = last_round - first_round
                change_pct = (change / first_round) * 100 if first_round != 0 else 0
                
                trend = "↑" if change > 0 else "↓" if change < 0 else "→"
                print(f"{metric.upper():20} {trend} {change:+.4f} ({change_pct:+.1f}%)")
        
        # Plot progression
        print("\nGenerating progression plots...")
        self.plot_progression(df, metrics)
        
        return df, aggregated


def main():
    """
    Main function to run the analysis
    """
    analyzer = RoundAnalyzer()
    
    # You can specify a base_run_id to filter results
    # For example: analyzer.generate_report("20250709_140000")
    df, aggregated = analyzer.generate_report()
    
    # Save results to CSV for further analysis
    if df is not None and not df.empty:
        df.to_csv('round_analysis_individual.csv', index=False)
        aggregated.to_csv('round_analysis_aggregated.csv', index=False)
        print("\nResults saved to:")
        print("- round_analysis_individual.csv")
        print("- round_analysis_aggregated.csv")


if __name__ == "__main__":
    main()