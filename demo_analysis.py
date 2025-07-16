"""
Simplified Data Analysis Framework for Twitter Persona Imitation

A streamlined version that removes redundancy and focuses on essential functionality.
"""

import json
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class Tweet:
    """Tweet with essential fields only."""
    tweet_id: str
    text: str
    created_at: str
    user_id: str
    is_reply: bool = False
    is_retweet: bool = False
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Tweet':
        return cls(
            tweet_id=str(data.get('tweet_id', '')),
            text=data.get('full_text', ''),
            created_at=data.get('created_at', ''),
            user_id=str(data.get('original_user_id', '')),
            is_reply=bool(data.get('reply_to_id')),
            is_retweet=bool(data.get('retweeted_user_ID'))
        )


@dataclass
class Evaluation:
    """Evaluation result for a single round."""
    run_id: str
    timestamp: str
    round_num: int
    overall_scores: Dict[str, float]  # bleu, rouge1, rouge2, rougeL, combined
    individual_scores: List[Dict[str, Any]]
    total_samples: int
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Evaluation':
        eval_results = data.get('evaluation_results', {})
        overall = eval_results.get('overall', {})
        
        # Extract ROUGE scores
        rouge_scores = overall.get('rouge', {})
        rouge1 = rouge_scores.get('rouge1', 0.0)
        rouge2 = rouge_scores.get('rouge2', 0.0)
        rougeL = rouge_scores.get('rougeL', 0.0)
        
        # Extract BLEU score
        bleu_scores = overall.get('bleu', {})
        bleu = bleu_scores.get('bleu', 0.0)
        
        # Calculate combined score
        combined = (rouge1 + rouge2 + rougeL + bleu) / 4.0
        
        return cls(
            run_id=data.get('run_id', ''),
            timestamp=data.get('timestamp', ''),
            round_num=data.get('round', 0),
            overall_scores={
                'bleu': float(bleu),
                'rouge1': float(rouge1),
                'rouge2': float(rouge2),
                'rougeL': float(rougeL),
                'combined': float(combined)
            },
            individual_scores=eval_results.get('individual_scores', []),
            total_samples=overall.get('total_samples', 0)
        )


@dataclass
class Run:
    """A single run with persona and imitations."""
    run_id: str
    user_id: str
    persona: str
    imitations: List[Dict[str, Any]]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], user_id: str) -> 'Run':
        return cls(
            run_id=data.get('run_id', ''),
            user_id=user_id,
            persona=data.get('persona', ''),
            imitations=data.get('imitations', [])
        )


@dataclass
class UserData:
    """Complete user data."""
    user_id: str
    historical_tweets: List[Tweet]
    holdout_tweets: List[Tweet]
    runs: List[Run]
    evaluations: List[Evaluation]


class DataLoader:
    """Simplified data loader."""
    
    def load_jsonl(self, file_path: str) -> Dict[int, Dict[str, Any]]:
        """Load JSONL file."""
        data = {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if line.strip():
                        try:
                            data[line_num] = json.loads(line)
                        except json.JSONDecodeError as e:
                            logger.error(f"Error parsing line {line_num + 1}: {e}")
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
        return data
    
    def load_user_data(self, file_path: str) -> Optional[UserData]:
        """Load complete user data."""
        data = self.load_jsonl(file_path)
        if not data:
            return None
        
        # Extract user_id
        user_id = None
        for line_data in data.values():
            if 'user_id' in line_data:
                user_id = str(line_data['user_id'])
                break
        
        if not user_id:
            return None
        
        # Load historical tweets (line 0)
        historical_tweets = []
        if 0 in data and data[0].get('set') == 'history':
            historical_tweets = [Tweet.from_dict(tweet) for tweet in data[0].get('tweets', [])]
        
        # Load holdout tweets (line 1)
        holdout_tweets = []
        if 1 in data and data[1].get('set') == 'holdout':
            holdout_tweets = [Tweet.from_dict(tweet) for tweet in data[1].get('tweets', [])]
        
        # Load runs (line 2)
        runs = []
        if 2 in data and 'runs' in data[2]:
            runs = [Run.from_dict(run_data, user_id) for run_data in data[2]['runs']]
        
        # Load evaluations (line 3)
        evaluations = []
        if 3 in data and 'evaluations' in data[3]:
            evaluations = [Evaluation.from_dict(eval_data) for eval_data in data[3]['evaluations']]
        
        return UserData(
            user_id=user_id,
            historical_tweets=historical_tweets,
            holdout_tweets=holdout_tweets,
            runs=runs,
            evaluations=evaluations
        )


class RunAnalyzer:
    """Main analyzer class."""
    
    def __init__(self, data_dir: str = "data/filtered_users"):
        self.data_dir = Path(data_dir)
        self.loader = DataLoader()
    
    def get_available_runs(self) -> List[str]:
        """Get all available run IDs."""
        run_ids = set()
        for file_path in self.data_dir.glob("*.jsonl"):
            data = self.loader.load_jsonl(str(file_path))
            if 2 in data and 'runs' in data[2]:
                for run in data[2]['runs']:
                    if run.get('run_id'):
                        run_ids.add(run['run_id'])
        return sorted(list(run_ids))
    
    def load_run_data(self, run_id: str) -> Dict[str, UserData]:
        """Load all user data for a specific run."""
        users = {}
        for file_path in self.data_dir.glob("*.jsonl"):
            user_data = self.loader.load_user_data(str(file_path))
            if user_data and any(run.run_id == run_id for run in user_data.runs):
                users[user_data.user_id] = user_data
        return users
    
    def get_run_stats(self, run_id: str) -> Dict[str, Any]:
        """Get run statistics across all rounds."""
        users = self.load_run_data(run_id)
        if not users:
            return {}
        
        # Collect all evaluations for this run across all rounds
        all_evaluations = []
        total_imitations = 0
        
        for user_data in users.values():
            # Count imitations for this run
            for run in user_data.runs:
                if run.run_id == run_id:
                    total_imitations += len(run.imitations)
            
            # Get all evaluations for this run (across all rounds)
            for evaluation in user_data.evaluations:
                if evaluation.run_id == run_id:
                    all_evaluations.append(evaluation)
        
        if not all_evaluations:
            return {
                'run_id': run_id,
                'users': len(users),
                'total_imitations': total_imitations,
                'total_rounds': 0,
                'avg_scores': {}
            }
        
        # Calculate averages across all rounds
        avg_scores = {}
        for metric in ['bleu', 'rouge1', 'rouge2', 'rougeL', 'combined']:
            scores = [eval.overall_scores[metric] for eval in all_evaluations if metric in eval.overall_scores]
            avg_scores[metric] = sum(scores) / len(scores) if scores else 0.0
        
        # Get round statistics
        rounds = [eval.round_num for eval in all_evaluations]
        total_rounds = max(rounds) if rounds else 0
        
        return {
            'run_id': run_id,
            'users': len(users),
            'total_imitations': total_imitations,
            'total_rounds': total_rounds,
            'total_evaluations': len(all_evaluations),
            'avg_scores': avg_scores,
            'rounds_analyzed': sorted(list(set(rounds)))
        }
    
    def get_user_performance(self, user_id: str, run_id: str) -> Dict[str, Any]:
        """Get performance for a specific user across all rounds."""
        file_path = self.data_dir / f"{user_id}.jsonl"
        if not file_path.exists():
            return {}
        
        user_data = self.loader.load_user_data(str(file_path))
        if not user_data:
            return {}
        
        # Find the run
        target_run = None
        for run in user_data.runs:
            if run.run_id == run_id:
                target_run = run
                break
        
        if not target_run:
            return {}
        
        # Get all evaluations for this run
        run_evaluations = [eval for eval in user_data.evaluations if eval.run_id == run_id]
        
        # Calculate average scores across all rounds
        avg_scores = {}
        if run_evaluations:
            for metric in ['bleu', 'rouge1', 'rouge2', 'rougeL', 'combined']:
                scores = [eval.overall_scores[metric] for eval in run_evaluations if metric in eval.overall_scores]
                avg_scores[metric] = sum(scores) / len(scores) if scores else 0.0
        
        return {
            'user_id': user_id,
            'run_id': run_id,
            'persona': target_run.persona,
            'imitations': len(target_run.imitations),
            'total_rounds': len(run_evaluations),
            'avg_scores': avg_scores,
            'round_scores': [
                {
                    'round': eval.round_num,
                    'scores': eval.overall_scores,
                    'samples': eval.total_samples
                } for eval in sorted(run_evaluations, key=lambda x: x.round_num)
            ]
        }
    
    def export_to_csv(self, run_id: str, output_path: str = None) -> str:
        """Export run data to CSV with round-by-round breakdown."""
        users = self.load_run_data(run_id)
        
        rows = []
        for user_data in users.values():
            # Get run info
            target_run = None
            for run in user_data.runs:
                if run.run_id == run_id:
                    target_run = run
                    break
            
            if not target_run:
                continue
            
            # Get evaluations for this run
            run_evaluations = [eval for eval in user_data.evaluations if eval.run_id == run_id]
            
            # Create row for each round
            for evaluation in run_evaluations:
                row = {
                    'user_id': user_data.user_id,
                    'run_id': run_id,
                    'round': evaluation.round_num,
                    'persona_length': len(target_run.persona),
                    'imitations': len(target_run.imitations),
                    'total_samples': evaluation.total_samples,
                    'timestamp': evaluation.timestamp
                }
                row.update(evaluation.overall_scores)
                rows.append(row)
        
        df = pd.DataFrame(rows)
        output_path = output_path or f"run_{run_id}.csv"
        df.to_csv(output_path, index=False)
        return output_path


# Simple usage example
def main(run_id: str = None):
    analyzer = RunAnalyzer()
    
    # Get available runs
    runs = analyzer.get_available_runs()
    print(f"Available runs: {runs}")
    
    if not runs:
        print("No runs found!")
        return
    
    # Use provided run_id or default to latest
    if run_id and run_id in runs:
        target_run = run_id
    elif run_id:
        print(f"Run {run_id} not found. Using latest run.")
        target_run = runs[-1]
    else:
        target_run = runs[-1]  # Use latest run
    
    print(f"\nAnalyzing run: {target_run}")
    
    # Get run statistics
    stats = analyzer.get_run_stats(target_run)
    print(f"\nRun {target_run} stats:")
    print(f"Users: {stats['users']}")
    print(f"Total imitations: {stats['total_imitations']}")
    print(f"Average scores: {stats.get('avg_scores', {})}")
    
    # Export to CSV
    csv_file = analyzer.export_to_csv(target_run)
    print(f"Data exported to: {csv_file}")


if __name__ == "__main__":
    import sys
    
    # Check if run_id provided as command line argument
    run_id= '20250702_114023'
    main(run_id)
   