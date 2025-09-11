#!/usr/bin/env python3
"""
Baseline Experiments Main Pipeline

Complete end-to-end pipeline for running baseline experiments:
- No-Persona vs Generic-Persona comparison
- Within-subject design with balanced holdout splits
- Integration with existing evaluation infrastructure

Usage:
    python baseline_main.py --config baseline_config.yaml
    python baseline_main.py --config config.yaml --baseline-only
"""

import os
import argparse
import sys
from datetime import datetime
from pathlib import Path

# Import existing infrastructure
from baseline_experiments import BaselineExperiment, create_baseline_config
from utils import load_config
from logging_config import logger


def create_baseline_config_file(output_path: str = "baseline_config.yaml") -> str:
    """
    Create a baseline-specific configuration file.
    
    Args:
        output_path: Path where to save the baseline config
        
    Returns:
        Path to created config file
    """
    baseline_config_content = """# Baseline Experiment Configuration
# Based on main config.yaml but optimized for baseline comparisons

experiment:
  run_name_prefix: 'baseline_test'
  users_dict: "data/filtered_users_without_links_meta/filtered_users_without_links" 
  number_of_users: 2  # Use all 24 users for statistical power
  number_of_rounds: 1  # Baselines don't need iterative improvement
  num_stimuli_to_process: null  # Process all available stimuli for complete comparison

  # Baseline-specific settings
  baseline_conditions: ["no_persona", "generic_persona"]
  balanced_split_seed: 42  # For reproducible experiments
  
  # Parallel processing (same as main experiment)
  use_async: true 
  num_workers: 6

llm:
  persona_model: "google"      # Not used for no_persona condition
  imitation_model: "ollama"    # Same model for fair comparison
  reflection_model: "google_json"  # Not used in baseline (no iteration)
  ollama_model: "gemma3:latest"

templates:
  # Baseline-specific templates
  no_persona_post_template: "imitation_post_template_no_persona"
  no_persona_reply_template: "imitation_replies_template_no_persona"
  generic_post_template: "imitation_post_template_generic"
  generic_reply_template: "imitation_replies_template_generic"
  
  # Original templates (for reference)
  persona_template: "persona_template_simple"
  imitation_post_template: "imitation_post_template_simple"
  imitation_reply_template: "imitation_replies_template_simple"

# Evaluation settings (same as main experiment)
evaluation:
  metrics: ["bleu", "rouge_1", "rouge_2", "rouge_l"]
  save_individual_scores: true
  save_best_worst_predictions: true
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(baseline_config_content)
    
    logger.info(f"Created baseline config file: {output_path}")
    return output_path


def run_baseline_experiment(config_path: str = "baseline_config.yaml") -> dict:
    """
    Run complete baseline experiment.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary with experiment results
    """
    logger.info("="*60)
    logger.info("STARTING BASELINE EXPERIMENT")
    logger.info("="*60)
    
    start_time = datetime.now()
    
    # Load configuration
    try:
        config = load_config(config_path)
        logger.info(f"Loaded configuration from: {config_path}")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {"error": f"Config loading failed: {e}"}
    
    # Validate configuration
    experiment_config = config.get('experiment', {})
    required_keys = ['users_dict', 'number_of_users']
    
    for key in required_keys:
        if key not in experiment_config:
            logger.error(f"Missing required config key: experiment.{key}")
            return {"error": f"Missing config key: {key}"}
    
    users_dict_path = experiment_config['users_dict']
    if not os.path.exists(users_dict_path):
        logger.error(f"Users directory not found: {users_dict_path}")
        return {"error": f"Users directory not found: {users_dict_path}"}
    
    # Initialize baseline experiment manager
    baseline_exp = BaselineExperiment(users_dict_path)
    
    # Log experiment parameters
    logger.info(f"Experiment Parameters:")
    logger.info(f"  Users directory: {users_dict_path}")
    logger.info(f"  Number of users: {experiment_config.get('number_of_users', 'all')}")
    logger.info(f"  LLM model: {config.get('llm', {}).get('imitation_model', 'unknown')}")
    logger.info(f"  Parallel workers: {experiment_config.get('num_workers', 1)}")
    logger.info(f"  Split seed: {experiment_config.get('balanced_split_seed', 42)}")
    
    # Run baseline experiment
    try:
        results = baseline_exp.run_baseline_experiment(config)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        # Log results
        logger.info("="*60)
        logger.info("BASELINE EXPERIMENT COMPLETED")
        logger.info("="*60)
        logger.info(f"Duration: {duration}")
        logger.info(f"Successful users: {len(results.get('processed_users', []))}")
        logger.info(f"Failed users: {len(results.get('failed_users', []))}")
        logger.info(f"Run IDs generated:")
        
        for condition, run_id in results.get('run_ids', {}).items():
            logger.info(f"  {condition}: {run_id}")
        
        # Add timing information to results
        results.update({
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(), 
            "duration_seconds": duration.total_seconds(),
            "config_used": config_path
        })
        
        return results
        
    except Exception as e:
        logger.error(f"Baseline experiment failed: {e}")
        return {"error": f"Experiment failed: {e}"}


def validate_baseline_results(results: dict) -> dict:
    """
    Validate that baseline experiment produced valid results.
    
    Args:
        results: Results dictionary from run_baseline_experiment
        
    Returns:
        Validation report dictionary
    """
    validation = {
        "is_valid": True,
        "issues": [],
        "statistics": {}
    }
    
    if "error" in results:
        validation["is_valid"] = False
        validation["issues"].append(f"Experiment error: {results['error']}")
        return validation
    
    # Check if we have run IDs
    run_ids = results.get('run_ids', {})
    if not run_ids:
        validation["is_valid"] = False
        validation["issues"].append("No run IDs generated")
        return validation
    
    # Check for both baseline conditions
    expected_conditions = ["no_persona", "generic_persona"]
    for condition in expected_conditions:
        if condition not in run_ids:
            validation["is_valid"] = False
            validation["issues"].append(f"Missing {condition} condition")
    
    # Check processing statistics
    processed_users = results.get('processed_users', [])
    failed_users = results.get('failed_users', [])
    total_users = results.get('total_users', 0)
    
    success_rate = len(processed_users) / total_users if total_users > 0 else 0
    
    validation["statistics"] = {
        "total_users": total_users,
        "processed_users": len(processed_users),
        "failed_users": len(failed_users),
        "success_rate": success_rate
    }
    
    # Warn if success rate is low
    if success_rate < 0.8:
        validation["issues"].append(f"Low success rate: {success_rate:.1%}")
    
    # Check duration
    duration = results.get('duration_seconds', 0)
    if duration == 0:
        validation["issues"].append("No duration information")
    
    validation["statistics"]["duration_minutes"] = duration / 60
    
    logger.info("Baseline results validation completed")
    logger.info(f"Valid: {validation['is_valid']}")
    logger.info(f"Success rate: {success_rate:.1%}")
    logger.info(f"Duration: {duration/60:.1f} minutes")
    
    if validation["issues"]:
        logger.warning(f"Validation issues: {validation['issues']}")
    
    return validation


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Run baseline experiments for persona imitation comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default baseline config
  python baseline_main.py
  
  # Run with custom config
  python baseline_main.py --config my_baseline_config.yaml
  
  # Create baseline config and run
  python baseline_main.py --create-config --config baseline_config.yaml
  
  # Validate existing baseline results
  python baseline_main.py --validate-only --run-ids baseline_no_persona_20250911,baseline_generic_20250911
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="baseline_config.yaml",
        help="Path to configuration file (default: baseline_config.yaml)"
    )
    
    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Create a baseline configuration file"
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true", 
        help="Only validate results without running experiment"
    )
    
    parser.add_argument(
        "--run-ids",
        type=str,
        help="Comma-separated run IDs for validation (used with --validate-only)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform dry run without calling LLM"
    )
    
    args = parser.parse_args()
    
    try:
        # Create config if requested
        if args.create_config:
            config_path = create_baseline_config_file(args.config)
            print(f"✅ Created baseline config: {config_path}")
            
            if not args.validate_only:
                print("📝 Review the config file and run again to execute experiment")
                return 0
        
        # Validate only mode
        if args.validate_only:
            if not args.run_ids:
                print("❌ --run-ids required for validation-only mode")
                return 1
            
            run_ids = args.run_ids.split(',')
            print(f"🔍 Validating run IDs: {run_ids}")
            
            # TODO: Implement validation logic for specific run IDs
            print("⚠️ Validation-only mode not yet implemented")
            return 0
        
        # Check if config exists
        if not os.path.exists(args.config):
            print(f"❌ Config file not found: {args.config}")
            print("💡 Use --create-config to create a baseline configuration")
            return 1
        
        # Dry run mode
        if args.dry_run:
            print("🔍 DRY RUN MODE - No LLM calls will be made")
            print("⚠️ Dry run mode not yet implemented")
            return 0
        
        # Run baseline experiment
        print(f"🚀 Starting baseline experiment with config: {args.config}")
        results = run_baseline_experiment(args.config)
        
        # Validate results
        validation = validate_baseline_results(results)
        
        if validation["is_valid"]:
            print("✅ Baseline experiment completed successfully!")
            print(f"📊 Processed {validation['statistics']['processed_users']} users")
            print(f"⏱️ Duration: {validation['statistics']['duration_minutes']:.1f} minutes")
            
            if "run_ids" in results:
                print("🔍 Generated run IDs:")
                for condition, run_id in results["run_ids"].items():
                    print(f"   {condition}: {run_id}")
            
            print("\n📈 Next steps:")
            print("1. Use round_analysis.py to analyze results")
            print("2. Compare No-Persona vs Generic-Persona performance")
            print("3. Run statistical tests for significance")
            
            return 0
        else:
            print("❌ Baseline experiment completed with issues:")
            for issue in validation["issues"]:
                print(f"   - {issue}")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️ Experiment interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())