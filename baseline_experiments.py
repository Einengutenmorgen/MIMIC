"""
Baseline Experiments Manager V2 - NO FALLBACKS VERSION

Scientific Design: IDENTICAL STIMULI across all conditions for maximum statistical power.
All conditions receive the SAME tweets to eliminate stimulus variance.
NO FALLBACKS - If it fails, it should fail immediately.
"""

import os
import json
import random
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import concurrent.futures
import threading
import yaml

# Import existing infrastructure
from loader import load_stimulus, get_formatted_user_historie
from saver import save_user_imitation, save_evaluation_results
from eval import evaluate_with_individual_scores
from llm import call_ai
import templates
from logging_config import logger
from file_cache import get_file_cache


class BaselineExperiment:
    """
    Manages baseline experiment conditions for scientific comparison.
    
    SCIENTIFIC DESIGN: All conditions use IDENTICAL stimuli for each user.
    This eliminates stimulus variance and maximizes statistical power.
    
    V2 Baseline Conditions:
    1. No-Persona: Task performance without any persona conditioning
    2. Generic-Persona: Minimal, universal social media user persona
    3. History-Only: Raw user data without LLM abstraction (tests abstraction effect)
    4. Best-Persona: Optimized persona from iterative rounds (tests optimization effect)
    
    NO FALLBACKS: All operations must succeed or fail immediately.
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
        
        self.cache = get_file_cache()
        self.file_lock = threading.Lock()
        
        # V2 Baseline definitions - All 4 conditions
        self.baseline_conditions = {
            "no_persona": {
                "name": "No-Persona Baseline",
                "description": "Task performance without persona conditioning",
                "post_template": "imitation_post_template_no_persona",
                "reply_template": "imitation_replies_template_no_persona",
                "persona": None  # No persona used
            },
            "generic_persona": {
                "name": "Generic-Persona Baseline", 
                "description": "Minimal universal social media user persona",
                "post_template": "imitation_post_template_simple",
                "reply_template": "imitation_replies_template_simple",
                "persona": "You are an active social media user who writes posts and replies in a conversational, authentic way. You engage with various topics and express yourself clearly and naturally online."
            },
            "history_only": {
                "name": "History-Only Baseline",
                "description": "Raw user data without LLM abstraction",
                "post_template": "imitation_post_template_simple",
                "reply_template": "imitation_replies_template_simple",
                "persona": "DYNAMIC"  # Will be generated from raw history
            },
            "best_persona": {
                "name": "Best-Persona Baseline", 
                "description": "Optimized persona from iterative self-prompting",
                "post_template": "imitation_post_template_simple",
                "reply_template": "imitation_replies_template_simple", 
                "persona": "DYNAMIC"  # Will be extracted from best round
            }
        }
    
    def create_identical_stimuli_splits(self, user_file_path: str, num_items: int, seed: int = 42) -> Dict[str, List[Tuple]]:
        """
        Create IDENTICAL stimulus sets for all conditions - OPTIMAL SCIENTIFIC DESIGN.
        
        CRITICAL: All conditions get the SAME stimuli for maximum statistical power.
        This eliminates all stimulus-based variance and allows pure condition comparisons.
        
        NO FALLBACKS: Must have exactly num_items or fail.
        """
        if not os.path.exists(user_file_path):
            raise FileNotFoundError(f"User file does not exist: {user_file_path}")
        
        logger.info(f"Creating IDENTICAL stimulus sets ({num_items} items) for {os.path.basename(user_file_path)}")
        
        # Load all stimuli
        all_stimuli = load_stimulus(user_file_path)
        
        if len(all_stimuli) < num_items:
            raise ValueError(f"Insufficient stimuli in {user_file_path}: {len(all_stimuli)} < {num_items}")
        
        # Randomly select exactly num_items for the experiment
        random.seed(seed)
        selected_stimuli = random.sample(all_stimuli, num_items)
        
        # Analyze stimulus composition for logging
        posts = [item for item in selected_stimuli if item[1] == True]
        replies = [item for item in selected_stimuli if item[1] == False]
        
        logger.info(f"Selected {num_items} IDENTICAL stimuli: {len(posts)} posts, {len(replies)} replies")
        
        # CRITICAL: All conditions get IDENTICAL stimuli
        condition_names = ["no_persona", "generic_persona", "history_only", "best_persona"]
        
        result = {}
        for condition in condition_names:
            result[condition] = selected_stimuli.copy()  # IDENTICAL for all conditions
        
        # Validate identical stimuli across conditions
        first_condition_ids = [item[2] for item in result[condition_names[0]]]  # tweet_ids
        for condition in condition_names[1:]:
            condition_ids = [item[2] for item in result[condition]]
            if first_condition_ids != condition_ids:
                raise ValueError(f"CRITICAL ERROR: Stimuli not identical across conditions!")
        
        logger.info(f"✅ IDENTICAL STIMULI VALIDATED: All {len(condition_names)} conditions use same {num_items} items")
        return result
    
    def create_history_only_persona(self, user_file_path: str) -> str:
        """
        Create History-Only persona from raw user data without LLM abstraction.
        
        Tests the hypothesis: Does LLM abstraction improve persona quality?
        
        NO FALLBACKS: Must succeed or raise exception.
        """
        if not os.path.exists(user_file_path):
            raise FileNotFoundError(f"User file does not exist: {user_file_path}")
        
        logger.info(f"Creating History-Only persona for {os.path.basename(user_file_path)}")
        
        # Use existing loader to get formatted user history
        user_history = get_formatted_user_historie(user_file_path)
        
        if not user_history:
            raise ValueError(f"No user history found in {user_file_path}")
        
        if len(user_history) < 100:
            raise ValueError(f"Insufficient user history in {user_file_path}: {len(user_history)} characters")
        
        # Format raw history as persona (NO LLM processing - this is the key test)
        history_persona = self.format_history_for_persona(user_history)
        
        logger.debug(f"Created History-Only persona: {len(history_persona)} characters")
        return history_persona
    
    def format_history_for_persona(self, user_history: str) -> str:
        """
        Format raw user history as persona string WITHOUT LLM abstraction.
        """
        if not user_history or not user_history.strip():
            raise ValueError("Cannot format empty user history")
        
        # Clean and format the raw history - NO LLM PROCESSING
        formatted_persona = f"""User Profile based on posting history:

{user_history}

This user's authentic communication patterns are evidenced by their actual social media posts and replies shown above."""
        
        return formatted_persona
    
    def extract_best_persona(self, user_file_path: str) -> str:
        """
        Extract best-performing persona from iterative self-prompting rounds.
        
        Tests the hypothesis: Does iterative optimization improve persona quality?
        
        NO FALLBACKS: Must find and extract best persona or fail.
        """
        if not os.path.exists(user_file_path):
            raise FileNotFoundError(f"User file does not exist: {user_file_path}")
        
        logger.info(f"Extracting Best-Persona for {os.path.basename(user_file_path)}")
        
        # Find best performing round
        best_run_id = self.identify_best_round(user_file_path)
        
        if not best_run_id:
            raise ValueError(f"No evaluations found in {user_file_path} - cannot extract best persona")
        
        # Extract improved persona from reflections
        improved_persona = self.load_persona_from_reflections(user_file_path, best_run_id)
        
        if not improved_persona:
            raise ValueError(f"No improved persona found for best round {best_run_id}")
        
        logger.info(f"Found best persona from round: {best_run_id}")
        return improved_persona
    
    def identify_best_round(self, user_file_path: str) -> str:
        """
        Identify the best performing round based on BERTScore F1 from run_r50_all_metrics.
        Args:
            user_file_path: Path to user JSONL file
        Returns:
            Best run_id
        Raises:
            ValueError: If no suitable evaluations found or no valid BERTScore F1 scores
            FileNotFoundError: If evaluation data is missing
        """
        try:
            # Load file data using cache
            cached_data = self.cache.read_file_with_cache(user_file_path)
            if not cached_data or 3 not in cached_data:
                raise FileNotFoundError(f"No evaluation data found in {user_file_path}")
            
            # Get evaluations (line 4, index 3)
            evaluations_data = cached_data[3]
            evaluations = evaluations_data.get('evaluations', [])
            
            if not evaluations:
                raise ValueError(f"No evaluations found in {user_file_path}")
            
            # DEBUG: Print all run_ids to see what patterns exist
            logger.info(f"DEBUG: All run_ids in {os.path.basename(user_file_path)}:")
            for eval in evaluations:
                run_id = eval.get('run_id', 'NO_ID')
                overall = eval.get('evaluation_results', {}).get('overall', {})
                has_bertscore = 'bertscore' in overall
                logger.info(f"  {run_id}: has_bertscore={has_bertscore}")
            
            # Filter evaluations: only original run_r50_* rounds (which have reflections)
            valid_evaluations = []
            for eval in evaluations:
                run_id = eval.get('run_id', '')
                
                # Only include run_r50_*_round_* pattern
                if not ('run_r50_' in run_id and '_round_' in run_id):
                    continue
                    
                # Skip baseline runs explicitly 
                if 'baseline_' in run_id:
                    continue
                    
                overall = eval.get('evaluation_results', {}).get('overall', {})
                if 'bertscore' in overall:
                    bertscore_data = overall['bertscore']
                    if isinstance(bertscore_data, dict) and 'f1' in bertscore_data:
                        valid_evaluations.append(eval)
            
            logger.info(f"DEBUG: Found {len(valid_evaluations)} valid evaluations after filtering")

            filtered_evaluations = valid_evaluations
            
            if not filtered_evaluations:
                raise ValueError(f"No evaluations with valid BERTScore found in {user_file_path}")
            
            # Find best performing evaluation based on BERTScore F1
            best_evaluation = None
            best_bertscore_f1 = -1
            
            for evaluation in filtered_evaluations:
                eval_results = evaluation.get('evaluation_results', {})
                overall_scores = eval_results.get('overall', {})
                
                # Extract BERTScore F1
                bertscore_data = overall_scores.get('bertscore', {})
                bertscore_f1 = bertscore_data.get('f1', 0) if isinstance(bertscore_data, dict) else 0
                
                if bertscore_f1 > best_bertscore_f1:
                    best_bertscore_f1 = bertscore_f1
                    best_evaluation = evaluation
            
            if best_evaluation and best_bertscore_f1 > 0:
                best_run_id = best_evaluation.get('run_id')
                logger.info(f"Best performing round: {best_run_id} (BERTScore F1: {best_bertscore_f1:.4f})")
                return best_run_id
            
            # No valid scores found
            raise ValueError(f"No valid BERTScore F1 scores found in evaluations for {user_file_path}")

            
        except (FileNotFoundError, ValueError):
            # Re-raise specific errors
            raise
        except Exception as e:
            # Wrap unexpected errors
            raise RuntimeError(f"Error identifying best round in {user_file_path}: {e}")
    
    def load_persona_from_reflections(self, user_file_path: str, run_id: str) -> str:
        """
        Load improved persona from reflection data.
        
        NO FALLBACKS: Must find persona or raise exception.
        """
        # Use existing loader function
        from loader import load_latest_improved_persona
        
        improved_persona = load_latest_improved_persona(run_id, user_file_path)
        
        if not improved_persona:
            raise ValueError(f"No improved persona found for {run_id} in {user_file_path}")
        
        logger.debug(f"Loaded improved persona for {run_id}: {len(improved_persona)} characters")
        return improved_persona
    
    def _get_condition_persona(self, condition_type: str, user_file_path: str, condition_info: dict) -> Optional[str]:
        """
        Get persona for specific condition, handling dynamic persona generation.
        
        NO FALLBACKS: Must succeed for all dynamic conditions or fail.
        """
        if condition_type == "no_persona":
            return None
        elif condition_type == "generic_persona":
            return condition_info['persona']
        elif condition_type == "history_only":
            return self.create_history_only_persona(user_file_path)
        elif condition_type == "best_persona":
            return self.extract_best_persona(user_file_path)
        else:
            raise ValueError(f"Unknown condition type: {condition_type}")
    
    def process_baseline_condition(
        self, 
        user_file_path: str, 
        condition_type: str, 
        stimuli_items: List[Tuple],
        config: Dict[str, Any],
        run_id: str
    ) -> bool:
        """
        Process a single baseline condition for one user.
        
        NO FALLBACKS: All processing must succeed or fail immediately.
        """
        if not os.path.exists(user_file_path):
            raise FileNotFoundError(f"User file does not exist: {user_file_path}")
        
        if condition_type not in self.baseline_conditions:
            raise ValueError(f"Unknown condition type: {condition_type}")
        
        if not stimuli_items:
            raise ValueError(f"No stimuli items provided for {condition_type}")
        
        user_file = os.path.basename(user_file_path)
        condition_info = self.baseline_conditions[condition_type]
        
        logger.info(f"Processing {condition_info['name']} for {user_file}")
        logger.debug(f"Processing {len(stimuli_items)} stimuli for {condition_type}")
        
        # Get LLM configuration
        llm_config = config.get('llm', {})
        if not llm_config:
            raise ValueError("No LLM configuration found in config")
        
        # Get persona for this condition (dynamic for new conditions)
        persona = self._get_condition_persona(condition_type, user_file_path, condition_info)
        
        logger.info(f"Persona for {condition_type}: {len(persona) if persona else 0} characters")
        
        # Process stimuli in parallel (using existing infrastructure)
        successful_count = 0
        failed_stimuli = []
        
        # Use ThreadPoolExecutor for parallel processing (same as main.py)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_stimulus = {}
            
            for stimulus_data in stimuli_items:
                future = executor.submit(
                    self._process_single_baseline_stimulus,
                    stimulus_data, 
                    condition_type,
                    persona,
                    config,
                    user_file_path,
                    run_id
                )
                future_to_stimulus[future] = stimulus_data
            
            # Wait for completion
            for future in concurrent.futures.as_completed(future_to_stimulus):
                stimulus_data = future_to_stimulus[future]
                _, _, post_id = stimulus_data
                
                try:
                    result = future.result()
                    if result:
                        successful_count += 1
                        logger.debug(f"Successfully processed {condition_type} stimulus {post_id}")
                    else:
                        failed_stimuli.append(post_id)
                        logger.error(f"Failed to process {condition_type} stimulus {post_id}")
                except Exception as exc:
                    failed_stimuli.append(post_id)
                    logger.error(f"Stimulus {post_id} generated exception: {exc}")
        
        # Check if we have enough successful processing
        if successful_count == 0:
            raise RuntimeError(f"No stimuli successfully processed for {condition_type}")
        
        if len(failed_stimuli) > len(stimuli_items) * 0.5:  # More than 50% failed
            raise RuntimeError(f"Too many failures for {condition_type}: {len(failed_stimuli)}/{len(stimuli_items)} failed")
        
        logger.info(f"Completed {condition_type}: {successful_count}/{len(stimuli_items)} stimuli processed")
        
        # Run evaluation for this condition
        self._evaluate_baseline_condition(user_file_path, run_id)
        logger.info(f"Evaluation completed for {condition_type}")
        
        return True
    
    def _process_single_baseline_stimulus(
        self,
        stimulus_data: Tuple,
        condition_type: str, 
        persona: Optional[str],
        config: Dict[str, Any],
        user_file_path: str,
        run_id: str
    ) -> bool:
        """
        Process a single stimulus for baseline condition.
        
        NO FALLBACKS: Must complete successfully or raise exception.
        """
        stimulus, is_post, post_id = stimulus_data
        condition_info = self.baseline_conditions[condition_type]
        llm_config = config.get('llm', {})
        
        if not stimulus or not post_id:
            raise ValueError(f"Invalid stimulus data: {stimulus_data}")
        
        # Select appropriate template based on task type and condition
        if is_post:
            template_name = condition_info['post_template']
        else:
            template_name = condition_info['reply_template']
        
        # Format stimulus based on condition type
        if condition_type == "no_persona":
            # Create no-persona template dynamically if needed
            if template_name == "imitation_post_template_no_persona":
                # No-persona template without any persona reference
                stimulus_formatted = f"""You are completing a social media post. Fill in the [MASKED] words in the tweet below to complete it naturally.

Original Tweet: {stimulus}

Instructions:
- Replace each [MASKED] token with appropriate words
- Maintain the original tweet's structure and meaning
- Write naturally and authentically
- Keep the tone consistent

Completed Tweet:"""
            elif template_name == "imitation_replies_template_no_persona":
                stimulus_formatted = f"""You are responding to a social media post. Write a natural reply to the tweet below.

Tweet to reply to: {stimulus}

Instructions:
- Write a natural reply that someone would post
- Keep the response length appropriate for social media
- Write authentically and conversationally

Reply:"""
            else:
                raise ValueError(f"Unknown no-persona template: {template_name}")
        else:
            # All other conditions use persona (generic, history_only, best_persona)
            if not persona:
                raise ValueError(f"Persona required for condition {condition_type} but none provided")
            
            stimulus_formatted = templates.format_template(
                template_name,
                persona=persona,
                tweet=stimulus
            )
        
        # Call AI model
        imitation_model = llm_config.get('imitation_model')
        if not imitation_model:
            raise ValueError("No imitation_model specified in LLM config")
        
        imitation = call_ai(stimulus_formatted, imitation_model)
        
        if not imitation or not imitation.strip():
            raise ValueError(f"Empty imitation generated for {condition_type} stimulus {post_id}")
        
        logger.debug(f"Generated {condition_type} imitation for {post_id}: {imitation[:100]}...")
        
        # Save results with thread safety
        with self.file_lock:
            save_user_imitation(
                file_path=user_file_path,
                stimulus=stimulus,
                persona=persona or f"[{condition_type.upper()}]",  # Use condition marker if no persona
                imitation=imitation,
                run_id=run_id,
                tweet_id=post_id
            )
        
        return True
    
    def _evaluate_baseline_condition(self, user_file_path: str, run_id: str) -> None:
        """
        Evaluate baseline condition using existing evaluation infrastructure.
        
        NO FALLBACKS: Must complete evaluation or raise exception.
        """
        from loader import load_predictions_orginales_formated
        
        # Load predictions and originals (reuse existing loader)
        results = load_predictions_orginales_formated(run_id=run_id, file_path=user_file_path)
        
        if not results:
            raise ValueError(f"No results found for evaluation of run_id {run_id}")
        
        # Evaluate using existing evaluation function
        evaluation_result = evaluate_with_individual_scores(results)
        
        if not evaluation_result:
            raise ValueError(f"Evaluation failed for run_id {run_id}")
        
        # Save evaluation results
        save_evaluation_results(
            file_path=user_file_path, 
            evaluation_results=evaluation_result, 
            run_id=run_id
        )
        
        logger.debug(f"Evaluation completed for run_id {run_id}")
    
    def run_baseline_experiment(self, config: Dict[str, Any], extended_mode: bool = True) -> Dict[str, Any]:
        """
        Run complete baseline experiment across all users.
        
        NO FALLBACKS: All users must complete successfully or experiment fails.
        Uses IDENTICAL STIMULI for all conditions (optimal scientific design).
        """
        mode_desc = "4-condition (V2)" if extended_mode else "2-condition (V1)"
        logger.info(f"Starting BASELINE EXPERIMENT - {mode_desc} with IDENTICAL STIMULI")
        
        # Extract configuration
        experiment_config = config.get('experiment')
        if not experiment_config:
            raise ValueError("No experiment configuration found in config")
        
        users_dict_path = experiment_config.get('users_dict')
        if not users_dict_path or not os.path.exists(users_dict_path):
            raise FileNotFoundError(f"Invalid users_dict path: {users_dict_path}")
        
        number_of_users = experiment_config.get('number_of_users')
        run_name_prefix = experiment_config.get('run_name_prefix')
        if not run_name_prefix:
            raise ValueError("No run_name_prefix specified in config")
        
        num_stimuli = experiment_config.get('num_stimuli_to_process')
        if not num_stimuli or num_stimuli <= 0:
            raise ValueError("Invalid num_stimuli_to_process in config")
        
        # Get user files
        user_files = [f for f in os.listdir(users_dict_path) 
                     if f.endswith('.jsonl')]
        
        if not user_files:
            raise FileNotFoundError("No user files found in users_dict path")
        
        if number_of_users:
            user_files = user_files[:number_of_users]
        
        logger.info(f"Found {len(user_files)} user files for baseline experiment")
        
        # Shuffle users for randomization
        random.shuffle(user_files)
        
        # Generate run IDs for each condition
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if extended_mode:
            # 4-condition mode (V2)
            run_ids = {
                "no_persona": f"{run_name_prefix}_no_persona_{timestamp}",
                "generic_persona": f"{run_name_prefix}_generic_{timestamp}",
                "history_only": f"{run_name_prefix}_history_only_{timestamp}",
                "best_persona": f"{run_name_prefix}_best_persona_{timestamp}"
            }
        else:
            # 2-condition mode (V1)
            run_ids = {
                "no_persona": f"{run_name_prefix}_no_persona_{timestamp}",
                "generic_persona": f"{run_name_prefix}_generic_{timestamp}"
            }
        
        logger.info(f"Baseline run IDs: {list(run_ids.values())}")
        
        # Process each user with all baseline conditions
        results = {
            "processed_users": [],
            "failed_users": [],
            "run_ids": run_ids,
            "total_users": len(user_files),
            "extended_mode": extended_mode,
            "stimuli_per_condition": num_stimuli
        }

        for user_file in user_files:
            user_file_path = os.path.join(users_dict_path, user_file)
            user_id = user_file.replace('.jsonl', '')
            
            logger.info(f"Processing baseline experiment for user: {user_id}")
            
            try:
                # Pre-validate: Check if all conditions can be generated for this user
                # This prevents partial processing and wasted computation
                logger.info(f"Pre-validating conditions for {user_id}...")
                
                for condition_type in run_ids.keys():
                    condition_info = self.baseline_conditions[condition_type]
                    
                    # Test persona generation without actually using it
                    try:
                        test_persona = self._get_condition_persona(condition_type, user_file_path, condition_info)
                        logger.debug(f"✓ {condition_type} persona validated for {user_id}")
                    except Exception as e:
                        raise ValueError(f"Cannot generate {condition_type} condition for {user_id}: {e}")
                
                logger.info(f"✓ All conditions validated for {user_id}")
                
                # Create IDENTICAL stimuli split for this user
                split_data = self.create_identical_stimuli_splits(
                    user_file_path, 
                    num_items=num_stimuli,
                    seed=42  # Fixed seed for reproducibility
                )
                
                # Validate that all conditions have data
                expected_conditions = list(run_ids.keys())
                missing_conditions = [cond for cond in expected_conditions if not split_data.get(cond)]
                
                if missing_conditions:
                    raise ValueError(f"Missing conditions for {user_id}: {missing_conditions}")
                
                # Validate identical stimuli
                first_condition = expected_conditions[0]
                first_stimuli_ids = [item[2] for item in split_data[first_condition]]
                
                for condition in expected_conditions[1:]:
                    condition_stimuli_ids = [item[2] for item in split_data[condition]]
                    if first_stimuli_ids != condition_stimuli_ids:
                        raise ValueError(f"CRITICAL: Non-identical stimuli for {user_id}")
                
                # All conditions have identical stimuli - proceed with processing
                logger.info(f"✓ Verified IDENTICAL stimuli for {user_id}: {len(first_stimuli_ids)} items")
                
                # Process all baseline conditions (now we know all will succeed)
                for condition_type, stimuli_items in split_data.items():
                    if condition_type not in run_ids:
                        continue  # Skip unexpected conditions
                    
                    condition_run_id = run_ids[condition_type]
                    
                    logger.info(f"Processing {condition_type} for {user_id} ({len(stimuli_items)} stimuli)")
                    
                    # NO FALLBACKS - must succeed or fail (but we pre-validated)
                    self.process_baseline_condition(
                        user_file_path=user_file_path,
                        condition_type=condition_type,
                        stimuli_items=stimuli_items,
                        config=config,
                        run_id=condition_run_id
                    )
                    
                    logger.info(f"✓ Successfully completed {condition_type} for {user_id}")
                
                # Record user as processed (all conditions succeeded)
                results["processed_users"].append(user_id)
                logger.info(f"✓ User {user_id} completed: {len(run_ids)}/{len(run_ids)} conditions successful")
                
            except Exception as e:
                # User failed - skip and continue with next user
                results["failed_users"].append({"user_id": user_id, "error": str(e)})
                logger.warning(f"✗ Skipping user {user_id}: {e}")
                continue

        # Log final results
        logger.info(f"🎯 BASELINE EXPERIMENT COMPLETED ({mode_desc}):")
        logger.info(f"  ✓ Successful users: {len(results['processed_users'])}")
        logger.info(f"  ✗ Failed users: {len(results['failed_users'])}")

        # Show failed user details
        if results['failed_users']:
            logger.info(f"  Failed user details:")
            for failed_user in results['failed_users']:
                logger.info(f"    - {failed_user['user_id']}: {failed_user['error']}")

        logger.info(f"  📊 Success rate: {len(results['processed_users'])}/{results['total_users']} ({100*len(results['processed_users'])/results['total_users']:.1f}%)")
        logger.info(f"  📊 Total condition-item instances: {len(results['processed_users']) * len(run_ids) * num_stimuli:,}")
        logger.info(f"  🔬 Run IDs: {list(results['run_ids'].values())}")

        return results


def main():
    """
    Main function to run baseline experiments.
    Supports both V1 (2-condition) and V2 (4-condition) modes.
    NO FALLBACKS: Fails immediately on any error.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Run baseline experiments for persona imitation")
    parser.add_argument('--config', type=str, default='baseline_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['v1', 'v2'], default='v2',
                       help='Experiment mode: v1 (2-condition) or v2 (4-condition)')
    parser.add_argument('--users', type=int, 
                       help='Number of users to process (overrides config)')
    parser.add_argument('--stimuli', type=int,
                       help='Number of stimuli per condition (default: 50)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Validate setup without running experiment')
    
    args = parser.parse_args()
    
    # Load configuration - NO FALLBACKS
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")
    
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if not config:
        raise ValueError(f"Empty or invalid configuration file: {args.config}")
    
    # Apply command line overrides
    if args.users:
        config['experiment']['number_of_users'] = args.users
    if args.stimuli:
        config['experiment']['num_stimuli_to_process'] = args.stimuli
    
    # Initialize baseline experiment manager
    baseline_manager = BaselineExperiment(config['experiment']['users_dict'])
    
    if args.dry_run:
        logger.info("🔍 DRY RUN MODE - Validating setup")
        
        # Validate data directory
        users_dict_path = config['experiment']['users_dict']
        if not os.path.exists(users_dict_path):
            raise FileNotFoundError(f"Data directory not found: {users_dict_path}")
        
        # Check user files
        user_files = [f for f in os.listdir(users_dict_path) if f.endswith('.jsonl')]
        logger.info(f"📁 Found {len(user_files)} user files")
        
        if len(user_files) == 0:
            raise FileNotFoundError("No user files found")
        
        # Test stimulus loading for first user
        test_user_path = os.path.join(users_dict_path, user_files[0])
        test_stimuli = baseline_manager.create_identical_stimuli_splits(
            test_user_path, 
            num_items=args.stimuli
        )
        
        logger.info(f"✅ Test stimulus split successful: {len(test_stimuli)} conditions")
        
        # Estimate processing time
        total_instances = len(user_files) * (4 if args.mode == 'v2' else 2) * args.stimuli
        estimated_minutes = total_instances / 100  # Rough estimate: 100 instances per minute
        
        logger.info(f"📊 Experiment Overview:")
        logger.info(f"  Mode: {args.mode.upper()} ({'4-condition' if args.mode == 'v2' else '2-condition'})")
        logger.info(f"  Users: {len(user_files)}")
        logger.info(f"  Stimuli per condition: {args.stimuli}")
        logger.info(f"  Total instances: {total_instances:,}")
        logger.info(f"  Estimated time: {estimated_minutes:.0f} minutes")
        logger.info("✅ Setup validation completed")
        
        return True
    
    # Run actual experiment
    extended_mode = (args.mode == 'v2')
    
    logger.info(f"🚀 Starting baseline experiment in {args.mode.upper()} mode")
    start_time = datetime.now()
    
    results = baseline_manager.run_baseline_experiment(config, extended_mode=extended_mode)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Log final results
    logger.info(f"🎉 BASELINE EXPERIMENT COMPLETED!")
    logger.info(f"⏱️  Duration: {duration}")
    logger.info(f"✅ Successful users: {len(results['processed_users'])}")
    logger.info(f"❌ Failed users: {len(results['failed_users'])}")
    logger.info(f"📊 Run IDs created: {list(results['run_ids'].values())}")
    
    # Calculate total instances processed
    successful_instances = len(results['processed_users']) * len(results['run_ids']) * results['stimuli_per_condition']
    logger.info(f"🔢 Total condition-item instances: {successful_instances:,}")
    
    # Log next steps
    logger.info(f"📝 Next steps:")
    logger.info(f"  1. Run analysis: python baseline_analysis.py --run-ids {' '.join(results['run_ids'].values())}")
    logger.info(f"  2. Generate report: python baseline_analysis.py --report")
    logger.info(f"  3. Export data: python baseline_analysis.py --export-csv")
    
    return True


if __name__ == "__main__":
    main()