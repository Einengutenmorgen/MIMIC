"""
Baseline Experiments Manager V2 - COMPLETED

Scientific Design: IDENTICAL STIMULI across all conditions for maximum statistical power.
All conditions receive the SAME tweets to eliminate stimulus variance.
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
    """
    
    def __init__(self, data_dir: str = "/Users/christophhau/Desktop/HA_Projekt/MIMIC/MIMIC/data/filtered_users_without_links_meta/filtered_users_without_links"):
        self.data_dir = Path(data_dir)
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
                "post_template": "imitation_post_template_simple",  # Use same templates as individual
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
    
    def create_identical_stimuli_splits(self, user_file_path: str, num_items: int = 50, seed: int = 42) -> Dict[str, List[Tuple]]:
        """
        Create IDENTICAL stimulus sets for all conditions - OPTIMAL SCIENTIFIC DESIGN.
        
        CRITICAL: All conditions get the SAME stimuli for maximum statistical power.
        This eliminates all stimulus-based variance and allows pure condition comparisons.
        
        Scientific Rationale:
        - Eliminates confounding variables (stimulus content, difficulty, length)
        - Maximizes within-subject statistical power
        - Enables direct condition comparisons
        - Follows best practices for controlled experiments
        
        Args:
            user_file_path: Path to user JSONL file
            num_items: Number of items per condition (default: 50)
            seed: Random seed for reproducible selection
            
        Returns:
            Dict with condition keys, all containing IDENTICAL stimulus lists
        """
        logger.info(f"Creating IDENTICAL stimulus sets ({num_items} items) for {os.path.basename(user_file_path)}")
        
        # Load all stimuli
        all_stimuli = load_stimulus(user_file_path)
        
        if len(all_stimuli) < num_items:
            logger.warning(f"Insufficient stimuli in {user_file_path}: {len(all_stimuli)} < {num_items}")
            selected_stimuli = all_stimuli  # Use all available
            actual_items = len(all_stimuli)
        else:
            # Randomly select num_items for the experiment
            random.seed(seed)
            selected_stimuli = random.sample(all_stimuli, num_items)
            actual_items = num_items
        
        # Analyze stimulus composition for logging
        posts = [item for item in selected_stimuli if item[1] == True]
        replies = [item for item in selected_stimuli if item[1] == False]
        
        logger.info(f"Selected {actual_items} IDENTICAL stimuli: {len(posts)} posts, {len(replies)} replies")
        
        # CRITICAL: All conditions get IDENTICAL stimuli
        condition_names = ["no_persona", "generic_persona", "history_only", "best_persona"]
        
        result = {}
        for condition in condition_names:
            result[condition] = selected_stimuli.copy()  # IDENTICAL for all conditions
            logger.debug(f"Condition {condition}: {len(result[condition])} items (IDENTICAL to all others)")
        
        # Validate identical stimuli across conditions
        first_condition_ids = [item[2] for item in result[condition_names[0]]]  # tweet_ids
        for condition in condition_names[1:]:
            condition_ids = [item[2] for item in result[condition]]
            if first_condition_ids != condition_ids:
                raise ValueError(f"CRITICAL ERROR: Stimuli not identical across conditions!")
        
        logger.info(f"✅ IDENTICAL STIMULI VALIDATED: All {len(condition_names)} conditions use same {actual_items} items")
        return result
    
    def create_history_only_persona(self, user_file_path: str) -> str:
        """
        Create History-Only persona from raw user data without LLM abstraction.
        
        Tests the hypothesis: Does LLM abstraction improve persona quality?
        
        Args:
            user_file_path: Path to user JSONL file
            
        Returns:
            Raw user history formatted as persona string
        """
        logger.info(f"Creating History-Only persona for {os.path.basename(user_file_path)}")
        
        try:
            # Use existing loader to get formatted user history
            user_history = get_formatted_user_historie(user_file_path)
            
            if not user_history or len(user_history) < 100:
                logger.warning(f"Insufficient user history in {user_file_path}")
                return "Limited user history available for persona creation."
            
            # Format raw history as persona (NO LLM processing - this is the key test)
            history_persona = self.format_history_for_persona(user_history)
            
            logger.debug(f"Created History-Only persona: {len(history_persona)} characters")
            return history_persona
            
        except Exception as e:
            logger.error(f"Error creating History-Only persona: {e}")
            return "Error: Could not create History-Only persona from available data."
    
    def format_history_for_persona(self, user_history: str) -> str:
        """
        Format raw user history as persona string WITHOUT LLM abstraction.
        
        Args:
            user_history: Raw formatted user history from loader
            
        Returns:
            Formatted persona string for History-Only condition
        """
        # Clean and format the raw history - NO LLM PROCESSING
        formatted_persona = f"""User Profile based on posting history:

{user_history}

This user's authentic communication patterns are evidenced by their actual social media posts and replies shown above."""
        
        return formatted_persona
    
    def extract_best_persona(self, user_file_path: str, fallback_persona: str = None) -> str:
        """
        Extract best-performing persona from iterative self-prompting rounds.
        
        Tests the hypothesis: Does iterative optimization improve persona quality?
        
        Args:
            user_file_path: Path to user JSONL file  
            fallback_persona: Fallback persona if no iterative rounds found
            
        Returns:
            Best optimized persona or fallback
        """
        logger.info(f"Extracting Best-Persona for {os.path.basename(user_file_path)}")
        
        try:
            # Find best performing round
            best_run_id = self.identify_best_round(user_file_path)
            
            if best_run_id:
                # Extract improved persona from reflections
                improved_persona = self.load_persona_from_reflections(user_file_path, best_run_id)
                
                if improved_persona and len(improved_persona) > 50:
                    logger.info(f"Found best persona from round: {best_run_id}")
                    return improved_persona
            
            # Fallback: Use provided fallback or create initial persona
            if fallback_persona:
                logger.info("Using provided fallback persona for Best-Persona condition")
                return fallback_persona
            else:
                logger.info("Creating initial persona as fallback for Best-Persona condition")
                return self.create_initial_persona(user_file_path)
                
        except Exception as e:
            logger.error(f"Error extracting Best-Persona: {e}")
            return fallback_persona or "Error: Could not extract best persona."
    
    def identify_best_round(self, user_file_path: str) -> Optional[str]:
        """
        Identify the best performing round based on evaluation metrics.
        
        Args:
            user_file_path: Path to user JSONL file
            
        Returns:
            Best run_id or None if no evaluations found
        """
        try:
            # Load file data using cache
            cached_data = self.cache.read_file_with_cache(user_file_path)
            if not cached_data or 3 not in cached_data:
                logger.warning(f"No evaluation data found in {user_file_path}")
                return None
            
            # Get evaluations (line 4, index 3)
            evaluations_data = cached_data[3]
            evaluations = evaluations_data.get('evaluations', [])
            
            if not evaluations:
                logger.warning(f"No evaluations found in {user_file_path}")
                return None
            
            # Find best performing evaluation based on combined score
            best_evaluation = None
            best_combined_score = -1
            
            for evaluation in evaluations:
                eval_results = evaluation.get('evaluation_results', {})
                overall_scores = eval_results.get('overall', {})
                
                # Calculate combined score (same logic as eval.py)
                rouge_scores = overall_scores.get('rouge', {})
                bleu_scores = overall_scores.get('bleu', {})
                
                rouge1 = rouge_scores.get('rouge1', 0) if isinstance(rouge_scores, dict) else 0
                rouge2 = rouge_scores.get('rouge2', 0) if isinstance(rouge_scores, dict) else 0
                rougeL = rouge_scores.get('rougeL', 0) if isinstance(rouge_scores, dict) else 0
                bleu = bleu_scores.get('bleu', 0) if isinstance(bleu_scores, dict) else bleu_scores if isinstance(bleu_scores, (int, float)) else 0
                
                combined_score = (rouge1 + rouge2 + rougeL + bleu) / 4.0
                
                if combined_score > best_combined_score:
                    best_combined_score = combined_score
                    best_evaluation = evaluation
            
            if best_evaluation:
                best_run_id = best_evaluation.get('run_id')
                logger.info(f"Best performing round: {best_run_id} (score: {best_combined_score:.4f})")
                return best_run_id
            
            return None
            
        except Exception as e:
            logger.error(f"Error identifying best round: {e}")
            return None
    
    def load_persona_from_reflections(self, user_file_path: str, run_id: str) -> Optional[str]:
        """
        Load improved persona from reflection data.
        
        Args:
            user_file_path: Path to user JSONL file
            run_id: Run ID to load persona for
            
        Returns:
            Improved persona string or None
        """
        try:
            # Use existing loader function
            from loader import load_latest_improved_persona
            
            improved_persona = load_latest_improved_persona(run_id, user_file_path)
            
            if improved_persona:
                logger.debug(f"Loaded improved persona for {run_id}: {len(improved_persona)} characters")
                return improved_persona
            else:
                logger.warning(f"No improved persona found for {run_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading persona from reflections: {e}")
            return None
    
    def create_initial_persona(self, user_file_path: str) -> str:
        """
        Create initial persona using LLM (same as main experiment).
        
        Args:
            user_file_path: Path to user JSONL file
            
        Returns:
            Initial persona string
        """
        try:
            # Use same logic as main.py for initial persona creation
            user_history = get_formatted_user_historie(user_file_path)
            
            # Use existing template
            formatted_user_history = templates.format_template(
                'persona_template_simple',
                historie=user_history
            )
            
            # Call LLM (use Google model as in main experiment)
            initial_persona = call_ai(formatted_user_history, 'google')
            
            logger.info(f"Created initial persona as fallback: {len(initial_persona)} characters")
            return initial_persona
            
        except Exception as e:
            logger.error(f"Error creating initial persona: {e}")
            return "Generic social media user with varied interests and communication style."
    
    def _get_condition_persona(self, condition_type: str, user_file_path: str, condition_info: dict) -> Optional[str]:
        """
        Get persona for specific condition, handling dynamic persona generation.
        
        Args:
            condition_type: Type of baseline condition
            user_file_path: Path to user file
            condition_info: Condition configuration
            
        Returns:
            Persona string or None for no_persona condition
        """
        if condition_type == "no_persona":
            return None
        elif condition_type == "generic_persona":
            return condition_info['persona']
        elif condition_type == "history_only":
            return self.create_history_only_persona(user_file_path)
        elif condition_type == "best_persona":
            # Try to extract best persona, fallback to initial if needed
            initial_fallback = self.create_initial_persona(user_file_path)
            return self.extract_best_persona(user_file_path, initial_fallback)
        else:
            logger.error(f"Unknown condition type: {condition_type}")
            return None
    
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
        
        Updated for V2: Supports all 4 baseline conditions with dynamic persona generation.
        
        Args:
            user_file_path: Path to user JSONL file
            condition_type: 'no_persona', 'generic_persona', 'history_only', or 'best_persona'
            stimuli_items: List of (stimulus, is_post, post_id) tuples
            config: Experiment configuration
            run_id: Unique run identifier
            
        Returns:
            True if successful, False otherwise
        """
        user_file = os.path.basename(user_file_path)
        condition_info = self.baseline_conditions[condition_type]
        
        logger.info(f"Processing {condition_info['name']} for {user_file}")
        logger.debug(f"Processing {len(stimuli_items)} stimuli for {condition_type}")
        
        # Get LLM configuration
        llm_config = config.get('llm', {})
        
        # Get persona for this condition (dynamic for new conditions)
        persona = self._get_condition_persona(condition_type, user_file_path, condition_info)
        
        if persona is None and condition_type != "no_persona":
            logger.error(f"Failed to create persona for {condition_type}")
            return False
        
        logger.info(f"Persona for {condition_type}: {len(persona) if persona else 0} characters")
        
        # Process stimuli in parallel (using existing infrastructure)
        successful_count = 0
        
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
                        logger.warning(f"Failed to process {condition_type} stimulus {post_id}")
                except Exception as exc:
                    logger.error(f"Stimulus {post_id} generated exception: {exc}")
        
        logger.info(f"Completed {condition_type}: {successful_count}/{len(stimuli_items)} stimuli processed")
        
        # Run evaluation for this condition
        try:
            self._evaluate_baseline_condition(user_file_path, run_id)
            logger.info(f"Evaluation completed for {condition_type}")
            return True
        except Exception as e:
            logger.error(f"Evaluation failed for {condition_type}: {e}")
            return False
    
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
        
        Updated for V2: Handles all 4 baseline conditions with appropriate template selection.
        """
        stimulus, is_post, post_id = stimulus_data
        condition_info = self.baseline_conditions[condition_type]
        llm_config = config.get('llm', {})
        
        try:
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
                    # Fallback: use template without persona
                    stimulus_formatted = templates.format_template(
                        "imitation_post_template_simple" if is_post else "imitation_replies_template_simple",
                        persona="",
                        tweet=stimulus
                    )
            else:
                # All other conditions use persona (generic, history_only, best_persona)
                stimulus_formatted = templates.format_template(
                    template_name,
                    persona=persona,
                    tweet=stimulus
                )
            
            # Call AI model
            imitation_model = llm_config.get('imitation_model', 'ollama')
            imitation = call_ai(stimulus_formatted, imitation_model)
            
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
            
        except Exception as e:
            logger.error(f"Error processing {condition_type} stimulus {post_id}: {e}")
            return False
    
    def _evaluate_baseline_condition(self, user_file_path: str, run_id: str) -> None:
        """
        Evaluate baseline condition using existing evaluation infrastructure.
        """
        from loader import load_predictions_orginales_formated
        
        try:
            # Load predictions and originals (reuse existing loader)
            results = load_predictions_orginales_formated(run_id=run_id, file_path=user_file_path)
            
            if not results:
                logger.warning(f"No results found for evaluation of run_id {run_id}")
                return
            
            # Evaluate using existing evaluation function
            evaluation_result = evaluate_with_individual_scores(results)
            
            # Save evaluation results
            save_evaluation_results(
                file_path=user_file_path, 
                evaluation_results=evaluation_result, 
                run_id=run_id
            )
            
            logger.debug(f"Evaluation completed for run_id {run_id}")
            
        except Exception as e:
            logger.error(f"Error in evaluation for {run_id}: {e}")
            raise
    
    def run_baseline_experiment(self, config: Dict[str, Any], extended_mode: bool = True) -> Dict[str, Any]:
        """
        Run complete baseline experiment across all users.
        
        Updated for V2: Supports both 2-condition and 4-condition modes.
        Uses IDENTICAL STIMULI for all conditions (optimal scientific design).
        
        Args:
            config: Experiment configuration dict
            extended_mode: If True, run 4-condition experiment (V2)
            
        Returns:
            Dictionary with experiment results and statistics
        """
        mode_desc = "4-condition (V2)" if extended_mode else "2-condition (V1)"
        logger.info(f"Starting BASELINE EXPERIMENT - {mode_desc} with IDENTICAL STIMULI")
        
        # Extract configuration
        experiment_config = config.get('experiment', {})
        number_of_users = experiment_config.get('number_of_users')
        users_dict_path = experiment_config.get('users_dict')
        run_name_prefix = experiment_config.get('run_name_prefix', 'baseline')
        num_stimuli = experiment_config.get('num_stimuli_to_process', 50)
        
        if not users_dict_path or not os.path.exists(users_dict_path):
            logger.error(f"Invalid users_dict path: {users_dict_path}")
            return {"error": "Invalid data path"}
        
        # Get user files
        user_files = [f for f in os.listdir(users_dict_path) 
                     if f.endswith('.jsonl')]
        
        if number_of_users:
            user_files = user_files[:number_of_users]
        
        if not user_files:
            logger.error("No user files found")
            return {"error": "No user files found"}
        
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
            
            try:
                logger.info(f"Processing baseline experiment for user: {user_id}")
                
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
                    logger.warning(f"Missing conditions for {user_id}: {missing_conditions}")
                    results["failed_users"].append(user_id)
                    continue
                
                # Validate identical stimuli
                first_condition = expected_conditions[0]
                first_stimuli_ids = [item[2] for item in split_data[first_condition]]
                
                for condition in expected_conditions[1:]:
                    condition_stimuli_ids = [item[2] for item in split_data[condition]]
                    if first_stimuli_ids != condition_stimuli_ids:
                        logger.error(f"CRITICAL: Non-identical stimuli for {user_id}")
                        results["failed_users"].append(user_id)
                        break
                else:
                    # All conditions have identical stimuli - proceed with processing
                    logger.info(f"✅ Verified IDENTICAL stimuli for {user_id}: {len(first_stimuli_ids)} items")
                    
                    # Process all baseline conditions
                    success_count = 0
                    
                    for condition_type, stimuli_items in split_data.items():
                        if condition_type not in run_ids:
                            continue  # Skip unexpected conditions
                        
                        condition_run_id = run_ids[condition_type]
                        
                        logger.info(f"Processing {condition_type} for {user_id} ({len(stimuli_items)} stimuli)")
                        
                        success = self.process_baseline_condition(
                            user_file_path=user_file_path,
                            condition_type=condition_type,
                            stimuli_items=stimuli_items,
                            config=config,
                            run_id=condition_run_id
                        )
                        
                        if success:
                            success_count += 1
                            logger.info(f"✅ Successfully completed {condition_type} for {user_id}")
                        else:
                            logger.error(f"❌ Failed {condition_type} for {user_id}")
                    
                    # Record user as processed if most conditions succeeded
                    min_success = max(1, len(run_ids) // 2)  # At least half should succeed
                    if success_count >= min_success:
                        results["processed_users"].append(user_id)
                        logger.info(f"✅ User {user_id} completed: {success_count}/{len(run_ids)} conditions successful")
                    else:
                        results["failed_users"].append(user_id)
                        logger.error(f"❌ User {user_id} failed: only {success_count}/{len(run_ids)} conditions successful")
                    
            except Exception as e:
                logger.error(f"Error processing user {user_id}: {e}")
                results["failed_users"].append(user_id)
        
        # Log final results
        logger.info(f"🎯 BASELINE EXPERIMENT COMPLETED ({mode_desc}):")
        logger.info(f"  ✅ Successful users: {len(results['processed_users'])}")
        logger.info(f"  ❌ Failed users: {len(results['failed_users'])}")
        logger.info(f"  📊 Total condition-item instances: {len(results['processed_users']) * len(run_ids) * num_stimuli:,}")
        logger.info(f"  🔬 Run IDs: {list(results['run_ids'].values())}")
        
        return results


def create_baseline_config(base_config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Create configuration for baseline experiments.
    
    Args:
        base_config_path: Path to base configuration file
        
    Returns:
        Configuration dict optimized for baseline experiments
    """
    from utils import load_config
    
    try:
        # Load base configuration
        base_config = load_config(base_config_path)
        
        # Override settings for baseline experiments
        baseline_config = base_config.copy()
        
        # Experiment settings for baseline
        baseline_config['experiment'].update({
            'run_name_prefix': 'baseline',
            'number_of_rounds': 1,  # Baseline doesn't use iterative rounds
            'num_stimuli_to_process': 50,  # Standard baseline size
            'use_async': True,
            'num_workers': 4  # Parallel processing for efficiency
        })
        
        # Use same LLM settings as main experiment for comparability
        # No changes
        logger.info("Created baseline experiment configuration")
        return baseline_config
        
    except Exception as e:
        logger.error(f"Error creating baseline config: {e}")
        # Return minimal fallback config
        return {
            'experiment': {
                'run_name_prefix': 'baseline',
                'users_dict': 'data/filtered_users',
                'number_of_users': 24,
                'number_of_rounds': 1,
                'num_stimuli_to_process': 50,
                'use_async': True,
                'num_workers': 4
            },
            'llm': {
                'persona_model': 'google',
                'imitation_model': 'ollama',
                'reflection_model': 'google_json',
                'ollama_model': 'gemma3:latest'
            },
            'templates': {
                'persona_template': 'persona_template_simple',
                'imitation_post_template': 'imitation_post_template_simple',
                'imitation_reply_template': 'imitation_replies_template_simple'
            }
        }


def main():
    """
    Main function to run baseline experiments.
    Supports both V1 (2-condition) and V2 (4-condition) modes.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Run baseline experiments for persona imitation")
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['v1', 'v2'], default='v2',
                       help='Experiment mode: v1 (2-condition) or v2 (4-condition)')
    parser.add_argument('--users', type=int, 
                       help='Number of users to process (overrides config)')
    parser.add_argument('--stimuli', type=int, default=50,
                       help='Number of stimuli per condition (default: 50)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Validate setup without running experiment')
    
    args = parser.parse_args()
    
    try:
        # Create baseline configuration
        config = create_baseline_config(args.config)
        
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
                logger.error(f"❌ Data directory not found: {users_dict_path}")
                return False
            
            # Check user files
            user_files = [f for f in os.listdir(users_dict_path) if f.endswith('.jsonl')]
            logger.info(f"📁 Found {len(user_files)} user files")
            
            if len(user_files) == 0:
                logger.error("❌ No user files found")
                return False
            
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
        if "error" in results:
            logger.error(f"❌ Experiment failed: {results['error']}")
            return False
        
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
        
    except KeyboardInterrupt:
        logger.info("⏹️  Experiment interrupted by user")
        return False
    except Exception as e:
        logger.error(f"❌ Experiment failed with error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)