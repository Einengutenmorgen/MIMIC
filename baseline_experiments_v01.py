"""
Baseline Experiments Manager

Handles all baseline experiment conditions:
- No-Persona: Complete persona removal
- Generic-Persona: Universal social media user persona
- History-Only: Persona created only from user's post history
- Best-Persona: Persona from the main experimental track (for comparison)
"""

import os
import json
import random
import math
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
    
    Now supports V2 conditions including History-Only persona.
    
    Baseline Conditions:
    1. No-Persona: Task performance without any persona conditioning.
    2. Generic-Persona: Minimal, universal social media user persona.
    3. History-Only: Persona dynamically generated from the user's recent post history.
    4. Best-Persona: The best-performing persona from the main experiment (placeholder).
    """
    
    def __init__(self, data_dir: str = "data/filtered_users"):
        self.data_dir = Path(data_dir)
        self.cache = get_file_cache()
        self.file_lock = threading.Lock()
        
        # V2 Baseline definitions, including new 'history_only' and 'best_persona'
        self.baseline_conditions = {
            "no_persona": {
                "name": "No-Persona Baseline",
                "description": "Task performance without persona conditioning",
                "post_template": "imitation_post_template_no_persona",
                "reply_template": "imitation_replies_template_no_persona",
                "persona": None  # Static: No persona used
            },
            "generic_persona": {
                "name": "Generic-Persona Baseline", 
                "description": "Minimal universal social media user persona",
                "post_template": "imitation_post_template_generic",
                "reply_template": "imitation_replies_template_generic",
                "persona": "You are an active social media user who writes posts and replies in a conversational, authentic way. You engage with various topics and express yourself clearly and naturally online."
            },
            "history_only": {
                "name": "History-Only Persona Baseline",
                "description": "Persona dynamically generated from user's post history",
                "post_template": "imitation_post_template_history",
                "reply_template": "imitation_replies_template_history",
                "persona": "[DYNAMIC]" # Dynamic: Generated per user
            },
            "best_persona": {
                "name": "Best-Persona Comparison",
                "description": "Comparison against the best individual persona from the main experiment",
                "post_template": "imitation_post_template_best_persona",
                "reply_template": "imitation_replies_template_best_persona",
                "persona": "[DYNAMIC]" # Dynamic: Loaded per user
            }
        }

    def format_history_for_persona(self, history: List[Dict[str, str]]) -> str:
        """Formats a list of user history items into a single string persona."""
        if not history:
            return "This user has no available post history."
            
        formatted_items = []
        for item in history:
            item_type = item.get('type', 'Post').capitalize()
            content = item.get('content', '')
            formatted_items.append(f"- {item_type}: \"{content}\"")
            
        return "Here are some examples of the user's past posts and replies:\n" + "\n".join(formatted_items)

    def create_history_only_persona(self, user_file_path: str, history_length: int = 20) -> str:
        """Creates a persona string from a user's post history."""
        logger.debug(f"Creating history-only persona for {user_file_path}")
        user_history = get_formatted_user_historie(user_file_path)
        
        if len(user_history) < 1:
            logger.warning(f"Insufficient history for {user_file_path}, returning empty persona.")
            return "This user has no available post history."
            
        # Take the most recent items up to history_length
        recent_history = user_history[-history_length:]
        return self.format_history_for_persona(recent_history)

    def _get_condition_persona(self, condition_type: str, user_file_path: str, config: Dict[str, Any]) -> Optional[str]:
        """Dynamically gets the persona for a given condition."""
        condition_info = self.baseline_conditions[condition_type]
        persona_source = condition_info['persona']
        
        if persona_source == "[DYNAMIC]":
            if condition_type == "history_only":
                history_config = config.get('persona_generation', {})
                history_length = history_config.get('history_length', 20)
                return self.create_history_only_persona(user_file_path, history_length)
            elif condition_type == "best_persona":
                # Placeholder: In a real run, this would load the best persona from a file.
                # For testing, we return a predictable string.
                logger.warning("Using placeholder for 'best_persona'. Implement loading logic for production.")
                return f"This is the best-performing persona for user {os.path.basename(user_file_path)}."
            else:
                raise ValueError(f"Unknown dynamic persona type: {condition_type}")
        else:
            # Return static persona (e.g., None for no_persona, text for generic_persona)
            return persona_source

    def split_holdout_balanced(self, user_file_path: str, seed: int = 42, num_conditions: int = 2) -> Dict[str, List[Tuple]]:
        """
        Split holdout items into N balanced groups for baseline comparison.
        
        Args:
            user_file_path: Path to user JSONL file
            seed: Random seed for reproducible splits
            num_conditions: The number of groups to split the data into (e.g., 2 or 4)
            
        Returns:
            Dict with condition keys, values are lists of (stimulus, is_post, post_id)
        """
        logger.info(f"Creating {num_conditions}-way balanced holdout split for {os.path.basename(user_file_path)}")
        all_stimuli = load_stimulus(user_file_path)
        
        condition_names = list(self.baseline_conditions.keys())[:num_conditions]
        
        if len(all_stimuli) < num_conditions:
            logger.warning(f"Insufficient stimuli ({len(all_stimuli)}) for a {num_conditions}-way split.")
            return {name: [] for name in condition_names}
        
        posts = [item for item in all_stimuli if item[1] == True]
        replies = [item for item in all_stimuli if item[1] == False]
        
        random.seed(seed)
        random.shuffle(posts)
        random.shuffle(replies)
        
        # Initialize dictionary to hold items for each condition
        splits = {name: [] for name in condition_names}
        
        # Distribute posts
        for i, post in enumerate(posts):
            condition_key = condition_names[i % num_conditions]
            splits[condition_key].append(post)
            
        # Distribute replies
        for i, reply in enumerate(replies):
            condition_key = condition_names[i % num_conditions]
            splits[condition_key].append(reply)
            
        # Log and return final splits
        for name, items in splits.items():
            random.shuffle(items) # Shuffle within each condition
            num_posts = sum(1 for item in items if item[1] == True)
            num_replies = len(items) - num_posts
            logger.info(f"Split created - {name}: {len(items)} items ({num_posts} posts, {num_replies} replies)")
            
        return splits
    
    def process_baseline_condition(
        self, 
        user_file_path: str, 
        condition_type: str, 
        stimuli_items: List[Tuple],
        config: Dict[str, Any],
        run_id: str
    ) -> bool:
        """Process a single baseline condition for one user."""
        user_file = os.path.basename(user_file_path)
        condition_info = self.baseline_conditions[condition_type]
        
        logger.info(f"Processing {condition_info['name']} for {user_file}")
        
        # Get persona for this condition, generating it dynamically if needed
        persona = self._get_condition_persona(condition_type, user_file_path, config)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_stimulus = {
                executor.submit(
                    self._process_single_baseline_stimulus,
                    stimulus_data, condition_type, persona, config, user_file_path, run_id
                ): stimulus_data for stimulus_data in stimuli_items
            }
            
            successful_count = 0
            for future in concurrent.futures.as_completed(future_to_stimulus):
                stimulus_data = future_to_stimulus[future]
                _, _, post_id = stimulus_data
                try:
                    if future.result():
                        successful_count += 1
                except Exception as exc:
                    logger.error(f"Stimulus {post_id} generated an exception: {exc}")
        
        logger.info(f"Completed {condition_type}: {successful_count}/{len(stimuli_items)} stimuli processed")
        
        if successful_count > 0:
            try:
                self._evaluate_baseline_condition(user_file_path, run_id)
                logger.info(f"Evaluation completed for {condition_type} under run_id {run_id}")
                return True
            except Exception as e:
                logger.error(f"Evaluation failed for {condition_type}: {e}")
                return False
        return successful_count > 0

    def _process_single_baseline_stimulus(
        self,
        stimulus_data: Tuple,
        condition_type: str, 
        persona: Optional[str],
        config: Dict[str, Any],
        user_file_path: str,
        run_id: str
    ) -> bool:
        """Process a single stimulus for a baseline condition."""
        stimulus, is_post, post_id = stimulus_data
        condition_info = self.baseline_conditions[condition_type]
        llm_config = config.get('llm', {})
        
        try:
            template_name = condition_info['post_template'] if is_post else condition_info['reply_template']
            
            template_params = {"tweet": stimulus}
            if persona is not None:
                template_params["persona"] = persona
                
            stimulus_formatted = templates.format_template(template_name, **template_params)
            
            imitation_model = llm_config.get('imitation_model', 'ollama')
            imitation = call_ai(stimulus_formatted, imitation_model)
            
            # Use condition marker if no persona string is available (e.g., for No-Persona)
            persona_for_saving = persona or f"[{condition_type}]"
            
            with self.file_lock:
                save_user_imitation(
                    file_path=user_file_path,
                    stimulus=stimulus,
                    persona=persona_for_saving,
                    imitation=imitation,
                    run_id=run_id,
                    tweet_id=post_id
                )
            return True
            
        except Exception as e:
            logger.error(f"Error processing {condition_type} stimulus {post_id}: {e}", exc_info=True)
            return False
    
    def _evaluate_baseline_condition(self, user_file_path: str, run_id: str) -> None:
        """Evaluate baseline condition using existing evaluation infrastructure."""
        from loader import load_predictions_orginales_formated
        
        results = load_predictions_orginales_formated(run_id=run_id, file_path=user_file_path)
        if not results:
            logger.warning(f"No results found for evaluation of run_id {run_id} on file {user_file_path}")
            return
        
        evaluation_result = evaluate_with_individual_scores(results)
        save_evaluation_results(
            file_path=user_file_path, 
            evaluation_results=evaluation_result, 
            run_id=run_id
        )
        logger.debug(f"Evaluation completed for run_id {run_id}")
    
    def run_baseline_experiment(self, config: Dict[str, Any], extended_mode: bool = False) -> Dict[str, Any]:
        """Run complete baseline experiment across all users."""
        mode_desc = "4-condition (V2)" if extended_mode else "2-condition (V1)"
        logger.info(f"Starting baseline experiment - {mode_desc}")
        
        experiment_config = config.get('experiment', {})
        number_of_users = experiment_config.get('number_of_users')
        users_dict_path = experiment_config.get('users_dict')
        
        user_files = [f for f in os.listdir(users_dict_path) 
                     if f.endswith('.jsonl')][:number_of_users]
        random.shuffle(user_files)
        
        num_conditions = 4 if extended_mode else 2
        condition_names = list(self.baseline_conditions.keys())[:num_conditions]
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_ids = {name: f"baseline_{name}_{timestamp}" for name in condition_names}
        logger.info(f"Baseline run IDs: {run_ids}")
        
        results = {"processed_users": [], "failed_users": [], "run_ids": run_ids}
        
        for user_file in user_files:
            user_file_path = os.path.join(users_dict_path, user_file)
            user_id = user_file.replace('.jsonl', '')
            try:
                logger.info(f"Processing baseline experiment for user: {user_id}")
                split_data = self.split_holdout_balanced(user_file_path, num_conditions=num_conditions)
                
                success_count = 0
                for condition_type, stimuli_items in split_data.items():
                    if not stimuli_items:
                        logger.warning(f"No stimuli for condition '{condition_type}' for user {user_id}. Skipping.")
                        continue
                    
                    if self.process_baseline_condition(user_file_path, condition_type, stimuli_items, config, run_ids[condition_type]):
                        success_count += 1
                
                if success_count == num_conditions:
                    results["processed_users"].append(user_id)
                else:
                    results["failed_users"].append(user_id)
                    
            except Exception as e:
                logger.error(f"Unhandled error processing user {user_id}: {e}", exc_info=True)
                results["failed_users"].append(user_id)
        
        logger.info(f"Baseline experiment completed ({mode_desc}): {len(results['processed_users'])} successful, {len(results['failed_users'])} failed.")
        return results