import os
import json
import numpy as np
from logging_config import logger
import pandas as pd
import pathlib
from file_cache import get_file_cache

#helper
def numpy_json_handler(obj):
    """Convert numpy objects to Python native types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):  # numpy scalars
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


#def save user imitation 

# def save_user_imitation(file_path, stimulus, persona, imitation, run_id, tweet_id):
#     """
#     Save user imitation data
#     Stimiulus 
#     Persona
    
#     :param file_path: Path to the JSON file where user imitation data will be saved.
#     :return: None
#     """
#     if not os.path.exists(file_path):
#         logger.error(f"File not found: {file_path}")

#     with open(file_path, 'r', encoding='utf-8') as file:
#         first_line = file.readline()
#         data = json.loads(first_line)
#         user_id = data['user_id']
    
    

    
#     with open(file_path, 'a', encoding='utf-8') as file:
#         try:
#             #formatt data 
#             imitations = {'tweet_id': tweet_id, 'stimulus': stimulus, 'imitation': imitation}
#             run_data = {
#                 "run_id": run_id,
#                 "persona": persona,
#                 "imitations": [imitations]
#                 }
#             runs = {
#                 "user_id": user_id,
#                 "runs": [run_data]
#                 }
#             logger.info(f"Data loaded. Try to append data to file: {file_path} ...")
#             file.write(json.dumps(runs, ensure_ascii=False) + "\n")
#             logger.info(f"Data successfully appended")
#         except json.JSONDecodeError as e:
#             logger.error(f"Error decoding JSON from {file_path}: {e}")
    
def save_user_imitation(file_path, stimulus, persona, imitation, run_id, tweet_id):
    """
    Optimized version using file cache to minimize file I/O operations
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return

    cache = get_file_cache()
    
    try:
        # Use cached data to avoid repeated file reads
        cached_data = cache.read_file_with_cache(file_path)
        if cached_data is None:
            logger.error(f"Failed to read file {file_path}")
            return
        
        # Get user_id from first line
        if 0 not in cached_data:
            logger.error(f"File {file_path} is empty")
            return
        
        first_line_data = cached_data[0]
        if first_line_data is None:
            logger.error(f"Failed to parse first line in {file_path}")
            return
        
        user_id = first_line_data.get('user_id')
        if not user_id:
            logger.error(f"No user_id found in first line of {file_path}")
            return
        
        # Create new imitation
        new_imitation = {
            'tweet_id': tweet_id,
            'stimulus': stimulus,
            'imitation': imitation
        }
        
        # Load or create runs data (line 3, index 2)
        runs_line_index = 2
        
        if runs_line_index in cached_data and cached_data[runs_line_index] is not None:
            runs_data = cached_data[runs_line_index]
        else:
            runs_data = {"user_id": user_id, "runs": []}
        
        # Find or create run
        target_run = None
        for run in runs_data['runs']:
            if run['run_id'] == run_id:
                target_run = run
                break
        
        if target_run is None:
            target_run = {
                "run_id": run_id,
                "persona": persona,
                "imitations": []
            }
            runs_data['runs'].append(target_run)
        
        # Add imitation
        target_run['imitations'].append(new_imitation)
        
        # Prepare lines to write
        lines_to_write = []
        
        # Add existing lines up to runs_line_index
        for i in range(runs_line_index):
            if i in cached_data and cached_data[i] is not None:
                lines_to_write.append(json.dumps(cached_data[i], ensure_ascii=False) + "\n")
            else:
                lines_to_write.append("\n")
        
        # Add updated runs line
        lines_to_write.append(json.dumps(runs_data, ensure_ascii=False) + "\n")
        
        # Add remaining lines if they exist
        max_line_index = max(cached_data.keys()) if cached_data else 0
        for i in range(runs_line_index + 1, max_line_index + 1):
            if i in cached_data and cached_data[i] is not None:
                lines_to_write.append(json.dumps(cached_data[i], ensure_ascii=False) + "\n")
            else:
                lines_to_write.append("\n")
        
        # Write back using cache (this will invalidate the cache)
        success = cache.write_file_with_cache_invalidation(file_path, lines_to_write)
        
        if success:
            logger.info(f"Data successfully updated in {file_path} for run_id: {run_id}")
        else:
            logger.error(f"Failed to write file {file_path}")
        
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")


def save_evaluation_results(file_path, evaluation_results, run_id):
    """
    Save evaluation results to a JSON file without overwriting existing evaluations.
    
    :param file_path: Path to the JSON file where evaluation results will be saved.
    :param evaluation_results: Dictionary containing evaluation results.
    :param run_id: ID of the current run
    :return: None
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return
    if not isinstance(evaluation_results, dict):
        logger.error("Evaluation results must be a dictionary.")
        return
    
    cache = get_file_cache()
    
    try:
        # Use cached data to avoid repeated file reads
        cached_data = cache.read_file_with_cache(file_path)
        if cached_data is None:
            logger.error(f"Failed to read file {file_path}")
            return
        
        # Create new evaluation
        new_evaluation = {
            "run_id": run_id,
            "timestamp": pd.Timestamp.now().isoformat(),
            "evaluation_results": evaluation_results
        }
        
        # Determine which line is used for evaluations (line 4, index 3)
        evaluations_line_index = 3
        
        # Load or create evaluations data
        if evaluations_line_index in cached_data and cached_data[evaluations_line_index] is not None:
            evaluations_data = cached_data[evaluations_line_index]
        else:
            # Create new evaluations structure
            if 0 not in cached_data or cached_data[0] is None:
                logger.error(f"Failed to parse first line in {file_path}")
                return
            
            first_line_data = cached_data[0]
            user_id = first_line_data.get('user_id')
            if not user_id:
                logger.error(f"No user_id found in first line of {file_path}")
                return
            
            evaluations_data = {
                "user_id": user_id,
                "evaluations": []
            }
        
        # Check if evaluation for this run_id already exists
        existing_eval_index = None
        for i, eval_entry in enumerate(evaluations_data['evaluations']):
            if eval_entry['run_id'] == run_id:
                existing_eval_index = i
                break
        
        if existing_eval_index is not None:
            # Update existing evaluation for this run_id
            evaluations_data['evaluations'][existing_eval_index] = new_evaluation
            logger.info(f"Updated existing evaluation for run_id: {run_id}")
        else:
            # Add new evaluation
            evaluations_data['evaluations'].append(new_evaluation)
            logger.info(f"Added new evaluation for run_id: {run_id}")
        
        # Prepare lines to write
        lines_to_write = []
        
        # Add existing lines up to evaluations_line_index
        for i in range(evaluations_line_index):
            if i in cached_data and cached_data[i] is not None:
                lines_to_write.append(json.dumps(cached_data[i], ensure_ascii=False) + "\n")
            else:
                lines_to_write.append("\n")
        
        # Add updated evaluations line
        lines_to_write.append(json.dumps(evaluations_data, ensure_ascii=False) + "\n")
        
        # Add remaining lines if they exist
        max_line_index = max(cached_data.keys()) if cached_data else 0
        for i in range(evaluations_line_index + 1, max_line_index + 1):
            if i in cached_data and cached_data[i] is not None:
                lines_to_write.append(json.dumps(cached_data[i], ensure_ascii=False) + "\n")
            else:
                lines_to_write.append("\n")
        
        # Write back using cache (this will invalidate the cache)
        success = cache.write_file_with_cache_invalidation(file_path, lines_to_write)
        
        if success:
            logger.info(f"Evaluation results successfully saved to {file_path}")
        else:
            logger.error(f"Failed to write file {file_path}")
        
    except Exception as e:
        logger.error(f"Error saving evaluation results to {file_path}: {e}")





def save_reflection_results(file_path: str, reflections: dict, run_id: str, iteration: int):
    """
    Save reflection on imitation results to a JSON file.
    
    :param file_path: Path to the JSON file
    :param reflections: Dictionary containing API call results (e.g., {'Reflection': '...', 'improved_persona': '...'})
    :param run_id: ID of the current run
    :param iteration: Iteration number for this reflection
    :return: None
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return
    
    cache = get_file_cache()
    
    try:
        # Extract reflection data
        reflection_on_results = reflections['Reflection']
        improved_persona = reflections['improved_persona']
    except KeyError as e:
        logger.error(f"Missing key in reflections dictionary: {e}")
        return
    
    try:
        # Use cached data to avoid repeated file reads
        cached_data = cache.read_file_with_cache(file_path)
        if cached_data is None:
            logger.error(f"Failed to read file {file_path}")
            return
        
        # Load or create reflections data
        reflections_line_index = 4
        if reflections_line_index in cached_data and cached_data[reflections_line_index] is not None:
            reflections_data = cached_data[reflections_line_index]
        else:
            reflections_data = {"reflections": []}
        
        # Create new reflection
        new_reflection = {
            "run_id": run_id,
            "iteration": iteration,
            "reflection_results": {
                "reflection_on_results": reflection_on_results,
                "improved_persona": improved_persona
            }
        }
        
        # Check if run_id and iteration combination exists
        existing_index = None
        for i, entry in enumerate(reflections_data.get('reflections', [])):
            if entry.get('run_id') == run_id and entry.get('iteration') == iteration:
                existing_index = i
                break
        
        # Add or update
        if existing_index is not None:
            reflections_data['reflections'][existing_index] = new_reflection
            logger.warning(f"Overwriting existing reflection for run_id: {run_id}, iteration: {iteration}")
        else:
            if 'reflections' not in reflections_data:
                reflections_data['reflections'] = []
            reflections_data['reflections'].append(new_reflection)
            logger.info(f"Added new reflection for run_id: {run_id}, iteration: {iteration}")
        
        # Prepare lines to write
        lines_to_write = []
        
        # Add existing lines up to reflections_line_index
        for i in range(reflections_line_index):
            if i in cached_data and cached_data[i] is not None:
                lines_to_write.append(json.dumps(cached_data[i], ensure_ascii=False) + "\n")
            else:
                lines_to_write.append("\n")
        
        # Add updated reflections line
        lines_to_write.append(json.dumps(reflections_data, ensure_ascii=False, default=numpy_json_handler) + "\n")
        
        # Add remaining lines if they exist
        max_line_index = max(cached_data.keys()) if cached_data else 0
        for i in range(reflections_line_index + 1, max_line_index + 1):
            if i in cached_data and cached_data[i] is not None:
                lines_to_write.append(json.dumps(cached_data[i], ensure_ascii=False) + "\n")
            else:
                lines_to_write.append("\n")
        
        # Write back using cache (this will invalidate the cache)
        success = cache.write_file_with_cache_invalidation(file_path, lines_to_write)
        
        if success:
            logger.info(f"Reflection results saved to {file_path}")
        else:
            logger.error(f"Failed to write file {file_path}")
        
    except Exception as e:
        logger.error(f"Error saving reflection results to {file_path}: {e}")