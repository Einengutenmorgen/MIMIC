import os 
import json
from logging_config import logger
import pandas as pd
import pathlib

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
    Optimierte Version - lädt runs-Daten nur einmal und cached sie
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return

    try:
        # Lese die Datei
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        first_line_data = json.loads(lines[0].strip())
        user_id = first_line_data['user_id']
        
        # Erstelle neue imitation
        new_imitation = {
            'tweet_id': tweet_id, 
            'stimulus': stimulus, 
            'imitation': imitation
        }
        
        # Lade oder erstelle runs-Daten
        runs_data = None
        runs_line_index = 2  # Index der runs-Zeile
        
        if len(lines) > runs_line_index:
            runs_data = json.loads(lines[runs_line_index].strip())
        else:
            runs_data = {"user_id": user_id, "runs": []}
        
        # Finde oder erstelle run
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
        
        # Füge imitation hinzu
        target_run['imitations'].append(new_imitation)
        
        # Überschreibe die runs-Zeile
        lines_to_write = lines[:runs_line_index] if len(lines) > runs_line_index else lines[:]
        lines_to_write.append(json.dumps(runs_data, ensure_ascii=False) + "\n")
        
        # Füge weitere Zeilen hinzu falls vorhanden
        if len(lines) > runs_line_index + 1:
            lines_to_write.extend(lines[runs_line_index + 1:])
        
        # Schreibe zurück
        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(lines_to_write)
        
        logger.info(f"Data successfully updated in {file_path} for run_id: {run_id}")
        
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
    
    try:
        # Lese die bestehende Datei
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        # Erstelle neue evaluation
        new_evaluation = {
            "run_id": run_id,
            "timestamp": pd.Timestamp.now().isoformat(),  # Optional: Zeitstempel hinzufügen
            "evaluation_results": evaluation_results
        }
        
        # Bestimme welche Zeile für evaluations verwendet wird (z.B. Zeile 4, Index 3)
        evaluations_line_index = 3
        
        # Lade oder erstelle evaluations-Daten
        evaluations_data = None
        
        if len(lines) > evaluations_line_index:
            # Lade bestehende evaluations
            evaluations_data = json.loads(lines[evaluations_line_index].strip())
        else:
            # Erstelle neue evaluations-Struktur
            first_line_data = json.loads(lines[0].strip())
            user_id = first_line_data['user_id']
            evaluations_data = {
                "user_id": user_id,
                "evaluations": []
            }
        
        # Prüfe ob bereits eine Evaluation für diese run_id existiert
        existing_eval_index = None
        for i, eval_entry in enumerate(evaluations_data['evaluations']):
            if eval_entry['run_id'] == run_id:
                existing_eval_index = i
                break
        
        if existing_eval_index is not None:
            # Überschreibe bestehende Evaluation für diese run_id
            evaluations_data['evaluations'][existing_eval_index] = new_evaluation
            logger.info(f"Updated existing evaluation for run_id: {run_id}")
        else:
            # Füge neue Evaluation hinzu
            evaluations_data['evaluations'].append(new_evaluation)
            logger.info(f"Added new evaluation for run_id: {run_id}")
        
        # Bereite Zeilen zum Schreiben vor
        lines_to_write = lines[:evaluations_line_index] if len(lines) > evaluations_line_index else lines[:]
        lines_to_write.append(json.dumps(evaluations_data, ensure_ascii=False) + "\n")
        
        # Füge weitere Zeilen hinzu falls vorhanden
        if len(lines) > evaluations_line_index + 1:
            lines_to_write.extend(lines[evaluations_line_index + 1:])
        
        # Schreibe zurück
        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(lines_to_write)
        
        logger.info(f"Evaluation results successfully saved to {file_path}")
        
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
    
    # Clean markdown code blocks
    # cleaned = reflections.strip()
    # if cleaned.startswith('```'):
    #     start = cleaned.find('\n') + 1
    #     end = cleaned.rfind('```')
    #     if start > 0 and end > start:
    #         cleaned = cleaned[start:end].strip()
    
    # Parse JSON
    try:
        # reflections is already a dictionary, as per debugging analysis.
        # Original line: data = json.loads(reflections)
        reflection_on_results = reflections['Reflection']
        improved_persona = reflections['improved_persona']
    except KeyError as e: # Changed to only catch KeyError as json.loads is removed
        logger.error(f"Missing key in reflections dictionary: {e}")
        return
    
    # Read file
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Load or create reflections data
    reflections_line_index = 4
    if len(lines) > reflections_line_index:
        try:
            reflections_data = json.loads(lines[reflections_line_index].strip())
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {file_path} at line {reflections_line_index}: {lines[reflections_line_index]}")
            
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
    
    # Write back
    lines_to_write = lines[:reflections_line_index]
    while len(lines_to_write) < reflections_line_index:
        lines_to_write.append("\n")
    lines_to_write.append(json.dumps(reflections_data, ensure_ascii=False, default=numpy_json_handler) + "\n")
    if len(lines) > reflections_line_index + 1:
        lines_to_write.extend(lines[reflections_line_index + 1:])
    
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(lines_to_write)
    
    logger.info(f"Reflection results saved to {file_path}")