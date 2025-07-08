import os
import templates
import loader
from llm import call_ai
from datetime import datetime
from saver import save_user_imitation, save_evaluation_results, save_reflection_results
from logging_config import logger
from eval import evaluate, evaluate_with_individual_scores
from utils import load_config
import random
import concurrent.futures
import threading


# Thread lock for file operations to ensure thread safety
file_lock = threading.Lock()


def process_single_stimulus(stimulus_data, persona, config, user_file_path, run_id):
    """
    Process a single stimulus in parallel.
    
    :param stimulus_data: Tuple of (stimulus, is_post, post_id)
    :param persona: User persona
    :param config: Configuration dictionary
    :param user_file_path: Path to user file
    :param run_id: Current run ID
    :return: Success status
    """
    stimulus, is_post, post_id = stimulus_data
    template_config = config.get('templates', {})
    llm_config = config.get('llm', {})
    
    try:
        logger.debug(f"Processing stimulus {post_id}: {stimulus}, is_post: {is_post}")
        
        # Format the stimulus template
        if is_post:
            stimulus_formatted = templates.format_template(
                template_config.get('imitation_post_template', 'imitation_post_template_simple'),
                persona=persona,
                tweet=stimulus
            )
        else:
            stimulus_formatted = templates.format_template(
                template_config.get('imitation_reply_template', 'imitation_replies_template_simple'),
                persona=persona,
                tweet=stimulus
            )
        
        # Call AI model
        imitation_model = llm_config.get('imitation_model', 'ollama')
        imitation = call_ai(stimulus_formatted, imitation_model)
        logger.debug(f"Imitation for post/reply {post_id}:\n{imitation}")
        
        # Save results with thread safety
        with file_lock:
            save_user_imitation(
                file_path=user_file_path,
                stimulus=stimulus,
                persona=persona,
                imitation=imitation,
                run_id=run_id,
                tweet_id=post_id
            )
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing stimulus {post_id}: {e}")
        return False


def process_user(user_file_path, config, run_id):
    """
    Führt die Verarbeitung für einen einzelnen Benutzer durch.
    """
    user_file = os.path.basename(user_file_path)
    logger.debug(f"Processing user: {user_file}")

    experiment_config = config.get('experiment', {})
    llm_config = config.get('llm', {})
    template_config = config.get('templates', {})
    num_stimuli_to_process = experiment_config.get('num_stimuli_to_process', 3)
    number_of_rounds = experiment_config.get('number_of_rounds')

    # --- Persona Generierung ---
    logger.info(f"Generating persona for {user_file}...")
    user_history = loader.get_formatted_user_historie(user_file_path)
    formatted_user_history = templates.format_template(
        template_config.get('persona_template', 'persona_template_simple'),
        historie=user_history
    )
    persona_model = llm_config.get('persona_model', 'google')
    persona = call_ai(formatted_user_history, persona_model)
    logger.debug(f"Persona for user {user_file.split('.')[0]}:\n{persona}")

    # Mittlere Schleife: Iteration über Runden pro Benutzer
    for round_num in range(1, number_of_rounds + 1):
        logger.debug(f"Starting round {round_num}/{number_of_rounds} for user {user_file}")

        # --- Imitation Generierung ---
        logger.info(f"Starting imitation generation for {user_file}...")
        all_stimuli = loader.load_stimulus(user_file_path)

        # Parallel processing of stimuli using ThreadPoolExecutor
        stimuli_to_process = all_stimuli[:num_stimuli_to_process]
        logger.info(f"Processing {len(stimuli_to_process)} stimuli in parallel with up to 4 threads...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all stimuli for parallel processing
            future_to_stimulus = {
                executor.submit(process_single_stimulus, stimulus_data, persona, config, user_file_path, run_id): stimulus_data
                for stimulus_data in stimuli_to_process
            }
            
            # Wait for all tasks to complete and handle results
            successful_count = 0
            for future in concurrent.futures.as_completed(future_to_stimulus):
                stimulus_data = future_to_stimulus[future]
                _, _, post_id = stimulus_data
                try:
                    result = future.result()
                    if result:
                        successful_count += 1
                        logger.debug(f"Successfully processed stimulus {post_id}")
                    else:
                        logger.warning(f"Failed to process stimulus {post_id}")
                except Exception as exc:
                    logger.error(f"Stimulus {post_id} generated an exception: {exc}")
            
            logger.info(f"Completed parallel processing: {successful_count}/{len(stimuli_to_process)} stimuli processed successfully")

        # --- Evaluation ---
        logger.info(f"Starting evaluation for {user_file}...")
        results = loader.load_predictions_orginales_formated(run_id=run_id, file_path=user_file_path)
        logger.debug(f"Results for run_id {run_id}:")
        evaluation_result = evaluate_with_individual_scores(results)
        save_evaluation_results(file_path=user_file_path, evaluation_results=evaluation_result, run_id=run_id)
        logger.debug(f"Evaluation results saved for run_id {run_id}: {evaluation_result}")

        # --- Reflection (nur wenn nicht die letzte Runde) ---
        if round_num < number_of_rounds:
            logger.info(f"Starting reflection for {user_file}...")
            data_for_reflection = loader.load_results_for_reflection(run_id, user_file_path)
            reflection_template = templates.format_template(
                template_config.get('reflection_template', 'reflect_results_template'),
                **data_for_reflection
            )
            reflection_model = llm_config.get('reflection_model', 'google_json')
            improved_persona = call_ai(reflection_template, reflection_model)

            try:
                save_reflection_results(
                    file_path=user_file_path,
                    run_id=run_id,
                    reflections=improved_persona,
                    iteration=round_num
                )
                logger.debug(f"Reflection results saved for run_id {run_id}, iteration {round_num}.")
            except Exception as e:
                logger.error(f"Error saving reflection results: {e}")

            # Lade die neueste verbesserte Persona nach der Reflexion
            persona = loader.load_latest_improved_persona(run_id=run_id, file_path=user_file_path)
            logger.debug("Persona updated with reflection results (if available).")
        
        logger.debug(f"Completed round {round_num}/{number_of_rounds} for user {user_file}")
    
    logger.info(f"Completed all rounds for user {user_file}")
    return True


def run_experiment(config):
    """
    Führt das gesamte Experiment basierend auf der geladenen Konfiguration aus.
    """
    logger.info("Starting experiment run...")

    # Konfigurationsparameter extrahieren
    experiment_config = config.get('experiment', {})
    number_of_users = experiment_config.get('number_of_users')
    users_dict_path = experiment_config.get('users_dict')
    run_id = experiment_config.get('run_name_prefix')
    max_workers = experiment_config.get('max_parallel_users', 4)

    if not users_dict_path:
        logger.error("Fehler: 'users_dict' nicht in der Konfiguration gefunden.")
        return

    if not run_id or str(run_id).lower() == 'none':
        run_id = str(datetime.now().strftime('%Y%m%d_%H%M%S'))
    
    if not os.path.exists(users_dict_path):
        logger.error(f"Path does not exist: {users_dict_path}")
        return
    else:
        files = [f for f in os.listdir(users_dict_path) if os.path.isfile(os.path.join(users_dict_path, f))]
        if len(files) < 1:
            logger.error(f"Keine dateien im angegebenen directory")
            return
        logger.info(f"Found {len(files)} Users in {users_dict_path}")
    
    random.shuffle(files)
    
    logger.info(f"Starting experiment with up to {max_workers} parallel user processes.")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_user = {
            executor.submit(process_user, os.path.join(users_dict_path, user_file), config, run_id): user_file
            for user_file in files[:number_of_users]
        }

        for future in concurrent.futures.as_completed(future_to_user):
            user_file = future_to_user[future]
            try:
                future.result()
                logger.info(f"Successfully completed processing for user {user_file}.")
            except Exception as exc:
                logger.error(f"User {user_file} generated an exception: {exc}")

    logger.info("Experiment completed for all users.")


if __name__ == "__main__":
    logger.info('Starting...')
    import argparse
    parser = argparse.ArgumentParser(description="Run experiment with YAML configuration.")
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to the YAML configuration file.')
    args = parser.parse_args()

    # Konfiguration laden
    experiment_config = load_config(args.config)

    # Experiment ausführen
    run_experiment(experiment_config)