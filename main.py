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


def run_experiment(config):
    """
    Führt das gesamte Experiment basierend auf der geladenen Konfiguration aus.
    """
    logger.info("Starting experiment run...")

    # Konfigurationsparameter extrahieren
    experiment_config = config.get('experiment', {})
    llm_config = config.get('llm', {})
    template_config = config.get('templates', {})
    
    num_stimuli_to_process = experiment_config.get('num_stimuli_to_process', 3)
    number_of_users = experiment_config.get('number_of_users')
    number_of_rounds = experiment_config.get('number_of_rounds')
    users_dict_path = experiment_config.get('users_dict')  # Umbenannt für Klarheit
    run_id = f"{experiment_config.get('run_name_prefix')}"
    
    if not users_dict_path:
        logger.error("Fehler: 'users_dict' nicht in der Konfiguration gefunden.")
        return

    if not run_id:
        run_id = str(datetime.now().strftime('%Y%m%d_%H%M%S'))
    
    # Check if path contains files
    if not os.path.exists(users_dict_path):
        logger.error(f"Path does not exist: {users_dict_path}")
        return
    else:
        files = [f for f in os.listdir(users_dict_path) if os.path.isfile(os.path.join(users_dict_path, f))]
        if len(files) < 1:
            logger.error(f"Keine dateien im angegebenen directory")
            return
        logger.info(f"Found {len(files)} Users in {users_dict_path}")
    
    # Shuffle list of user files
    random.shuffle(files)
    
    # Äußere Schleife: Iteration über Benutzer
    for user_file in files[:number_of_users]:
        user_file_path = os.path.join(users_dict_path, user_file)
        logger.info(f"Processing user: {user_file}")
        
        # --- Persona Generierung ---
        logger.info("Generating persona...")
        user_history = loader.get_formatted_user_historie(user_file_path)
        formatted_user_history = templates.format_template(
            template_config.get('persona_template', 'persona_template_simple'),
            historie=user_history
        )
        persona_model = llm_config.get('persona_model', 'google')
        persona = call_ai(formatted_user_history, persona_model)
        print(f"Persona for user {user_file.split('.')[0]}:\n{persona}")
        
        # Mittlere Schleife: Iteration über Runden pro Benutzer
        for round_num in range(1, number_of_rounds + 1):
            logger.info(f"Starting round {round_num}/{number_of_rounds} for user {user_file}")
            
            # --- Imitation Generierung ---
            logger.info("Starting imitation generation...")
            all_stimuli = loader.load_stimulus(user_file_path)

            # Innere Schleife: Iteration über Stimuli
            for i, x in enumerate(all_stimuli[:num_stimuli_to_process]):
                stimulus, is_post, post_id = x
                print(f"Processing Stimulus {i+1}/{num_stimuli_to_process}: {stimulus}, is_post: {is_post}, post_id: {post_id}")

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

                imitation_model = llm_config.get('imitation_model', 'ollama')
                imitation = call_ai(stimulus_formatted, imitation_model)
                print(f"Imitation for post/reply {post_id}:\n{imitation}")

                save_user_imitation(
                    file_path=user_file_path,
                    stimulus=stimulus,
                    persona=persona,
                    imitation=imitation,
                    run_id=run_id,
                    tweet_id=post_id
                )

            # --- Evaluation ---
            logger.info("Starting evaluation...")
            results = loader.load_predictions_orginales_formated(run_id=run_id, file_path=user_file_path)
            print(f"Results for run_id {run_id}:")
            evaluation_result = evaluate_with_individual_scores(results)
            save_evaluation_results(file_path=user_file_path, evaluation_results=evaluation_result, run_id=run_id)
            print(f"Evaluation results saved for run_id {run_id}: {evaluation_result}")

            # --- Reflection (nur wenn nicht die letzte Runde) ---
            if round_num < number_of_rounds:
                logger.info("Starting reflection...")
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
                    print(f"Reflection results saved for run_id {run_id}, iteration {round_num}.")
                except Exception as e:
                    logger.error(f"Error saving reflection results: {e}")

                # Lade die neueste verbesserte Persona nach der Reflexion
                persona = loader.load_latest_improved_persona(run_id=run_id, file_path=user_file_path)
                print("Persona updated with reflection results (if available).")
            
            logger.info(f"Completed round {round_num}/{number_of_rounds} for user {user_file}")
        
        logger.info(f"Completed all rounds for user {user_file}")
    
    logger.info("Experiment completed for all users.")


if __name__ == "__main__":
    print('starte ...')
    import argparse
    parser = argparse.ArgumentParser(description="Run experiment with YAML configuration.")
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to the YAML configuration file.')
    args = parser.parse_args()

    # Konfiguration laden
    experiment_config = load_config(args.config)

    # Experiment ausführen
    run_experiment(experiment_config)