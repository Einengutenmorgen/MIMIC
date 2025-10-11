# main.py

import argparse
import random
import json
from typing import List, Dict, Any

# Importiere alle unsere Bausteine
from user_selector import UserSelector
from persona_creation import PersonaCreationPipeline
from imitation_pipeline import ImitationPipeline
from evaluation_pipeline import EvaluationPipeline, BleuMetric, RougeMetric, LlmJudgeMetric
from persona_improvement import PersonaImprovementPipeline
from db_loader import DbLoader
from db_saver import DbSaver
from llm_handler import LlmHandler
from masking_pipeline import MaskingPipeline # <-- Wichtiger neuer Import

def parse_arguments():
    """
    Definiert und liest Kommandozeilen-Argumente.
    """
    parser = argparse.ArgumentParser(description="MIMIC v.02 Experiment Runner")
    parser.add_argument("--user_id", type=int, help="Eine spezifische User-ID für das Experiment.")
    parser.add_argument("--task_type", type=str, default="reply_generation", choices=["reply_generation", "post_completion"], help="Die Art der auszuführenden Aufgabe.")
    parser.add_argument("--rounds", type=int, default=2, help="Anzahl der Verbesserungs-Runden.")
    parser.add_argument("--imitations", type=float, default=0.1, help="Anteil (0.0-1.0) oder Anzahl (int) an Imitationen pro Runde.")
    parser.add_argument("--metrics", nargs='+', default=['bleu', 'rouge'], help="Liste der Metriken (bleu, rouge, llm_judge).")
    parser.add_argument("--exp_name", type=str, default="MIMIC Experiment", help="Ein Name für das Experiment.")
    parser.add_argument("--task_type", type=str, default="contextual_reply", 
                        choices=["style_imitation", "post_completion", "contextual_reply"], 
                        help="Die Art der auszuführenden Aufgabe.")
    return parser.parse_args()

def select_user(config_user_id: int) -> int:
    """Wählt einen Benutzer basierend auf der Konfiguration aus."""
    if config_user_id:
        print(f"Verwende vordefinierte User-ID: {config_user_id}")
        return config_user_id
    else:
        print("Suche nach einem zufälligen qualifizierten Benutzer (min. 10 History, 5 Holdout)...")
        selector = UserSelector()
        selected_user_id = selector.get_random_qualified_user(10, 5)
        if not selected_user_id:
            raise ValueError("Keine qualifizierten Benutzer für das Experiment gefunden.")
        print(f"Zufällig ausgewählter Benutzer für das Experiment: {selected_user_id}")
        return selected_user_id

def get_metrics(metric_names: List[str], llm_handler: LlmHandler) -> List[Any]:
    metric_map = {'bleu': BleuMetric, 'rouge': RougeMetric, 'llm_judge': lambda: LlmJudgeMetric(llm_handler)}
    metrics = [metric_map[name]() if name != 'llm_judge' else metric_map[name]() for name in metric_names if name in metric_map]
    if not metrics: raise ValueError("Keine gültigen Metriken angegeben.")
    return metrics

def calculate_average_scores(all_evaluations: List[Dict[str, float]]) -> Dict[str, float]:
    if not all_evaluations: return {}
    avg_scores = {key: sum(d[key] for d in all_evaluations) / len(all_evaluations) for key in all_evaluations[0]}
    return avg_scores

def main():
    args = parse_arguments()
    
    # --- 1. SETUP ---
    print("--- 1. INITIALISIERE PIPELINES UND KOMPONENTEN ---")
    llm_handler = LlmHandler()
    loader = DbLoader()
    saver = DbSaver()
    user_id = select_user(args.user_id)
    metrics = get_metrics(args.metrics, llm_handler)
    
    persona_pipeline = PersonaCreationPipeline()
    imitation_pipeline = ImitationPipeline()
    eval_pipeline = EvaluationPipeline(metrics=metrics)
    improvement_pipeline = PersonaImprovementPipeline()
    masking_pipeline = MaskingPipeline() if args.task_type == 'post_completion' else None

    # --- 2. EXPERIMENT STARTEN ---
    print(f"\n--- 2. STARTE EXPERIMENT '{args.exp_name}' ---")
    experiment_id = saver.save_experiment(name=args.exp_name, strategy=f"{args.task_type}_iterative")
    print(f"Experiment in DB gespeichert mit ID: {experiment_id}")

    # --- 3. DATEN LADEN UND VORBEREITEN ---
    print(f"\n--- 3. LADE DATEN FÜR BENUTZER {user_id} ---")
    
    # ÄNDERUNG: Lade die Daten basierend auf der Aufgabe
    if args.task_type == 'contextual_reply':
        stimulus_pool = loader.get_reply_stimuli(user_id, limit=50) # Lade bis zu 50 Antworten mit Kontext
    else:
        holdout_tweets = loader.get_tweets_by_user(user_id, is_holdout=True)
        stimulus_pool = random.sample(holdout_tweets, min(50, len(holdout_tweets))) # Nehmen wir max. 50

    num_imitations = int(len(stimulus_pool) * args.imitations) if 0.0 < args.imitations <= 1.0 else int(args.imitations)
    if num_imitations == 0: raise ValueError("Anzahl der Imitationen ist 0.")
    
    # Wähle die finale Stichprobe für die Runde
    stimulus_sample = random.sample(stimulus_pool, min(num_imitations, len(stimulus_pool)))
    
    # Bedingte Datenvorbereitung für Post Completion
    if args.task_type == 'post_completion':
        print("\n--- VORBEREITUNG: MASKIERE STIMULUS-TWEETS ---")
        stimulus_sample = masking_pipeline.process_batch(stimulus_pool)

    print(f"Es werden {len(stimulus_sample)} Imitationen pro Runde erzeugt.")

    # --- 4. ITERATIVER PROZESS ---
    current_persona = ""
    for i in range(args.rounds):
        round_num = i + 1
        print(f"\n{'='*20} RUNDE {round_num}/{args.rounds} {'='*20}")

        if round_num == 1:
            current_persona = persona_pipeline.create_persona_for_user(user_id)
        else:
            current_persona = improvement_pipeline.improve_persona(
                persona_description=current_persona,
                evaluation_results=avg_round_scores,
                ground_truth_example=last_ground_truth,
                imitation_example=last_imitation,
            )
        round_id = saver.save_round(experiment_id, user_id, round_num, current_persona)
        print(f"Runde {round_num} in DB gespeichert mit ID: {round_id}")

        round_evaluations = []
        for j, stimulus_data in enumerate(stimulus_sample):
            imitation = imitation_pipeline.generate_imitation(current_persona, stimulus_data, args.task_type)
            
            # Ground truth hängt jetzt von 3 Aufgabentypen ab
            if args.task_type == 'post_completion':
                original_tweet_id = stimulus_data['original_tweet_id']
                ground_truth_text = " ".join(stimulus_data['original_words'])
            elif args.task_type == 'contextual_reply':
                # Ground Truth ist die ECHTE ANTWORT des Nutzers
                original_tweet_id = stimulus_data['stimulus_tweet']['tweet_id']
                ground_truth_text = stimulus_data['stimulus_tweet']['full_text']
            else: # style_imitation
                original_tweet_id = stimulus_data['tweet_id']
                ground_truth_text = stimulus_data['full_text']

            evaluation_scores = eval_pipeline.evaluate(ground_truth_text, imitation)
            
            saver.save_imitation_and_evaluation(
                round_id=round_id,
                original_tweet_id=original_tweet_id,
                generated_text=imitation,
                task_type=args.task_type,
                evaluation_scores=evaluation_scores
            )
            print("Imitation und Evaluation in DB gespeichert.")
            round_evaluations.append(evaluation_scores)
            
            last_ground_truth = ground_truth_text
            last_imitation = imitation
            
        avg_round_scores = calculate_average_scores(round_evaluations)
        print(f"\n--- DURCHSCHNITTS-SCORES FÜR RUNDE {round_num} ---")
        print(avg_round_scores)

    print(f"\n{'='*20} EXPERIMENT ABGESCHLOSSEN {'='*20}")

if __name__ == "__main__":
    main()