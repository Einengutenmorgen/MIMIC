# persona_improvement.py

from llm_handler import LlmHandler
from typing import Dict, Any

class PersonaImprovementPipeline:
    """
    Orchestriert den Prozess der Verbesserung einer Persona basierend auf
    Evaluationsergebnissen und Beispielen.
    """
    def __init__(self):
        """
        Initialisiert die Pipeline mit den notwendigen Komponenten.
        """
        self.llm_handler = LlmHandler()
        self.improvement_prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        """
        Lädt die Prompt-Vorlage für die Persona-Verbesserung.
        """
        # Dieser Prompt ist bewusst detailliert, um dem LLM den vollen Kontext zu geben.
        template = (
            "Du bist ein Experte für Prompt-Engineering. Deine Aufgabe ist es, eine Persona-Beschreibung für eine KI zu verbessern. "
            "Die KI hat versucht, einen Social-Media-Nutzer zu imitieren, und wir haben ihre Leistung bewertet.\n\n"
            "Hier ist die ursprüngliche Persona, die verwendet wurde:\n"
            "--- URSPRÜNGLICHE PERSONA ---\n"
            "{persona_description}\n"
            "--- ENDE DER PERSONA ---\n\n"
            "Die KI wurde gebeten, auf einen Tweet zu antworten. Hier sind die Details:\n"
            "- **Originaltext des Nutzers (Ground Truth):** \"{ground_truth}\"\n"
            "- **Generierte Imitation der KI:** \"{imitation}\"\n\n"
            "Hier sind die quantitativen Evaluationsergebnisse der Imitation:\n"
            "--- EVALUATIONSERGEBNISSE ---\n"
            "{evaluation_results}\n"
            "--- ENDE DER ERGEBNISSE ---\n\n"
            "Deine Aufgabe: Analysiere die Persona, die Imitation und die Scores. "
            "Schreibe eine überarbeitete, verbesserte Version der Persona. "
            "Die neue Persona sollte die Schwächen beheben, die durch die niedrigen Scores oder den Vergleich "
            "von Imitation und Originaltext offensichtlich werden. Gib NUR die neue, verbesserte Persona-Beschreibung aus."
            "\n\nVERBESSERTE PERSONA:"
        )
        return template

    def improve_persona(self,
                        persona_description: str,
                        evaluation_results: Dict[str, float],
                        ground_truth_example: str,
                        imitation_example: str) -> str:
        """
        Führt die komplette Pipeline zur Verbesserung einer Persona aus.

        Args:
            persona_description: Die ursprüngliche Persona-Beschreibung.
            evaluation_results: Die Ergebnisse aus der EvaluationPipeline.
            ground_truth_example: Der Originaltext als Beispiel für die Analyse.
            imitation_example: Der generierte Text als Beispiel für die Analyse.

        Returns:
            Ein String, der die neue, verbesserte Persona-Beschreibung enthält.
        """
        print("\n[Pipeline Start] Verbessere Persona basierend auf Ergebnissen...")

        # 1. Evaluationsergebnisse für den Prompt formatieren
        # Wir wandeln das Dictionary in einen lesbaren String um.
        formatted_results = "\n".join([f"- {key}: {value:.4f}" for key, value in evaluation_results.items()])

        # 2. Prompt erstellen
        print("Schritt 1: Erstelle den finalen Prompt für die Verbesserung...")
        final_prompt = self.improvement_prompt_template.format(
            persona_description=persona_description,
            evaluation_results=formatted_results,
            ground_truth=ground_truth_example,
            imitation=imitation_example
        )

        # 3. LLM aufrufen, um die neue Persona zu generieren
        print("Schritt 2: Sende Anfrage an das LLM, um Persona zu verbessern...")
        improved_persona = self.llm_handler.generate_response(final_prompt)
        
        print("[Pipeline Ende] Persona erfolgreich verbessert.")
        return improved_persona

# --- Beispiel für die Verwendung des gesamten iterativen Kreislaufs ---
if __name__ == "__main__":
    # Importiere alle vorherigen Pipelines
    from db_loader import DbLoader
    from persona_creation import PersonaCreationPipeline
    from imitation_pipeline import ImitationPipeline
    from evaluation_pipeline import EvaluationPipeline, BleuMetric, RougeMetric, LlmJudgeMetric

    try:
        # --- SETUP ---
        print("--- SETUP: Initialisiere alle Pipelines ---")
        loader = DbLoader()
        llm_handler = LlmHandler()
        persona_pipeline = PersonaCreationPipeline()
        imitation_pipeline = ImitationPipeline()
        improvement_pipeline = PersonaImprovementPipeline()
        
        eval_metrics = [BleuMetric(), RougeMetric(), LlmJudgeMetric(llm_handler)]
        eval_pipeline = EvaluationPipeline(metrics=eval_metrics)
        
        sample_user_id = loader.get_all_user_ids()[0]
        
        # --- RUNDE 1: Initiale Erstellung und Evaluation ---
        print(f"\n--- RUNDE 1: Workflow für Benutzer {sample_user_id} ---")
        
        # 1. Persona erstellen
        initial_persona = persona_pipeline.create_persona_for_user(sample_user_id)

        # 2. Imitation erzeugen
        tweets = loader.get_tweets_by_user(sample_user_id)
        if len(tweets) < 2: raise ValueError("Nicht genug Tweets für den Workflow.")
        stimulus, ground_truth = tweets[0], tweets[1]
        
        imitation_v1 = imitation_pipeline.generate_imitation(initial_persona, stimulus)

        # 3. Evaluieren
        results_v1 = eval_pipeline.evaluate(ground_truth['full_text'], imitation_v1)
        
        print("\n--- ERGEBNISSE RUNDE 1 ---")
        print("Persona:", initial_persona)
        print("Imitation:", imitation_v1)
        print("Scores:", results_v1)
        print("--------------------------")

        # --- RUNDE 2: Verbesserung und erneute Evaluation ---
        print(f"\n--- RUNDE 2: Verbesserung der Persona ---")
        
        # 4. Persona verbessern
        improved_persona = improvement_pipeline.improve_persona(
            persona_description=initial_persona,
            evaluation_results=results_v1,
            ground_truth_example=ground_truth['full_text'],
            imitation_example=imitation_v1
        )
        
        # 5. Erneute Imitation mit verbesserter Persona
        imitation_v2 = imitation_pipeline.generate_imitation(improved_persona, stimulus)
        
        # 6. Erneute Evaluation
        results_v2 = eval_pipeline.evaluate(ground_truth['full_text'], imitation_v2)
        
        print("\n--- ERGEBNISSE RUNDE 2 ---")
        print("Verbesserte Persona:", improved_persona)
        print("Neue Imitation:", imitation_v2)
        print("Neue Scores:", results_v2)
        print("--------------------------")

    except Exception as e:
        print(f"\nEin Fehler im Hauptskript ist aufgetreten: {e}")