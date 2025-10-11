# evaluation_pipeline.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any

# Importe für spezifische Metriken
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from llm_handler import LlmHandler


# 1. Basisklasse für alle Metriken (unsere "Schnittstelle")
# ------------------------------------------------------------------------------
class BaseMetric(ABC):
    """
    Abstrakte Basisklasse für eine Evaluationsmetrik.
    Jede neue Metrik muss von dieser Klasse erben und die `calculate` Methode implementieren.
    """
    @property
    @abstractmethod
    def name(self) -> str:
        """Der Name der Metrik, z.B. 'bleu' oder 'rouge_l'."""
        pass

    @abstractmethod
    def calculate(self, ground_truth: str, generated_text: str) -> float:
        """
        Berechnet den Score zwischen dem Originaltext und dem generierten Text.

        Args:
            ground_truth: Der originale, erwartete Text.
            generated_text: Der von der KI generierte Text.

        Returns:
            Ein numerischer Score.
        """
        pass


# 2. Implementierungen der spezifischen Metriken
# ------------------------------------------------------------------------------
class BleuMetric(BaseMetric):
    """Implementierung für den BLEU-Score."""
    @property
    def name(self) -> str:
        return "bleu_score"

    def calculate(self, ground_truth: str, generated_text: str) -> float:
        reference = [ground_truth.split()]
        candidate = generated_text.split()
        # SmoothingFunction hilft, 0.0 Scores bei kurzen Sätzen zu vermeiden
        smoothie = SmoothingFunction().method4
        score = sentence_bleu(reference, candidate, smoothing_function=smoothie)
        return score

class RougeMetric(BaseMetric):
    """Implementierung für ROUGE-L (Longest Common Subsequence)."""
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    @property
    def name(self) -> str:
        return "rouge_l"

    def calculate(self, ground_truth: str, generated_text: str) -> float:
        scores = self.scorer.score(ground_truth, generated_text)
        return scores['rougeL'].fmeasure

class LlmJudgeMetric(BaseMetric):
    """
    Eine Metrik, die ein anderes LLM als 'Richter' verwendet, um die Qualität zu bewerten.
    """
    def __init__(self, llm_handler: LlmHandler):
        self.llm_handler = llm_handler
        self.judge_prompt_template = (
            "Bewerte die folgende generierte Antwort im Vergleich zum Originaltext auf einer Skala von 1 bis 5, "
            "wobei 5 'sehr authentisch' und 1 'überhaupt nicht authentisch' bedeutet. "
            "Berücksichtige Tonalität, Stil und Inhalt. Gib NUR die Zahl als Antwort.\n\n"
            "ORIGINALTEXT: \"{ground_truth}\"\n"
            "GENERIERTE ANTWORT: \"{generated_text}\"\n\n"
            "BEWERTUNG (1-5):"
        )

    @property
    def name(self) -> str:
        return "llm_judge_score"

    def calculate(self, ground_truth: str, generated_text: str) -> float:
        prompt = self.judge_prompt_template.format(
            ground_truth=ground_truth,
            generated_text=generated_text
        )
        response = self.llm_handler.generate_response(prompt)
        try:
            # Versuche, die Zahl aus der Antwort zu extrahieren
            return float(response.strip())
        except ValueError:
            print(f"Warnung: LLM Judge gab keine gültige Zahl zurück: '{response}'")
            return 0.0


# 3. Die eigentliche Evaluations-Pipeline
# ------------------------------------------------------------------------------
class EvaluationPipeline:
    """
    Orchestriert die Evaluation einer generierten Imitation gegen den Ground Truth
    mithilfe einer Liste von konfigurierbaren Metriken.
    """
    def __init__(self, metrics: List[BaseMetric]):
        """
        Initialisiert die Pipeline mit den Metriken, die verwendet werden sollen.

        Args:
            metrics: Eine Liste von Metrik-Objekten (z.B. [BleuMetric(), RougeMetric()]).
        """
        if not metrics:
            raise ValueError("Die Metrik-Liste darf nicht leer sein.")
        self.metrics = metrics
        print(f"EvaluationPipeline initialisiert mit Metriken: {[m.name for m in metrics]}")

    def evaluate(self, ground_truth: str, generated_text: str) -> Dict[str, float]:
        """
        Führt alle konfigurierten Metriken aus.

        Args:
            ground_truth: Der originale Tweet-Text.
            generated_text: Der imitierte Text.

        Returns:
            Ein Dictionary mit den Ergebnissen, z.B. {'bleu_score': 0.85, 'rouge_l': 0.92}.
        """
        results = {}
        for metric in self.metrics:
            score = metric.calculate(ground_truth, generated_text)
            results[metric.name] = score
            print(f" - {metric.name}: {score:.4f}")
        return results


# --- Beispiel für die Verwendung des gesamten Workflows ---
if __name__ == "__main__":
    # Importiere die vorherigen Pipelines
    from imitation_pipeline import ImitationPipeline
    from persona_creation import PersonaCreationPipeline
    from db_loader import DbLoader

    try:
        # 1. Initialisiere alle Komponenten
        llm_handler = LlmHandler()
        loader = DbLoader()
        persona_pipeline = PersonaCreationPipeline()
        imitation_pipeline = ImitationPipeline()
        
        # 2. Konfiguriere die Evaluations-Pipeline mit den gewünschten Metriken
        evaluation_metrics = [
            BleuMetric(),
            RougeMetric(),
            LlmJudgeMetric(llm_handler) 
        ]
        eval_pipeline = EvaluationPipeline(metrics=evaluation_metrics)

        # 3. Führe den bekannten Workflow aus, um eine Imitation zu erzeugen
        sample_user_id = loader.get_all_user_ids()[0]
        persona = persona_pipeline.create_persona_for_user(sample_user_id)
        
        # Nehmen wir einen History-Tweet als Ground Truth für die Evaluation
        history_tweets = loader.get_tweets_by_user(sample_user_id)
        if not history_tweets or len(history_tweets) < 2:
            raise ValueError("Nicht genügend Tweets für die Evaluation vorhanden.")
            
        stimulus_tweet = history_tweets[0]
        ground_truth_tweet = history_tweets[1] # Ein anderer Tweet als Referenz

        imitation = imitation_pipeline.generate_imitation(persona, stimulus_tweet)
        
        print("\n--- STARTE EVALUATION ---")
        print(f"Ground Truth: '{ground_truth_tweet['full_text']}'")
        print(f"Imitation:    '{imitation}'")
        
        # 4. Führe die Evaluation durch
        evaluation_results = eval_pipeline.evaluate(
            ground_truth=ground_truth_tweet['full_text'],
            generated_text=imitation
        )
        
        print("\n--- EVALUATIONSERGEBNISSE ---")
        print(evaluation_results)
        print("-----------------------------")

    except Exception as e:
        print(f"\nEin Fehler im Hauptskript ist aufgetreten: {e}")