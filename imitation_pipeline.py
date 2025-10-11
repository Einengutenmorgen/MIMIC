# imitation_pipeline.py

from db_loader import DbLoader
from prompt_formatter import PromptFormatter
from llm_handler import LlmHandler
from typing import Dict, Any

class ImitationPipeline:
    def __init__(self):
        self.formatter = PromptFormatter()
        self.llm_handler = LlmHandler()
        self.prompt_templates = self._load_prompt_templates()

    def _load_prompt_templates(self) -> Dict[str, str]:
        templates = {
            "style_imitation": ( # Umbenannt für mehr Klarheit
                "Du bist ein KI-Assistent, dessen Aufgabe es ist, einen Social-Media-Nutzer zu imitieren. "
                "Hier ist die Persona-Beschreibung des Nutzers, an die du dich halten musst:\n\n"
                "--- PERSONA ---\n"
                "{persona_description}\n"
                "--- ENDE DER PERSONA ---\n\n"
                "Deine Aufgabe ist es, einen authentischen Tweet im Stil des Nutzers zu verfassen. "
                "Hier ist der ursprüngliche Tweet als stilistisches Vorbild:\n\n"
                "--- BEISPIEL-TWEET ---\n"
                "{stimulus_text}\n"
                "--- ENDE DES TWEETS ---\n\n"
                "DEINE IMITATION:"
            ),
            "post_completion": (
                "Du bist ein KI-Assistent, dessen Aufgabe es ist, einen Social-Media-Nutzer zu imitieren. "
                "Hier ist die Persona-Beschreibung des Nutzers:\n\n"
                "--- PERSONA ---\n"
                "{persona_description}\n"
                "--- ENDE DER PERSONA ---\n\n"
                "Deine Aufgabe ist es, die [MASK]-Token im folgenden Text so zu ersetzen, "
                "dass ein authentischer Tweet entsteht, der zum Stil des Nutzers passt. "
                "Gib nur den vervollständigten Satz zurück.\n\n"
                "--- MASKIERTER TWEET ---\n"
                "{stimulus_text}\n"
                "--- ENDE DES TWEETS ---\n\n"
                "VERVOLLSTÄNDIGTER TWEET:"
            ),
            "contextual_reply": ( # <-- NEUER, BESSERER PROMPT
                "Du bist ein KI-Assistent, dessen Aufgabe es ist, einen Social-Media-Nutzer zu imitieren. "
                "Hier ist die Persona-Beschreibung des Nutzers, an die du dich halten musst:\n\n"
                "--- PERSONA ---\n"
                "{persona_description}\n"
                "--- ENDE DER PERSONA ---\n\n"
                "Deine Aufgabe ist es, eine authentische Antwort auf den folgenden Tweet zu verfassen, "
                "so wie es der beschriebene Nutzer tun würde. Antworte direkt, ohne zusätzliche Erklärungen.\n\n"
                "--- TWEET, AUF DEN DU ANTWORTEN SOLLST (KONTEXT) ---\n"
                "{context_text}\n"
                "--- ENDE DES TWEETS ---\n\n"
                "DEINE ANTWORT:"
            )
        }
        return templates

    def generate_imitation(self, persona_description: str, stimulus_data: Dict[str, Any], task_type: str) -> str:
        print(f"\n[Pipeline Start] Erzeuge Imitation für Aufgabe: '{task_type}'...")
        if task_type not in self.prompt_templates:
            raise ValueError(f"Unbekannter Aufgabentyp: {task_type}")

        template = self.prompt_templates[task_type]
        final_prompt = "" # Initialize final_prompt

        # Wähle die richtigen Daten für den Prompt basierend auf der Aufgabe
        if task_type == 'post_completion':
            stimulus_text = stimulus_data.get('masked_text', '')
            if stimulus_text:
                final_prompt = template.format(persona_description=persona_description, stimulus_text=stimulus_text)
        elif task_type == 'contextual_reply':
            context_text = stimulus_data.get('context_tweet', {}).get('full_text', '')
            if context_text:
                final_prompt = template.format(persona_description=persona_description, context_text=context_text)
        else: # style_imitation
            # HIER DIE KORREKTUR: Wir verwenden den Formatter wieder
            stimulus_text = self.formatter.format_stimulus_for_imitation(stimulus_data)
            if stimulus_text:
                final_prompt = template.format(persona_description=persona_description, stimulus_text=stimulus_text)

        if not final_prompt:
            print("Warnung: Leerer oder ungültiger Stimulus. Breche ab.")
            return ""

        generated_imitation = self.llm_handler.generate_response(final_prompt)
        print("[Pipeline Ende] Imitation erfolgreich erzeugt.")
        return generated_imitation