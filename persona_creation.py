# persona_creation.py

from db_loader import DbLoader
from prompt_formatter import PromptFormatter
# Wir importieren eine (noch zu erstellende) LLM-Handler-Klasse.
# Das ermöglicht uns, die Logik jetzt schon zu schreiben.
from llm_handler import LlmHandler 

class PersonaCreationPipeline:
    """
    Orchestriert den Prozess der Persona-Erstellung für einen einzelnen Benutzer.
    """
    def __init__(self):
        """
        Initialisiert die Pipeline mit den notwendigen Komponenten.
        """
        self.db_loader = DbLoader()
        self.prompt_formatter = PromptFormatter()
        self.llm_handler = LlmHandler()
        self.persona_prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        """
        Lädt die Prompt-Vorlage für die Persona-Erstellung.
        In einer realen Anwendung würde dies aus einer Datei oder einer 
        Konfiguration geladen. Hier halten wir es einfach.
        """
        # Dies ist ein simpler, aber effektiver Prompt.
        template = (
            "Basierend auf der folgenden Sammlung von Tweets, analysiere den Kommunikationsstil, "
            "die wiederkehrenden Themen, die Meinungen und die allgemeine Persönlichkeit des Autors. "
            "Fasse diese Analyse in einer detaillierten, zusammenhängenden Persona-Beschreibung zusammen, "
            "die als Anleitung für eine KI dienen kann, um diesen Benutzer zu imitieren.\n\n"
            "--- TWEETS ---\n"
            "{user_tweets}\n"
            "--- ENDE DER TWEETS ---\n\n"
            "PERSONA-BESCHREIBUNG:"
        )
        return template

    def create_persona_for_user(self, user_id: int) -> str:
        """
        Führt die komplette Pipeline zur Persona-Erstellung für einen Benutzer aus.

        Args:
            user_id: Die ID des Benutzers, für den die Persona erstellt werden soll.

        Returns:
            Ein String, der die vom LLM generierte Persona-Beschreibung enthält.
        """
        print(f"\n[Pipeline Start] Erstelle Persona für Benutzer {user_id}...")

        # 1. Daten laden
        print("Schritt 1: Lade History-Tweets aus der Datenbank...")
        history_tweets = self.db_loader.get_tweets_by_user(user_id, is_holdout=False)
        if not history_tweets:
            print(f"Warnung: Keine History-Tweets für Benutzer {user_id} gefunden. Breche ab.")
            return ""

        # 2. Daten formatieren
        print("Schritt 2: Formatiere Tweets für den Prompt...")
        formatted_tweets = self.prompt_formatter.format_tweets_for_persona_creation(history_tweets)

        # 3. Prompt erstellen
        print("Schritt 3: Erstelle den finalen Prompt aus der Vorlage...")
        final_prompt = self.persona_prompt_template.format(user_tweets=formatted_tweets)

        # 4. LLM aufrufen, um die Persona zu generieren
        print("Schritt 4: Sende Anfrage an das LLM...")
        # Der eigentliche Aufruf wird in der LlmHandler-Klasse gekapselt sein.
        generated_persona = self.llm_handler.generate_response(final_prompt)
        
        print("[Pipeline Ende] Persona erfolgreich erstellt.")
        return generated_persona

# --- Beispiel für die Verwendung ---
if __name__ == "__main__":
    try:
        pipeline = PersonaCreationPipeline()

        # Wir holen uns den ersten verfügbaren Benutzer für das Beispiel
        all_users = pipeline.db_loader.get_all_user_ids()
        if all_users:
            sample_user_id = all_users[0]
            
            # Führe die Pipeline aus
            persona = pipeline.create_persona_for_user(sample_user_id)

            print("\n--- GENERIERTE PERSONA ---")
            print(persona)
            print("--------------------------")

    except Exception as e:
        print(f"\nEin Fehler im Hauptskript ist aufgetreten: {e}")