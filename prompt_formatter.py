# prompt_formatter.py

from typing import List, Dict, Any

class PromptFormatter:
    """
    Eine Klasse zum Formatieren von Rohdaten (geladen vom DbLoader)
    in simple Strings, die für LLM-Prompts geeignet sind.
    Diese Klasse enthält keine LLM-Logik, nur String-Formatierung.
    """

    def format_tweets_for_persona_creation(self, history_tweets: List[Dict[str, Any]]) -> str:
        """
        Formatiert eine Liste von History-Tweets in einen einzigen,
        durch Zeilenumbruch getrennten String.

        Args:
            history_tweets: Eine Liste von Tweet-Dictionaries vom DbLoader.

        Returns:
            Ein einzelner String, der alle Tweet-Texte enthält.
        """
        if not history_tweets:
            return ""
        
        # Extrahiere nur den Text aus jedem Tweet-Dictionary
        tweet_texts = [tweet['full_text'] for tweet in history_tweets if 'full_text' in tweet and tweet['full_text']]
        
        # Verbinde die Texte mit einem Zeilenumbruch
        return "\n".join(tweet_texts)

    def format_stimulus_for_imitation(self, stimulus_tweet: Dict[str, Any]) -> str:
        """
        Formatiert einen einzelnen Stimulus-Tweet für eine Imitations-Aufgabe.

        Args:
            stimulus_tweet: Ein einzelnes Tweet-Dictionary.

        Returns:
            Der Text des Tweets.
        """
        if not stimulus_tweet or 'full_text' not in stimulus_tweet:
            return ""
        
        return stimulus_tweet['full_text']

# --- Beispiel für die Verwendung in Kombination mit dem DbLoader ---
if __name__ == "__main__":
    # Dieses Beispiel benötigt die DbLoader-Klasse aus der anderen Datei
    from db_loader import DbLoader

    try:
        print("Initialisiere DbLoader...")
        loader = DbLoader()
        
        print("\nInitialisiere PromptFormatter...")
        formatter = PromptFormatter()

        # Einen Beispiel-Benutzer laden
        all_users = loader.get_all_user_ids()
        if all_users:
            sample_user_id = all_users[0]
            print(f"\nLade History-Tweets für Benutzer {sample_user_id}...")
            history_tweets = loader.get_tweets_by_user(sample_user_id)

            # Tweets für die Persona-Erstellung formatieren
            formatted_history = formatter.format_tweets_for_persona_creation(history_tweets)

            print("\n--- Formatierter String für Persona-Erstellung (erste 500 Zeichen) ---")
            print(formatted_history[:500])
            print("...\n")

            # Einen Tweet als Stimulus für die Imitation formatieren
            if history_tweets:
                stimulus = history_tweets[0]
                formatted_stimulus = formatter.format_stimulus_for_imitation(stimulus)
                print("--- Formatierter String für Imitations-Aufgabe ---")
                print(f"Original-Tweet-Daten: {stimulus}")
                print(f"Formatierter Stimulus:   '{formatted_stimulus}'")

    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")