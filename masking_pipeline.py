# masking_pipeline.py

import re
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from typing import Dict, List, Tuple, Any, Optional

# Importiere unsere Datenbank-Helfer
from db_loader import DbLoader
from db_saver import DbSaver
from psycopg2.extras import RealDictCursor

# 1. Ihr OpinionWordIdentifier (unverändert)
# ------------------------------------------------------------------------------
class OpinionWordIdentifier:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.phrases = {
            'burn in hell': -0.9, 'piece of shit': -0.8, 'god bless': 0.7,
        }
    
    def identify_opinions(self, text: str) -> Dict[str, float]:
        results = {}
        text_lower = text.lower()
        for phrase, score in self.phrases.items():
            if phrase in text_lower:
                results[phrase] = score
                text = text.replace(phrase, '')
        words = re.findall(r'\b\w+\b|[^\w\s]', text)
        for word in words:
            word_clean = word.strip().lower()
            if not word_clean: continue
            vader_score = self.vader.lexicon.get(word_clean, None)
            if vader_score is not None:
                results[word] = vader_score
            else:
                try:
                    tb_score = TextBlob(word_clean).sentiment.polarity
                    if abs(tb_score) > 0.1: results[word] = tb_score
                    else: results[word] = 0.0
                except: results[word] = 0.0
        return results

    def mask_all_opinionated_words(self, text: str) -> Tuple[str, List[str]]:
        opinions = self.identify_opinions(text)
        opinionated_words = [word for word, score in opinions.items() if abs(score) > 0.1]
        if not opinionated_words:
            return text, []
        pattern = re.compile(r'\b(' + '|'.join(re.escape(word) for word in opinionated_words) + r')\b', re.IGNORECASE)
        masked_text = pattern.sub('[MASK]', text)
        return masked_text, opinionated_words

# 2. Die finale Masking-Pipeline mit "Get or Create"-Logik
# ------------------------------------------------------------------------------
class MaskingPipeline:
    def __init__(self):
        self.identifier = OpinionWordIdentifier()
        # Die Pipeline benötigt jetzt einen Saver, um neue Einträge zu speichern
        self.saver = DbSaver()
        # Wir benötigen auch eine eigene Verbindung, um Einträge zu suchen
        self.conn = psycopg2.connect(**DB_CONFIG)

    def _find_masked_tweet_in_db(self, original_tweet_id: int) -> Optional[Dict[str, Any]]:
        """Sucht nach einem bereits existierenden maskierten Tweet in der DB."""
        query = "SELECT original_tweet_id, masked_text, original_words FROM masked_tweets WHERE original_tweet_id = %s;"
        with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, (original_tweet_id,))
            return cursor.fetchone()

    def get_or_create_masked_tweet(self, original_tweet: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Holt einen maskierten Tweet aus der DB oder erstellt, speichert und 
        gibt ihn zurück, falls er nicht existiert.
        """
        tweet_id = original_tweet['tweet_id']
        
        # 1. Suche in der DB
        existing_masked_tweet = self._find_masked_tweet_in_db(tweet_id)
        if existing_masked_tweet:
            print(f"  -> Maskierter Tweet für ID {tweet_id} in DB gefunden.")
            return existing_masked_tweet

        # 2. Wenn nicht gefunden, erstellen
        print(f"  -> Maskiere Tweet ID {tweet_id} zum ersten Mal.")
        masked_text, original_words = self.identifier.mask_all_opinionated_words(original_tweet['full_text'])
        
        if not original_words:
            return None # Kein meinungsstarkes Wort gefunden

        # 3. Speichere das neue Ergebnis in der DB
        new_masked_tweet = self.saver.save_masked_tweet(
            original_tweet_id=tweet_id,
            masked_text=masked_text,
            original_words=original_words
        )
        print(f"  -> Neuer maskierter Tweet für ID {tweet_id} in DB gespeichert.")
        return new_masked_tweet

    def process_batch(self, tweets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Verarbeitet eine ganze Liste von Tweets für die Post-Completion-Aufgabe.
        """
        print(f"Starte 'Get or Create'-Maskierungsprozess für {len(tweets)} Tweets...")
        prepared_data = []
        for tweet in tweets:
            masked_data = self.get_or_create_masked_tweet(tweet)
            if masked_data:
                prepared_data.append(masked_data)
        
        print(f"{len(prepared_data)} Tweets wurden erfolgreich für die Post-Completion-Aufgabe vorbereitet.")
        return prepared_data

    def __del__(self):
        if self.conn:
            self.conn.close()

# --- Beispiel für die Verwendung ---
if __name__ == '__main__':
    pipeline = MaskingPipeline()
    loader = DbLoader()

    # Lade einige Beispiel-Tweets
    sample_user_id = loader.get_all_user_ids()[0]
    tweets_to_process = loader.get_tweets_by_user(sample_user_id, is_holdout=True)[:5]
    
    if tweets_to_process:
        print("\n--- ERSTER DURCHLAUF (Tweets werden erstellt und gespeichert) ---")
        processed_tweets_run1 = pipeline.process_batch(tweets_to_process)
        
        print("\n--- ZWEITER DURCHLAUF (Tweets werden aus der DB geladen) ---")
        processed_tweets_run2 = pipeline.process_batch(tweets_to_process)
        
        print("\n--- ERGEBNISSE ---")
        for item in processed_tweets_run2:
            print(f"Original ID: {item['original_tweet_id']}")
            print(f"Maskierter Text: {item['masked_text']}")
            print(f"Originale Wörter (Ground Truth): {item['original_words']}\n")