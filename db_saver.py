# db_saver.py

import psycopg2
import json # Wichtig für die Umwandlung der Wortliste in einen JSON-String
from typing import Dict, Any, List

# Wir können die gleiche Konfiguration wie im DbLoader verwenden
from db_loader import DB_CONFIG

class DbSaver:
    """
    Eine Klasse zum Speichern von Experiment-Ergebnissen in der MIMIC v.02 Datenbank.
    """
    def __init__(self):
        """
        Initialisiert den Saver und stellt die Datenbankverbindung her.
        """
        self.conn = None
        try:
            self.conn = psycopg2.connect(**DB_CONFIG)
            print("DbSaver: Datenbankverbindung erfolgreich hergestellt.")
        except psycopg2.Error as e:
            print(f"DbSaver Fehler bei der Datenbankverbindung: {e}")
            raise

    # ... (die bestehenden Methoden save_experiment, save_round, etc. bleiben unverändert) ...
    def save_experiment(self, name: str, strategy: str) -> str:
        """
        Erstellt einen neuen Eintrag in der 'experiments'-Tabelle.

        Returns:
            Die UUID des neu erstellten Experiments.
        """
        query = "INSERT INTO experiments (name, strategy) VALUES (%s, %s) RETURNING experiment_id;"
        with self.conn.cursor() as cursor:
            cursor.execute(query, (name, strategy))
            experiment_id = cursor.fetchone()[0]
            self.conn.commit()
            return experiment_id

    def save_round(self, experiment_id: str, user_id: int, round_number: int, persona: str) -> str:
        """
        Speichert eine einzelne Runde eines Experiments.

        Returns:
            Die UUID der neu erstellten Runde.
        """
        query = """
            INSERT INTO rounds (experiment_id, user_id, round_number, persona_description)
            VALUES (%s, %s, %s, %s) RETURNING round_id;
        """
        with self.conn.cursor() as cursor:
            cursor.execute(query, (experiment_id, user_id, round_number, persona))
            round_id = cursor.fetchone()[0]
            self.conn.commit()
            return round_id

    def save_imitation_and_evaluation(self, round_id: str, original_tweet_id: int,
                                      generated_text: str, task_type: str,
                                      evaluation_scores: Dict[str, float]):
        """
        Speichert eine Imitation und die zugehörigen Evaluations-Scores in einer Transaktion.
        """
        imitation_query = """
            INSERT INTO imitations (round_id, original_tweet_id, generated_text, task_type)
            VALUES (%s, %s, %s, %s) RETURNING imitation_id;
        """
        # Dynamisch die Spaltennamen und Platzhalter für die Evaluation-Query erstellen
        score_columns = evaluation_scores.keys()
        score_placeholders = ', '.join(['%s'] * len(score_columns))
        
        evaluation_query = f"""
            INSERT INTO evaluations (imitation_id, {', '.join(score_columns)})
            VALUES (%s, {score_placeholders});
        """
        
        with self.conn.cursor() as cursor:
            try:
                # 1. Imitation speichern und die neue ID erhalten
                cursor.execute(imitation_query, (round_id, original_tweet_id, generated_text, task_type))
                imitation_id = cursor.fetchone()[0]
                
                # 2. Evaluation speichern
                score_values = [imitation_id] + list(evaluation_scores.values())
                cursor.execute(evaluation_query, score_values)
                
                self.conn.commit()
            except psycopg2.Error as e:
                print(f"DbSaver Fehler: Transaktion fehlgeschlagen. Rollback wird ausgeführt. {e}")
                self.conn.rollback()
                raise

    # --- NEUE METHODE ---
    def save_masked_tweet(self, original_tweet_id: int, masked_text: str, original_words: List[str]) -> Dict[str, Any]:
        """
        Speichert einen neu maskierten Tweet in der 'masked_tweets' Tabelle.
        
        Returns:
            Ein Dictionary des neu gespeicherten maskierten Tweets.
        """
        # Wir müssen die Liste der Wörter in einen JSON-String umwandeln,
        # damit sie in der JSONB-Spalte gespeichert werden kann.
        original_words_json = json.dumps(original_words)
        
        query = """
            INSERT INTO masked_tweets (original_tweet_id, masked_text, original_words)
            VALUES (%s, %s, %s)
            RETURNING original_tweet_id, masked_text, original_words;
        """
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(query, (original_tweet_id, masked_text, original_words_json))
            new_masked_tweet = cursor.fetchone()
            self.conn.commit()
            return new_masked_tweet
            
    def __del__(self):
        """
        Schließt die Datenbankverbindung.
        """
        if self.conn:
            self.conn.close()