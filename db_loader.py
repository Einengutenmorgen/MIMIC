# db_loader.py

import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any, Optional

DB_CONFIG = {
    "user": "christophhau",
    "password": "",
    "host": "localhost",
    "port": "5432",
    "dbname": "mimic_v2"
}

class DbLoader:
    def __init__(self):
        self.conn = None
        try:
            self.conn = psycopg2.connect(**DB_CONFIG)
            print("Datenbankverbindung erfolgreich hergestellt.")
        except psycopg2.Error as e:
            print(f"Fehler bei der Datenbankverbindung: {e}")
            raise

    def get_all_user_ids(self) -> List[int]:
        if not self.conn: raise ConnectionError("Keine Datenbankverbindung.")
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT user_id FROM users ORDER BY user_id;")
            return [item[0] for item in cursor.fetchall()]

    def get_tweets_by_user(self, user_id: int, is_holdout: bool = False) -> List[Dict[str, Any]]:
        if not self.conn: raise ConnectionError("Keine Datenbankverbindung.")
        query = """
            SELECT tweet_id, full_text, created_at, reply_to_id
            FROM tweets
            WHERE user_id = %s AND is_holdout = %s
            ORDER BY created_at;
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, (user_id, is_holdout))
            return cursor.fetchall()

    def get_tweet_by_id(self, tweet_id: int) -> Optional[Dict[str, Any]]:
        if not self.conn: raise ConnectionError("Keine Datenbankverbindung.")
        query = "SELECT tweet_id, full_text, created_at, reply_to_id FROM tweets WHERE tweet_id = %s;"
        with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, (tweet_id,))
            return cursor.fetchone()

    def get_reply_stimuli(self, user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        if not self.conn: raise ConnectionError("Keine Datenbankverbindung.")
        reply_tweets = self.get_tweets_by_user(user_id, is_holdout=True)
        reply_tweets = [t for t in reply_tweets if t.get('reply_to_id')]
        stimuli = []
        for reply in reply_tweets:
            context = self.get_tweet_by_id(reply['reply_to_id'])
            if context:
                stimuli.append({"stimulus_tweet": reply, "context_tweet": context})
            if len(stimuli) >= limit: break
        return stimuli
    
    def __del__(self):
        if self.conn:
            self.conn.close()
            print("Datenbankverbindung geschlossen.")