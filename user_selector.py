# user_selector.py (vollständiger, final korrigierter Code)

import random
from db_loader import DbLoader
from typing import List, Dict, Any, Optional
from psycopg2.extras import RealDictCursor # <-- WICHTIGER IMPORT

class UserSelector:
    def __init__(self):
        self.loader = DbLoader()
        self.conn = self.loader.conn

    def find_users_with_min_tweets(self, min_history_tweets: int, min_holdout_tweets: int) -> List[Dict[str, Any]]:
        if not self.conn:
            raise ConnectionError("Keine Datenbankverbindung.")
        query = """
            SELECT
                user_id,
                COUNT(*) FILTER (WHERE is_holdout = FALSE) AS history_tweet_count,
                COUNT(*) FILTER (WHERE is_holdout = TRUE) AS holdout_tweet_count
            FROM tweets
            GROUP BY user_id
            HAVING
                COUNT(*) FILTER (WHERE is_holdout = FALSE) >= %s AND
                COUNT(*) FILTER (WHERE is_holdout = TRUE) >= %s
            ORDER BY history_tweet_count DESC;
        """
        # HIER DIE KORREKTUR: Wir verwenden explizit RealDictCursor
        with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, (min_history_tweets, min_holdout_tweets))
            return cursor.fetchall()

    def get_random_qualified_user(self, min_history_tweets: int, min_holdout_tweets: int) -> Optional[int]:
        """
        Findet alle qualifizierten Benutzer und wählt zufällig einen davon aus.
        """
        eligible_users = self.find_users_with_min_tweets(min_history_tweets, min_holdout_tweets)
        if not eligible_users:
            return None
        random_user = random.choice(eligible_users)
        return random_user['user_id']

# --- Beispiel für die Verwendung ---
if __name__ == "__main__":
    try:
        selector = UserSelector()
        MIN_HISTORY = 10
        MIN_HOLDOUT = 5
        print(f"\nSuche nach einem zufälligen Benutzer mit mindestens {MIN_HISTORY} History- und {MIN_HOLDOUT} Holdout-Tweets...")
        random_user_id = selector.get_random_qualified_user(MIN_HISTORY, MIN_HOLDOUT)
        if random_user_id:
            print(f"\nErfolg! Zufällig ausgewählter Benutzer: {random_user_id}")
        else:
            print("\nKein Benutzer erfüllt die Kriterien.")
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")