# prepare_holdout_data.py

import psycopg2
from db_loader import DB_CONFIG

def mark_holdout_tweets():
    """
    Markiert für jeden Benutzer mit > 150 Tweets die 50 neuesten als Holdout-Set.
    """
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("Datenbankverbindung hergestellt.")
        
        with conn.cursor() as cursor:
            # Diese komplexe SQL-Anweisung erledigt die ganze Arbeit in einem Schritt:
            # 1. Es findet alle Tweets von Benutzern, die mehr als 150 Tweets haben.
            # 2. Es nummeriert die Tweets jedes Benutzers, geordnet nach Erstellungsdatum (die neuesten zuerst).
            # 3. Es wählt die IDs der Top 50 neuesten Tweets für jeden dieser Benutzer aus.
            # 4. Schließlich aktualisiert es die 'tweets'-Tabelle und setzt 'is_holdout' auf TRUE für alle gefundenen IDs.
            update_query = """
                WITH ranked_tweets AS (
                    SELECT
                        tweet_id,
                        user_id,
                        ROW_NUMBER() OVER(PARTITION BY user_id ORDER BY created_at DESC) as rn
                    FROM
                        tweets
                    WHERE user_id IN (
                        SELECT user_id FROM tweets GROUP BY user_id HAVING COUNT(*) > 150
                    )
                )
                UPDATE tweets
                SET is_holdout = TRUE
                WHERE tweet_id IN (
                    SELECT tweet_id FROM ranked_tweets WHERE rn <= 50
                );
            """
            
            print("Markiere die 50 neuesten Tweets pro qualifiziertem Benutzer als Holdout-Set. Dies kann einen Moment dauern...")
            cursor.execute(update_query)
            
            # cursor.rowcount gibt die Anzahl der aktualisierten Zeilen zurück
            updated_rows = cursor.rowcount
            conn.commit()
            print(f"Erfolg! {updated_rows} Tweets wurden als Holdout-Daten markiert.")

    except psycopg2.Error as e:
        print(f"Datenbankfehler: {e}")
    finally:
        if conn:
            conn.close()
            print("Datenbankverbindung geschlossen.")

if __name__ == "__main__":
    mark_holdout_tweets()