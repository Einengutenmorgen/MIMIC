# update_database.py

import psycopg2
from db_loader import DB_CONFIG

def add_masked_tweets_table():
    """
    Fügt die 'masked_tweets' Tabelle zur Datenbank hinzu, falls sie nicht existiert.
    """
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("Datenbankverbindung hergestellt.")
        
        with conn.cursor() as cursor:
            # Erstellt die neue Tabelle mit einer Verknüpfung zur 'tweets' Tabelle.
            # `ON DELETE CASCADE` sorgt dafür, dass ein maskierter Tweet gelöscht wird,
            # wenn der Original-Tweet gelöscht wird.
            create_table_query = """
                CREATE TABLE IF NOT EXISTS masked_tweets (
                    masked_tweet_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    original_tweet_id NUMERIC NOT NULL REFERENCES tweets(tweet_id) ON DELETE CASCADE,
                    masked_text TEXT,
                    original_words JSONB, -- Wir verwenden JSONB, um eine Liste von Wörtern zu speichern
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(original_tweet_id) -- Stellt sicher, dass jeder Tweet nur einmal maskiert wird
                );
            """
            
            print("Erstelle 'masked_tweets' Tabelle (falls nicht vorhanden)...")
            cursor.execute(create_table_query)
            conn.commit()
            print("Tabelle 'masked_tweets' erfolgreich erstellt oder bereits vorhanden.")

    except psycopg2.Error as e:
        print(f"Datenbankfehler: {e}")
    finally:
        if conn:
            conn.close()
            print("Datenbankverbindung geschlossen.")

if __name__ == "__main__":
    add_masked_tweets_table()