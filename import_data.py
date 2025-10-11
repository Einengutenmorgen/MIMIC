import pandas as pd
import psycopg2
from psycopg2 import sql
import numpy as np

# --- KONFIGURATION ---
DB_CONFIG = {
    "user": "christophhau",
    "password": "",
    "host": "localhost",
    "port": "5432",
    "dbname": "mimic_v2"
}

CSV_FILE_PATH = "raw/Kopie von FolloweeIDs2_tweets_df_AugustPull.csv"

def clean_value(value):
    """Konvertiert NaN/NaT in None und stellt sicher, dass Zahlen korrekt behandelt werden."""
    if pd.isna(value):
        return None
    # Konvertiert numpy int/float in Python-Typen
    if isinstance(value, (np.integer, np.floating)):
        return float(value) # Sicherste Konvertierung für NUMERIC ist über float
    return value

def import_csv_to_db():
    """Liest eine CSV-Datei und importiert sie in die Tabellen 'users' und 'tweets'."""
    try:
        print(f"Lese CSV-Datei von: {CSV_FILE_PATH}")
        # WICHTIG: dtype=str verhindert, dass Pandas die großen Zahlen falsch interpretiert.
        # low_memory=False wird für große Dateien mit gemischten Typen empfohlen.
        df = pd.read_csv(CSV_FILE_PATH, dtype=str, low_memory=False)
        print(f"{len(df)} Zeilen erfolgreich geladen.")

        # Notwendige Spalten für die Eindeutigkeit der User und Tweets
        required_cols = ['original_user_id', 'screen_name', 'tweet_id']
        if not all(col in df.columns for col in required_cols):
            print(f"Fehler: Der CSV-Datei fehlen erforderliche Spalten. Benötigt: {required_cols}")
            return
            
    except FileNotFoundError:
        print(f"Fehler: Datei unter '{CSV_FILE_PATH}' nicht gefunden.")
        return
    except Exception as e:
        print(f"Fehler beim Laden der CSV: {e}")
        return

    conn = None
    try:
        print("Verbinde zur Datenbank '{}'...".format(DB_CONFIG["dbname"]))
        conn = psycopg2.connect(**DB_CONFIG)
        
        with conn.cursor() as cursor:
            # 1. User-Daten einfügen
            users_df = df[['original_user_id', 'screen_name']].dropna(subset=['original_user_id']).drop_duplicates()
            users = users_df.to_numpy()
            
            print(f"Füge {len(users)} eindeutige Benutzer ein...")
            
            insert_user_query = "INSERT INTO users (user_id, screen_name) VALUES (%s, %s) ON CONFLICT (user_id) DO NOTHING;"
            
            from psycopg2.extras import execute_batch
            execute_batch(cursor, insert_user_query, users)
            
            print("Benutzer erfolgreich eingefügt.")

            # 2. Tweet-Daten einfügen
            print(f"Füge {len(df)} Tweets ein...")

            insert_tweet_query = """
                INSERT INTO tweets (
                    tweet_id, user_id, full_text, created_at, collected_at, 
                    reply_to_id, reply_to_user, retweeted_user_id, expanded_url
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (tweet_id) DO NOTHING;
            """
            
            tweet_data = []
            for _, row in df.iterrows():
                # Bereinige die Zeile vor dem Anhängen
                tweet_tuple = (
                    clean_value(row.get('tweet_id')),
                    clean_value(row.get('original_user_id')),
                    clean_value(row.get('full_text')),
                    clean_value(row.get('created_at')),
                    clean_value(row.get('collected_at')),
                    clean_value(row.get('reply_to_id')),
                    clean_value(row.get('reply_to_user')),
                    clean_value(row.get('retweeted_user_ID')),
                    clean_value(row.get('expandedURL'))
                )
                # Nur Tweets mit einer gültigen ID hinzufügen
                if tweet_tuple[0] is not None and tweet_tuple[1] is not None:
                    tweet_data.append(tweet_tuple)

            execute_batch(cursor, insert_tweet_query, tweet_data)
            
            print("Tweets erfolgreich eingefügt.")
            
            conn.commit()
            print("Alle Änderungen wurden committet.")

    except psycopg2.Error as e:
        print(f"Datenbankfehler: {e}")
        if conn: conn.rollback()
    except Exception as e:
        print(f"Unerwarteter Fehler: {e}")
    finally:
        if conn:
            conn.close()
            print("Datenbankverbindung geschlossen.")

if __name__ == "__main__":
    import_csv_to_db()