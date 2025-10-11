import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

import psycopg2
from psycopg2 import sql

# --- KONFIGURATION ---
DB_CONFIG = {
    "user": "christophhau",
    "password": "",
    "host": "localhost",
    "port": "5432",
    "dbname": "mimic_v2"
}

def create_database_and_tables():
    """
    Erstellt die PostgreSQL-Tabellen für MIMIC v.02 neu, mit NUMERIC für IDs.
    Bestehende Tabellen werden zuerst gelöscht.
    """
    conn = None
    try:
        print(f"Verbinde zur Datenbank '{DB_CONFIG['dbname']}'...")
        conn = psycopg2.connect(**DB_CONFIG)
        
        with conn.cursor() as cursor:
            print("Lösche alte Tabellenstruktur (falls vorhanden)...")
            # CASCADE sorgt dafür, dass auch alle abhängigen Objekte gelöscht werden
            cursor.execute("""
                DROP TABLE IF EXISTS evaluations CASCADE;
                DROP TABLE IF EXISTS imitations CASCADE;
                DROP TABLE IF EXISTS rounds CASCADE;
                DROP TABLE IF EXISTS experiment_participants CASCADE;
                DROP TABLE IF EXISTS experiments CASCADE;
                DROP TABLE IF EXISTS tweets CASCADE;
                DROP TABLE IF EXISTS users CASCADE;
            """)
            print("Alte Tabellen gelöscht.")

            print("\nErstelle neue Tabellenstruktur mit NUMERIC für IDs...")

            # WICHTIG: user_id ist jetzt NUMERIC
            cursor.execute("""
                CREATE TABLE users (
                    user_id NUMERIC PRIMARY KEY,
                    screen_name VARCHAR(255) NOT NULL,
                    metadata JSONB
                );
            """)
            print("- Tabelle 'users' erstellt.")

            # WICHTIG: Alle ID-Spalten sind jetzt NUMERIC
            cursor.execute("""
                CREATE TABLE tweets (
                    tweet_id NUMERIC PRIMARY KEY,
                    user_id NUMERIC NOT NULL REFERENCES users(user_id),
                    full_text TEXT,
                    created_at TIMESTAMPTZ,
                    collected_at TIMESTAMPTZ,
                    reply_to_id NUMERIC,
                    reply_to_user NUMERIC,
                    retweeted_user_id NUMERIC,
                    expanded_url TEXT,
                    is_holdout BOOLEAN DEFAULT FALSE
                );
            """)
            print("- Tabelle 'tweets' erstellt.")
            
            cursor.execute("""
                CREATE TABLE experiments (
                    experiment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    name VARCHAR(255),
                    strategy VARCHAR(100),
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            print("- Tabelle 'experiments' erstellt.")

            cursor.execute("""
                CREATE TABLE experiment_participants (
                    experiment_id UUID NOT NULL REFERENCES experiments(experiment_id),
                    user_id NUMERIC NOT NULL REFERENCES users(user_id),
                    PRIMARY KEY (experiment_id, user_id)
                );
            """)
            print("- Tabelle 'experiment_participants' erstellt.")

            cursor.execute("""
                CREATE TABLE rounds (
                    round_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    experiment_id UUID NOT NULL REFERENCES experiments(experiment_id),
                    user_id NUMERIC NOT NULL REFERENCES users(user_id),
                    round_number INTEGER NOT NULL,
                    persona_description TEXT,
                    UNIQUE (experiment_id, user_id, round_number)
                );
            """)
            print("- Tabelle 'rounds' erstellt.")

            cursor.execute("""
                CREATE TABLE imitations (
                    imitation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    round_id UUID NOT NULL REFERENCES rounds(round_id),
                    original_tweet_id NUMERIC NOT NULL REFERENCES tweets(tweet_id),
                    generated_text TEXT,
                    task_type VARCHAR(100)
                );
            """)
            print("- Tabelle 'imitations' erstellt.")

            cursor.execute("""
                CREATE TABLE evaluations (
                    evaluation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    imitation_id UUID NOT NULL REFERENCES imitations(imitation_id) ON DELETE CASCADE,
                    bleu_score REAL,
                    rouge_1 REAL,
                    rouge_2 REAL,
                    rouge_l REAL,
                    bert_score_f1 REAL,
                    perplexity REAL,
                    content_accuracy REAL,
                    format_adherence BOOLEAN
                );
            """)
            print("- Tabelle 'evaluations' erstellt.")

            conn.commit()
            print("\nAlle Tabellen wurden erfolgreich neu erstellt.")

    except psycopg2.Error as e:
        print(f"Datenbankfehler: {e}")
    finally:
        if conn:
            conn.close()
            print("Datenbankverbindung geschlossen.")

if __name__ == "__main__":
    create_database_and_tables()