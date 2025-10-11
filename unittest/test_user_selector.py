# unittest/test_user_selector.py

import sys
import os

# FÜGEN SIE DIESEN BLOCK HINZU, UM DEN PROJEKTPFAD ZU KORRIGIERIEN
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import pytest
from unittest.mock import MagicMock, patch

# Die zu testende Klasse importieren
from user_selector import UserSelector

# Wir patchen den DbLoader im 'user_selector'-Modul
@patch('user_selector.DbLoader')
def test_find_users_with_min_tweets(MockDbLoader):
    """
    Testet die Logik der Benutzerauswahl, indem die Datenbankantwort simuliert wird.
    """
    # 1. Testdaten vorbereiten: Das ist, was wir vom DB-Cursor zurückerwarten.
    mock_db_return_value = [
        {'user_id': 101, 'history_tweet_count': 200, 'holdout_tweet_count': 100},
        {'user_id': 102, 'history_tweet_count': 150, 'holdout_tweet_count': 55},
    ]

    # 2. Mock-Objekte für die Datenbankverbindung und den Cursor erstellen
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = mock_db_return_value
    
    mock_conn = MagicMock()
    # Der Cursor-Manager (`with ... as cursor`) soll unseren Mock-Cursor zurückgeben
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    # 3. Die Mock-Instanz des DbLoaders so konfigurieren, dass sie unsere
    #    gemockte Verbindung zurückgibt.
    mock_loader_instance = MockDbLoader.return_value
    mock_loader_instance.conn = mock_conn
    
    # 4. Den UserSelector initialisieren (er wird den gemockten DbLoader verwenden)
    selector = UserSelector()
    
    # Kriterien für den Test definieren
    min_history = 100
    min_holdout = 50
    
    # 5. Die zu testende Methode aufrufen
    result = selector.find_users_with_min_tweets(min_history, min_holdout)

    # 6. Überprüfungen
    # a) Wurde die SQL-Abfrage mit den korrekten Parametern ausgeführt?
    mock_cursor.execute.assert_called_once()
    # Das erste Argument des Aufrufs ist die Query, das zweite das Tupel mit den Werten
    called_args = mock_cursor.execute.call_args[0]
    assert min_history in called_args[1]
    assert min_holdout in called_args[1]
    
    # b) Entspricht das Ergebnis der Methode dem, was wir von der DB simuliert haben?
    assert result == mock_db_return_value
    assert len(result) == 2
    assert result[0]['user_id'] == 101