# unittest/test_db_loader.py

import pytest
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from unittest.mock import patch, MagicMock
from db_loader import DbLoader, DB_CONFIG

@patch('db_loader.psycopg2.connect')
def test_connection_success(mock_connect):
    """Tests that the DbLoader initializes by calling connect with the correct config."""
    loader = DbLoader()
    mock_connect.assert_called_once_with(**DB_CONFIG)

@patch('db_loader.psycopg2.connect')
def test_get_tweets_by_user(mock_connect):
    """Tests that the correct SQL (including reply_to_id) is used for both calls."""
    mock_cursor = MagicMock()
    mock_connect.return_value.cursor.return_value.__enter__.return_value = mock_cursor
    
    loader = DbLoader()
    
    expected_sql = """
            SELECT tweet_id, full_text, created_at, reply_to_id
            FROM tweets
            WHERE user_id = %s AND is_holdout = %s
            ORDER BY created_at;
        """
        
    # Test the first call
    loader.get_tweets_by_user(user_id=123, is_holdout=False)
    mock_cursor.execute.assert_called_with(expected_sql, (123, False))
    
    # Test the second call
    loader.get_tweets_by_user(user_id=456, is_holdout=True)
    mock_cursor.execute.assert_called_with(expected_sql, (456, True))

# You can now delete the outdated tests from `unittest/test_task_pipelines.py` if you wish.
# The following tests ensure all DbLoader methods are correctly tested in this file.

@patch('db_loader.psycopg2.connect')
def test_get_tweet_by_id(mock_connect):
    """Tests fetching a single tweet, ensuring the SQL includes reply_to_id."""
    mock_cursor = MagicMock()
    mock_connect.return_value.cursor.return_value.__enter__.return_value = mock_cursor
    
    loader = DbLoader()
    loader.get_tweet_by_id(999)

    expected_sql = "SELECT tweet_id, full_text, created_at, reply_to_id FROM tweets WHERE tweet_id = %s;"
    mock_cursor.execute.assert_called_with(expected_sql, (999,))

@patch('db_loader.psycopg2.connect')
def test_get_all_user_ids(mock_connect):
    """Tests that get_all_user_ids correctly processes the cursor's return value."""
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [(123,), (456,)]
    mock_connect.return_value.cursor.return_value.__enter__.return_value = mock_cursor
    
    loader = DbLoader()
    user_ids = loader.get_all_user_ids()

    assert user_ids == [123, 456]
    mock_cursor.execute.assert_called_once_with("SELECT user_id FROM users ORDER BY user_id;")