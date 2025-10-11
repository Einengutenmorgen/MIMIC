# unittest/test_prompt_formatter.py

import sys
import os

# FÜGEN SIE DIESEN BLOCK HINZU, UM DEN PROJEKTPFAD ZU KORRIGIERIEN
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import pytest
from prompt_formatter import PromptFormatter

@pytest.fixture
def formatter():
    """Stellt eine Instanz des PromptFormatter für jeden Test bereit."""
    return PromptFormatter()

def test_format_tweets_for_persona_creation(formatter):
    """Testet, ob eine Liste von Tweets korrekt zu einem String zusammengefügt wird."""
    sample_tweets = [
        {'full_text': 'Dies ist der erste Tweet.'},
        {'full_text': 'Ein zweiter Gedanke hier.'},
        {'full_text': 'Und ein dritter.'}
    ]
    expected_output = "Dies ist der erste Tweet.\nEin zweiter Gedanke hier.\nUnd ein dritter."
    assert formatter.format_tweets_for_persona_creation(sample_tweets) == expected_output

def test_format_tweets_with_empty_and_missing_text(formatter):
    """Testet, wie die Formatierung mit leeren oder fehlerhaften Daten umgeht."""
    sample_tweets = [
        {'full_text': 'Gültiger Tweet.'},
        {'full_text': None}, # Leerer Text
        {'text': 'Falscher Schlüssel'} # Fehlender 'full_text' Schlüssel
    ]
    expected_output = "Gültiger Tweet."
    assert formatter.format_tweets_for_persona_creation(sample_tweets) == expected_output

def test_format_empty_tweet_list(formatter):
    """Testet das Verhalten bei einer leeren Eingabeliste."""
    assert formatter.format_tweets_for_persona_creation([]) == ""

def test_format_stimulus_for_imitation(formatter):
    """Testet die Formatierung eines einzelnen Stimulus-Tweets."""
    stimulus_tweet = {'full_text': 'Was denkt ihr über das neue Gesetz?'}
    expected_output = 'Was denkt ihr über das neue Gesetz?'
    assert formatter.format_stimulus_for_imitation(stimulus_tweet) == expected_output

def test_format_empty_stimulus(formatter):
    """Testet das Verhalten bei einem leeren oder ungültigen Stimulus."""
    assert formatter.format_stimulus_for_imitation({}) == ""
    assert formatter.format_stimulus_for_imitation({'text': 'Falsch'}) == ""