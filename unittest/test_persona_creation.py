# unittest/test_persona_creation.py

import sys
import os

# FÜGEN SIE DIESEN BLOCK HINZU, UM DEN PROJEKTPFAD ZU KORRIGIERIEN
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import pytest
from unittest.mock import MagicMock, patch

# Wir müssen die Klassen importieren, bevor wir sie patchen können
from persona_creation import PersonaCreationPipeline

# Mock-Objekte für die Abhängigkeiten der Pipeline
@pytest.fixture
def mock_db_loader():
    mock = MagicMock()
    mock.get_tweets_by_user.return_value = [{'full_text': 'Ein Tweet.'}]
    return mock

@pytest.fixture
def mock_formatter():
    mock = MagicMock()
    mock.format_tweets_for_persona_creation.return_value = "Formatierter Tweet-Text"
    return mock

@pytest.fixture
def mock_llm_handler():
    mock = MagicMock()
    mock.generate_response.return_value = "Generierte Persona"
    return mock

# Hier patchen wir die Klassen im 'persona_creation'-Modul, sodass jede neue Instanz
# der PersonaCreationPipeline unsere Mock-Objekte anstelle der echten verwendet.
@patch('persona_creation.DbLoader')
@patch('persona_creation.PromptFormatter')
@patch('persona_creation.LlmHandler')
def test_pipeline_full_run(MockLlmHandler, MockFormatter, MockDbLoader,
                           mock_db_loader, mock_formatter, mock_llm_handler):
    """
    Testet den gesamten Durchlauf der create_persona_for_user Methode.
    """
    # Weisen Sie die Instanzen unserer Fixtures den gemockten Klassen zu
    MockDbLoader.return_value = mock_db_loader
    MockFormatter.return_value = mock_formatter
    MockLlmHandler.return_value = mock_llm_handler
    
    pipeline = PersonaCreationPipeline()
    user_id = 12345
    result = pipeline.create_persona_for_user(user_id)

    # Überprüfen, ob jede Komponente mit den richtigen Argumenten aufgerufen wurde
    mock_db_loader.get_tweets_by_user.assert_called_once_with(user_id, is_holdout=False)
    
    mock_formatter.format_tweets_for_persona_creation.assert_called_once_with(
        [{'full_text': 'Ein Tweet.'}]
    )
    
    # Überprüfen, ob der Prompt korrekt zusammengesetzt und an den LLM-Handler gesendet wurde
    expected_prompt_fragment = "Formatierter Tweet-Text"
    # call_args[0][0] extrahiert das erste Argument des ersten Aufrufs
    actual_prompt = mock_llm_handler.generate_response.call_args[0][0]
    assert expected_prompt_fragment in actual_prompt
    
    # Überprüfen, ob das Endergebnis von der LLM stammt
    assert result == "Generierte Persona"

@patch('persona_creation.DbLoader')
@patch('persona_creation.PromptFormatter')
@patch('persona_creation.LlmHandler')
def test_pipeline_no_tweets_found(MockLlmHandler, MockFormatter, MockDbLoader):
    """
    Testet das Verhalten der Pipeline, wenn der DbLoader keine Tweets findet.
    """
    mock_db_loader_instance = MockDbLoader.return_value
    mock_db_loader_instance.get_tweets_by_user.return_value = [] # Leere Liste simulieren

    pipeline = PersonaCreationPipeline()
    result = pipeline.create_persona_for_user(999)

    # Sicherstellen, dass die Pipeline mit einem leeren String abbricht
    assert result == ""
    # Sicherstellen, dass der Formatter und LLM-Handler gar nicht erst aufgerufen wurden
    MockFormatter.return_value.format_tweets_for_persona_creation.assert_not_called()
    MockLlmHandler.return_value.generate_response.assert_not_called()