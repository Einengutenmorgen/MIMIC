# unittest/test_imitation_pipeline.py

import sys
import os

# FÜGEN SIE DIESEN BLOCK HINZU, UM DEN PROJEKTPFAD ZU KORRIGIERIEN
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import pytest
from unittest.mock import MagicMock, patch

from imitation_pipeline import ImitationPipeline

@pytest.fixture
def mock_formatter():
    mock = MagicMock()
    # Wir passen den Mock an, um die 'style_imitation' zu testen
    mock.format_stimulus_for_imitation.return_value = "Formatierter Stimulus für Stil-Imitation"
    return mock

@pytest.fixture
def mock_llm_handler():
    mock = MagicMock()
    mock.generate_response.return_value = "Generierte Imitation"
    return mock

@patch('imitation_pipeline.PromptFormatter')
@patch('imitation_pipeline.LlmHandler')
def test_pipeline_full_run_for_style_imitation(MockLlmHandler, MockFormatter, mock_formatter, mock_llm_handler):
    """
    Testet den Durchlauf für die 'style_imitation'-Aufgabe (der ursprüngliche Test).
    """
    MockFormatter.return_value = mock_formatter
    MockLlmHandler.return_value = mock_llm_handler
    
    pipeline = ImitationPipeline()
    
    persona_desc = "Eine sehr direkte Persona."
    stimulus_tweet = {'full_text': 'Ein Tweet, der als Stil-Vorlage dient.'}
    
    # HIER DIE KORREKTUR: task_type hinzugefügt
    result = pipeline.generate_imitation(persona_desc, stimulus_tweet, 'style_imitation')

    mock_formatter.format_stimulus_for_imitation.assert_called_once_with(stimulus_tweet)
    
    actual_prompt = mock_llm_handler.generate_response.call_args[0][0]
    assert "BEISPIEL-TWEET" in actual_prompt # Überprüft, ob der richtige Prompt verwendet wird
    
    assert result == "Generierte Imitation"

@patch('imitation_pipeline.PromptFormatter')
@patch('imitation_pipeline.LlmHandler')
def test_pipeline_empty_stimulus_for_style_imitation(MockLlmHandler, MockFormatter):
    """
    Testet das Verhalten mit leerem Stimulus für die 'style_imitation'-Aufgabe.
    """
    mock_formatter_instance = MockFormatter.return_value
    mock_formatter_instance.format_stimulus_for_imitation.return_value = ""

    pipeline = ImitationPipeline()
    
    # HIER DIE KORREKTUR: task_type hinzugefügt
    result = pipeline.generate_imitation("Eine Persona", {'full_text': ''}, 'style_imitation')

    assert result == ""
    MockLlmHandler.return_value.generate_response.assert_not_called()