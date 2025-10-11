# unittest/test_persona_improvement.py

import sys
import os

# FÜGEN SIE DIESEN BLOCK HINZU, UM DEN PROJEKTPFAD ZU KORRIGIERIEN
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import pytest
from unittest.mock import MagicMock, patch

# Die zu testende Klasse importieren
from persona_improvement import PersonaImprovementPipeline

@pytest.fixture
def mock_llm_handler():
    mock = MagicMock()
    mock.generate_response.return_value = "Dies ist die verbesserte Persona."
    return mock

@patch('persona_improvement.LlmHandler')
def test_improvement_pipeline_full_run(MockLlmHandler, mock_llm_handler):
    """
    Testet den gesamten Durchlauf der improve_persona Methode.
    """
    MockLlmHandler.return_value = mock_llm_handler
    
    pipeline = PersonaImprovementPipeline()
    
    # Testdaten
    initial_persona = "Eine einfache Persona."
    eval_results = {"bleu_score": 0.5, "rouge_l": 0.6}
    ground_truth = "Das ist der Originaltext."
    imitation = "Das ist die Imitation."
    
    result = pipeline.improve_persona(
        persona_description=initial_persona,
        evaluation_results=eval_results,
        ground_truth_example=ground_truth,
        imitation_example=imitation
    )

    # Überprüfen, ob der LLM-Handler aufgerufen wurde
    mock_llm_handler.generate_response.assert_called_once()
    
    # Überprüfen, ob alle relevanten Informationen im Prompt enthalten sind
    actual_prompt = mock_llm_handler.generate_response.call_args[0][0]
    assert initial_persona in actual_prompt
    assert "bleu_score: 0.5000" in actual_prompt
    assert "rouge_l: 0.6000" in actual_prompt
    assert ground_truth in actual_prompt
    assert imitation in actual_prompt
    
    # Überprüfen, ob das Endergebnis von der LLM stammt
    assert result == "Dies ist die verbesserte Persona."