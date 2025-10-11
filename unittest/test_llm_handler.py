# unittest/test_llm_handler.py
from unittest.mock import patch, MagicMock
import os
import sys
import pytest
from dotenv import load_dotenv

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from llm_handler import LlmHandler

# Neuer Mock für das google.genai.Client Objekt
@patch('llm_handler.Client')
@patch.dict(os.environ, {"GOOGLE_API_KEY": "fake_key_for_test"}, clear=True)
def test_llm_handler_logic_with_mock(MockClient):
    """
    Testet die Logik des LlmHandler mit neuem GenAI-Client, ohne echte API-Anfrage.
    """
    # Mock der Antwort
    mock_response = MagicMock()
    mock_response.text = "Mocked LLM response"

    # Mock der Client-Instanz
    mock_client_instance = MockClient.return_value
    mock_client_instance.models.generate_content.return_value = mock_response

    # Initialisierung des Handlers
    handler = LlmHandler()
    response = handler.generate_response("Ein Test-Prompt")

    # Überprüfe, dass generate_content korrekt aufgerufen wurde
    mock_client_instance.models.generate_content.assert_called_once()
    args, kwargs = mock_client_instance.models.generate_content.call_args
    assert kwargs["contents"] == "Ein Test-Prompt"

    # Überprüfe Rückgabe
    assert response == "Mocked LLM response"

@pytest.mark.real_api
def test_llm_handler_real_api_call():
    """
    Testet die tatsächliche Verbindung zur Google Gemini API.
    """
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY nicht gefunden. Überspringe echten API-Test.")
    try:
        handler = LlmHandler()
        test_prompt = "Gib nur das Wort 'test' zurück."
        response = handler.generate_response(test_prompt)
        assert isinstance(response, str)
        assert len(response) > 0
        print(f"\nAntwort vom echten API-Call: '{response}'")
    except Exception as e:
        pytest.fail(f"Der echte API-Aufruf ist fehlgeschlagen: {e}")