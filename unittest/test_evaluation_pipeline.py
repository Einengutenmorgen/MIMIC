# unittest/test_evaluation_pipeline.py

import sys
import os

# FÜGEN SIE DIESEN BLOCK HINZU, UM DEN PROJEKTPFAD ZU KORRIGIERIEN
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import pytest
from unittest.mock import MagicMock, patch

# Die zu testenden Klassen importieren
from evaluation_pipeline import (
    EvaluationPipeline,
    BleuMetric,
    RougeMetric,
    LlmJudgeMetric,
    LlmHandler
)

# --- Teil 1: Einzelne Metriken testen ---

def test_bleu_metric():
    """Testet die BleuMetric mit bekannten Werten."""
    metric = BleuMetric()
    ground_truth = "the quick brown fox jumps over the lazy dog"
    generated_text = "the quick brown fox jumps over the lazy dog"
    # Bei exakter Übereinstimmung sollte der Score nahe 1.0 sein
    assert metric.calculate(ground_truth, generated_text) == pytest.approx(1.0)
    
    generated_text_half = "the quick brown fox"
    # Bei teilweiser Übereinstimmung sollte der Score kleiner sein
    assert metric.calculate(ground_truth, generated_text_half) < 0.8

def test_rouge_metric():
    """Testet die RougeMetric mit bekannten Werten."""
    metric = RougeMetric()
    ground_truth = "the quick brown fox jumps over the lazy dog"
    generated_text = "the quick brown fox jumps over the lazy dog"
    assert metric.calculate(ground_truth, generated_text) == pytest.approx(1.0)

    generated_text_no_match = "a completely different sentence"
    assert metric.calculate(ground_truth, generated_text_no_match) == pytest.approx(0.0)

def test_llm_judge_metric():
    """Testet die LlmJudgeMetric, indem der LlmHandler gemockt wird."""
    mock_llm_handler = MagicMock(spec=LlmHandler)
    # Simulieren, dass der LLM die Zahl "4" als String zurückgibt
    mock_llm_handler.generate_response.return_value = "4.0"

    metric = LlmJudgeMetric(llm_handler=mock_llm_handler)
    score = metric.calculate("original", "imitation")

    # Überprüfen, ob der Score korrekt als float geparst wurde
    assert score == 4.0
    # Überprüfen, ob der Prompt die richtigen Texte enthält
    prompt_sent_to_llm = mock_llm_handler.generate_response.call_args[0][0]
    assert 'ORIGINALTEXT: "original"' in prompt_sent_to_llm
    assert 'GENERIERTE ANTWORT: "imitation"' in prompt_sent_to_llm

# --- Teil 2: Die Pipeline selbst testen ---

def test_evaluation_pipeline_integration():
    """Testet, ob die EvaluationPipeline ihre Metriken korrekt aufruft."""
    # Erstelle Mock-Metriken
    mock_metric_1 = MagicMock()
    mock_metric_1.name = "mock_bleu"
    mock_metric_1.calculate.return_value = 0.8

    mock_metric_2 = MagicMock()
    mock_metric_2.name = "mock_rouge"
    mock_metric_2.calculate.return_value = 0.9

    metrics_to_use = [mock_metric_1, mock_metric_2]
    pipeline = EvaluationPipeline(metrics=metrics_to_use)

    results = pipeline.evaluate("ground_truth", "generated_text")

    # Überprüfen, ob die `calculate` Methode jeder Metrik aufgerufen wurde
    mock_metric_1.calculate.assert_called_once_with("ground_truth", "generated_text")
    mock_metric_2.calculate.assert_called_once_with("ground_truth", "generated_text")

    # Überprüfen, ob die Ergebnisse korrekt aggregiert wurden
    expected_results = {"mock_bleu": 0.8, "mock_rouge": 0.9}
    assert results == expected_results

def test_evaluation_pipeline_empty_init():
    """Stellt sicher, dass die Pipeline nicht mit einer leeren Metrik-Liste initialisiert werden kann."""
    with pytest.raises(ValueError, match="Die Metrik-Liste darf nicht leer sein."):
        EvaluationPipeline(metrics=[])