import yaml
from evaluate import load

# Load the configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
evaluation_metrics = config.get("evaluation", {}).get("metrics", [])

# Load metrics based on the configuration
if "rouge" in evaluation_metrics:
    rouge = load("rouge")
if "bleu" in evaluation_metrics:
    bleu = load("bleu")
if "perplexity" in evaluation_metrics:
    perplexity = load("perplexity", module_type="metric")
if "bertscore" in evaluation_metrics:
    bertscore = load("bertscore")

from llm import call_ai
import re
from templates import format_template


def evaluate_with_llm(prediction, reference):
    """
    Evaluates coherence and consistency of a prediction against a reference using an LLM.
    """
    template = format_template(
        "llm_evaluation_template",
        reference=reference,
        prediction=prediction
    )
    try:
        response = call_ai(template, model="google")

        coherence_match = re.search(r"Coherence: (\d)", response)
        consistency_match = re.search(r"Consistency: (\d)", response)
        justification_match = re.search(r"Justification: (.*)", response)

        coherence = int(coherence_match.group(1)) if coherence_match else 0
        consistency = int(consistency_match.group(1)) if consistency_match else 0
        justification = justification_match.group(1).strip() if justification_match else "No justification provided."

        return {
            "coherence_score": coherence,
            "consistency_score": consistency,
            "justification": justification,
            "raw_response": response,
        }
    except Exception as e:
        print(f"Error during LLM evaluation: {e}")
        return {
            "coherence_score": 0,
            "consistency_score": 0,
            "justification": "Evaluation failed.",
            "raw_response": "",
        }

def evaluate(list_of_imitation_dict):
    """
    Evaluate the predictions against the references using ROUGE and BLEU metrics.
    method uses output from load_predictions_orginales_formated
    :input list_of_imitaion_dict: List of dictionaries containing 'predictions' and 'references'.
        
    Returns:
        dict: A dictionary containing the ROUGE and BLEU scores.
    """
    # Listen für alle Predictions und References aufbauen
    predictions = []
    references_rouge = []
    references_bleu = []
    
    for dict_item in list_of_imitation_dict:
        predictions.append(dict_item['imitation'])
        references_rouge.append(dict_item['original'])
        # BLEU braucht Listen von Listen für multiple Referenzen
        references_bleu.append([dict_item['original']])
    
    results = {}
    
    if "rouge" in evaluation_metrics:
        rouge_result = rouge.compute(predictions=predictions, references=references_rouge, use_stemmer=False, use_aggregator=True)
        results['rouge'] = rouge_result
    
    if "bleu" in evaluation_metrics:
        bleu_result = bleu.compute(predictions=predictions, references=references_bleu, use_aggregator=True)
        results['bleu'] = bleu_result
    
    if "perplexity" in evaluation_metrics:
        perplexity_result = perplexity.compute(predictions=predictions, model_id='gpt2')
        results['perplexity'] = perplexity_result
    
    if "bertscore" in evaluation_metrics:
        bertscore_result = bertscore.compute(predictions=predictions, references=references_rouge, lang="en")
        results['bertscore'] = bertscore_result
    
    return results
  
import numpy as np


def evaluate_with_individual_scores(list_of_imitation_dict, config=None):
    """
    Evaluate predictions with both overall and individual scores.
    Returns best/worst predictions based on combined ROUGE-L and BLEU scores.
    """
    if config is None:
        # Load the configuration if not provided
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
    
    evaluation_metrics = config.get("metrics", {})
    predictions = []
    references_rouge = []
    references_bleu = []
    individual_scores = []
    
    # Sammle alle Daten und berechne individuelle Scores
    for i, dict_item in enumerate(list_of_imitation_dict):
        pred = dict_item['imitation']
        ref = dict_item['original']
        
        predictions.append(pred)
        references_rouge.append(ref)
        references_bleu.append([ref])
        
        # Individuelle Scores berechnen
        rouge_individual = None
        if evaluation_metrics.get("rouge"):
            rouge_individual = rouge.compute(
                predictions=[pred],
                references=[ref],
                use_stemmer=False
            )

        bleu_individual = None
        if evaluation_metrics.get("bleu"):
            bleu_individual = bleu.compute(
                predictions=[pred],
                references=[[ref]]
            )
        
        perplexity_individual = None
        if evaluation_metrics.get("perplexity"):
            perplexity_individual = perplexity.compute(predictions=[pred], model_id='gpt2')

        bertscore_individual = None
        if evaluation_metrics.get("bert_score"):
            bertscore_individual = bertscore.compute(predictions=[pred], references=[ref], lang="en")
        
        # LLM-based evaluation
        llm_evaluation = None
        if evaluation_metrics.get("gpt_eval"):
            llm_evaluation = evaluate_with_llm(pred, ref)

        # Combined Score für Ranking (gewichteter Durchschnitt)
        combined_score = 0
        if rouge_individual:
            combined_score += rouge_individual['rougeL'] * 0.4
        if bleu_individual:
            combined_score += bleu_individual['bleu'] * 0.3
        if bertscore_individual:
            combined_score += np.mean(bertscore_individual['precision']) * 0.3 # BertScore precision
        
        scores_entry = {
            'index': i,
            'prediction': pred,
            'reference': ref,
            'combined_score': combined_score,
            'tweet_id': dict_item.get('tweet_id', f'item_{i}')
        }

        if rouge_individual:
            scores_entry['rouge'] = rouge_individual
        if bleu_individual:
            scores_entry['bleu'] = bleu_individual
        if perplexity_individual:
            scores_entry['perplexity'] = perplexity_individual['perplexities'][0]
        if bertscore_individual:
            scores_entry['bertscore'] = {
                'precision': bertscore_individual['precision'][0],
                'recall': bertscore_individual['recall'][0],
                'f1': bertscore_individual['f1'][0],
            }
        if llm_evaluation:
            scores_entry['llm_evaluation'] = llm_evaluation
        
        individual_scores.append(scores_entry)
    
    # Sortiere nach combined_score
    individual_scores.sort(key=lambda x: x['combined_score'], reverse=True)
    
    # Overall Scores
    overall_scores = {}
    if evaluation_metrics.get("rouge"):
        overall_scores['rouge'] = rouge.compute(
            predictions=predictions,
            references=references_rouge,
            use_stemmer=False
        )
    if evaluation_metrics.get("bleu"):
        overall_scores['bleu'] = bleu.compute(
            predictions=predictions,
            references=references_bleu
        )
    if evaluation_metrics.get("perplexity"):
        perplexity_overall = perplexity.compute(predictions=predictions, model_id='gpt2')
        overall_scores['perplexity'] = {
            'mean_perplexity': perplexity_overall['mean_perplexity'],
            'perplexities': perplexity_overall['perplexities']
        }
    if evaluation_metrics.get("bert_score"):
        bertscore_overall = bertscore.compute(predictions=predictions, references=references_rouge, lang="en")
        overall_scores['bertscore'] = {
            'precision': np.mean(bertscore_overall['precision']),
            'recall': np.mean(bertscore_overall['recall']),
            'f1': np.mean(bertscore_overall['f1']),
        }

    # Top 3 best and worst (with overlap prevention for fewer than 6 items)
    total_items = len(individual_scores)
    if total_items >= 6:
        # Standard case: top 3 and bottom 3
        best_predictions = individual_scores[:3]
        worst_predictions = individual_scores[-3:]
    else:
        # Handle cases with fewer than 6 items to avoid overlap
        if total_items <= 1:
            best_predictions = individual_scores
            worst_predictions = []
        else:
            # Calculate how many to take from each end without overlap
            take_from_each_end = total_items // 2
            best_predictions = individual_scores[:take_from_each_end]
            worst_predictions = individual_scores[-take_from_each_end:]
    
    overall_scores['total_samples'] = len(predictions)

    return {
        'overall': overall_scores,
        'individual_scores': individual_scores,
        'best_predictions': best_predictions,
        'worst_predictions': worst_predictions,
        'statistics': {
            'mean_combined_score': np.mean([s['combined_score'] for s in individual_scores]),
            'std_combined_score': np.std([s['combined_score'] for s in individual_scores]),
            'best_score': individual_scores[0]['combined_score'],
            'worst_score': individual_scores[-1]['combined_score']
        }
    }