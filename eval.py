import yaml
from evaluate import load
import numpy as np
from llm import call_ai
import re
from templates import format_template

# Load the configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Get evaluation metrics from correct config path
evaluation_config = config.get("evaluation", {})
evaluation_metrics = evaluation_config.get("metrics", [])

# Convert list to set for easier checking
metrics_set = set(evaluation_metrics) if isinstance(evaluation_metrics, list) else set()

# Load metrics based on the configuration
rouge = None
bleu = None
perplexity = None
bertscore = None

if "rouge" in metrics_set:
    rouge = load("rouge")
    print("Loaded ROUGE metric")
    
if "bleu" in metrics_set:
    bleu = load("bleu")
    print("Loaded BLEU metric")
    
if "perplexity" in metrics_set:
    perplexity = load("perplexity", module_type="metric")
    print("Loaded Perplexity metric")
    
if "bertscore" in metrics_set or "bert_score" in metrics_set:
    bertscore = load("bertscore")
    print("Loaded BERTScore metric")


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
    Evaluate the predictions against the references using configured metrics.
    """
    predictions = []
    references_rouge = []
    references_bleu = []
    
    for dict_item in list_of_imitation_dict:
        predictions.append(dict_item['imitation'])
        references_rouge.append(dict_item['original'])
        references_bleu.append([dict_item['original']])
    
    results = {}
    
    if rouge is not None:
        rouge_result = rouge.compute(predictions=predictions, references=references_rouge, use_stemmer=False, use_aggregator=True)
        results['rouge'] = rouge_result
        print(f"Computed ROUGE scores: {rouge_result}")
    
    if bleu is not None:
        bleu_result = bleu.compute(predictions=predictions, references=references_bleu, use_aggregator=True)
        results['bleu'] = bleu_result
        print(f"Computed BLEU score: {bleu_result}")
    
    if perplexity is not None:
        perplexity_result = perplexity.compute(predictions=predictions, model_id='gpt2')
        results['perplexity'] = perplexity_result
        print(f"Computed Perplexity: {perplexity_result}")
    
    if bertscore is not None:
        bertscore_result = bertscore.compute(predictions=predictions, references=references_rouge, lang="en")
        results['bertscore'] = bertscore_result
        print(f"Computed BERTScore")
    
    return results


def evaluate_with_individual_scores(list_of_imitation_dict, config_override=None):
    """
    Evaluate predictions with both overall and individual scores.
    Returns best/worst predictions based on combined scores.
    """
    # Use override config or load from file
    if config_override:
        current_config = config_override
    else:
        current_config = config
    
    # Get evaluation metrics - handle both possible config structures
    eval_config = current_config.get("evaluation", {})
    metrics_list = eval_config.get("metrics", [])
    
    # Convert to set for easier checking
    metrics_enabled = set(metrics_list) if isinstance(metrics_list, list) else set()
    
    print(f"Enabled metrics: {metrics_enabled}")
    
    predictions = []
    references_rouge = []
    references_bleu = []
    individual_scores = []
    
    # Process each imitation
    for i, dict_item in enumerate(list_of_imitation_dict):
        pred = dict_item['imitation']
        ref = dict_item['original']
        
        predictions.append(pred)
        references_rouge.append(ref)
        references_bleu.append([ref])
        
        # Calculate individual scores
        rouge_individual = None
        if "rouge" in metrics_enabled and rouge is not None:
            try:
                rouge_individual = rouge.compute(
                    predictions=[pred],
                    references=[ref],
                    use_stemmer=False
                )
                print(f"Item {i}: ROUGE scores calculated")
            except Exception as e:
                print(f"Error computing ROUGE for item {i}: {e}")

        bleu_individual = None
        if "bleu" in metrics_enabled and bleu is not None:
            try:
                bleu_individual = bleu.compute(
                    predictions=[pred],
                    references=[[ref]]
                )
                print(f"Item {i}: BLEU score = {bleu_individual.get('bleu', 0):.4f}")
            except Exception as e:
                print(f"Error computing BLEU for item {i}: {e}")
        
        perplexity_individual = None
        if "perplexity" in metrics_enabled and perplexity is not None:
            try:
                perplexity_individual = perplexity.compute(predictions=[pred], model_id='gpt2')
            except Exception as e:
                print(f"Error computing perplexity for item {i}: {e}")

        bertscore_individual = None
        if ("bertscore" in metrics_enabled or "bert_score" in metrics_enabled) and bertscore is not None:
            try:
                bertscore_individual = bertscore.compute(predictions=[pred], references=[ref], lang="en")
            except Exception as e:
                print(f"Error computing BERTScore for item {i}: {e}")
        
        # LLM-based evaluation
        llm_evaluation = None
        if "gpt_eval" in metrics_enabled:
            try:
                llm_evaluation = evaluate_with_llm(pred, ref)
            except Exception as e:
                print(f"Error in LLM evaluation for item {i}: {e}")

        # Calculate combined score (weighted average)
        combined_score = 0
        weight_sum = 0
        
        if rouge_individual:
            rouge_l_score = rouge_individual.get('rougeL', 0)
            combined_score += rouge_l_score * 0.4
            weight_sum += 0.4
            
        if bleu_individual:
            bleu_score = bleu_individual.get('bleu', 0)
            combined_score += bleu_score * 0.3
            weight_sum += 0.3
            
        if bertscore_individual:
            bert_precision = np.mean(bertscore_individual.get('precision', [0]))
            combined_score += bert_precision * 0.3
            weight_sum += 0.3
        
        # Normalize combined score if not all metrics were available
        if weight_sum > 0:
            combined_score = combined_score / weight_sum
        
        scores_entry = {
            'index': i,
            'prediction': pred,
            'reference': ref,
            'combined_score': float(combined_score),
            'tweet_id': dict_item.get('tweet_id', f'item_{i}')
        }

        if rouge_individual:
            scores_entry['rouge'] = rouge_individual
        if bleu_individual:
            scores_entry['bleu'] = bleu_individual
        if perplexity_individual:
            scores_entry['perplexity'] = perplexity_individual
        if bertscore_individual:
            scores_entry['bertscore'] = {
                'precision': bertscore_individual['precision'][0] if bertscore_individual.get('precision') else 0,
                'recall': bertscore_individual['recall'][0] if bertscore_individual.get('recall') else 0,
                'f1': bertscore_individual['f1'][0] if bertscore_individual.get('f1') else 0,
            }
        if llm_evaluation:
            scores_entry['llm_evaluation'] = llm_evaluation
        
        individual_scores.append(scores_entry)
    
    # Sort by combined_score
    individual_scores.sort(key=lambda x: x['combined_score'], reverse=True)
    
    # Calculate overall scores
    overall_scores = {}
    
    if "rouge" in metrics_enabled and rouge is not None:
        try:
            overall_rouge = rouge.compute(
                predictions=predictions,
                references=references_rouge,
                use_stemmer=False
            )
            overall_scores['rouge'] = overall_rouge
            print(f"Overall ROUGE scores: {overall_rouge}")
        except Exception as e:
            print(f"Error computing overall ROUGE: {e}")
            
    if "bleu" in metrics_enabled and bleu is not None:
        try:
            overall_bleu = bleu.compute(
                predictions=predictions,
                references=references_bleu
            )
            overall_scores['bleu'] = overall_bleu
            print(f"Overall BLEU score: {overall_bleu}")
        except Exception as e:
            print(f"Error computing overall BLEU: {e}")
            
    if "perplexity" in metrics_enabled and perplexity is not None:
        try:
            perplexity_overall = perplexity.compute(predictions=predictions, model_id='gpt2')
            overall_scores['perplexity'] = {
                'mean_perplexity': perplexity_overall['mean_perplexity'],
                'perplexities': perplexity_overall['perplexities']
            }
        except Exception as e:
            print(f"Error computing overall perplexity: {e}")
            
    if ("bertscore" in metrics_enabled or "bert_score" in metrics_enabled) and bertscore is not None:
        try:
            bertscore_overall = bertscore.compute(predictions=predictions, references=references_rouge, lang="en")
            overall_scores['bertscore'] = {
                'precision': np.mean(bertscore_overall['precision']),
                'recall': np.mean(bertscore_overall['recall']),
                'f1': np.mean(bertscore_overall['f1']),
            }
        except Exception as e:
            print(f"Error computing overall BERTScore: {e}")

    # Select best and worst predictions
    total_items = len(individual_scores)
    if total_items >= 6:
        best_predictions = individual_scores[:3]
        worst_predictions = individual_scores[-3:]
    else:
        if total_items <= 1:
            best_predictions = individual_scores
            worst_predictions = []
        else:
            take_from_each_end = total_items // 2
            best_predictions = individual_scores[:take_from_each_end]
            worst_predictions = individual_scores[-take_from_each_end:]
    
    overall_scores['total_samples'] = len(predictions)

    # Calculate statistics
    if individual_scores:
        combined_scores = [s['combined_score'] for s in individual_scores]
        statistics = {
            'mean_combined_score': float(np.mean(combined_scores)),
            'std_combined_score': float(np.std(combined_scores)),
            'best_score': float(individual_scores[0]['combined_score']) if individual_scores else 0,
            'worst_score': float(individual_scores[-1]['combined_score']) if individual_scores else 0
        }
    else:
        statistics = {
            'mean_combined_score': 0.0,
            'std_combined_score': 0.0,
            'best_score': 0,
            'worst_score': 0
        }
    
    print(f"Evaluation complete: Mean combined score = {statistics['mean_combined_score']:.4f}")
    
    return {
        'overall': overall_scores,
        'individual_scores': individual_scores,
        'best_predictions': best_predictions,
        'worst_predictions': worst_predictions,
        'statistics': statistics
    }