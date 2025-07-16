from evaluate import load

rouge = load("rouge")
bleu = load("bleu")


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
    
    # ROUGE berechnen
    rouge_result = rouge.compute(predictions=predictions, references=references_rouge, use_stemmer=False, use_aggregator=True)
    results['rouge'] = rouge_result
    
    # BLEU berechnen  
    bleu_result = bleu.compute(predictions=predictions, references=references_bleu, use_aggregator=True)
    results['bleu'] = bleu_result
    
    
    return results

import numpy as np


def evaluate_with_individual_scores(list_of_imitation_dict):
    """
    Evaluate predictions with both overall and individual scores.
    Returns best/worst predictions based on combined ROUGE-L and BLEU scores.
    """
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
        rouge_individual = rouge.compute(
            predictions=[pred], 
            references=[ref], 
            use_stemmer=False
        )
        bleu_individual = bleu.compute(
            predictions=[pred], 
            references=[[ref]]
        )
        
        # Combined Score für Ranking (gewichteter Durchschnitt)
        combined_score = (rouge_individual['rougeL'] * 0.6 + 
                         bleu_individual['bleu'] * 0.4)
        
        individual_scores.append({
            'index': i,
            'prediction': pred,
            'reference': ref,
            'rouge': rouge_individual,
            'bleu': bleu_individual,
            'combined_score': combined_score,
            'tweet_id': dict_item.get('tweet_id', f'item_{i}')
        })
    
    # Sortiere nach combined_score
    individual_scores.sort(key=lambda x: x['combined_score'], reverse=True)
    
    # Overall Scores
    rouge_overall = rouge.compute(
        predictions=predictions, 
        references=references_rouge, 
        use_stemmer=False
    )
    bleu_overall = bleu.compute(
        predictions=predictions, 
        references=references_bleu
    )
    
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
    
    return {
        'overall': {
            'rouge': rouge_overall,
            'bleu': bleu_overall,
            'total_samples': len(predictions)
        },
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