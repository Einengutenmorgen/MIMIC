import os
import json
import pathlib
from logging_config import logger
import pandas as pd
import math

from masking import mask_text
from file_cache import get_file_cache


def load_user_history(file_path):
    """
    Load user history from a JSON file.
    
    :param file_path: Path to the JSON file containing user history.
    :return: Dictionary containing user history data.
    """
    cache = get_file_cache()
    
    # Use cached data to avoid repeated file reads
    cached_data = cache.read_file_with_cache(file_path)
    if cached_data is None:
        return {}
    
    # Get first line (index 0) data
    if 0 not in cached_data:
        logger.warning(f"File {file_path} is empty")
        return {}
    
    histories = cached_data[0]
    if histories is None:
        logger.error(f"Error parsing JSON from {file_path}")
        return {}
    
    logger.info(f"User history loaded from {file_path}")
    
    if not isinstance(histories, dict):
        logger.warning(f"Invalid format in {file_path}, expected a dictionary for 'histories'.")
        return {}
    
    return histories

def get_formatted_user_historie(file_path):
    """
    Get formatted user history from a JSON file.
    
    :param file_path: Path to the JSON file containing user history.
    :return: Formatted string representation of user history.
    """
    user_history = load_user_history(file_path)
    if not user_history:
        return "No user history available."

    user_id = user_history.get('user_id', 'Unknown')
    screen_name = user_history.get('tweets', [{}])[0].get('screen_name', 'Unknown') if user_history.get('tweets') else 'Unknown'
    tweets = user_history.get('tweets', [])

    header = f"User ID: {user_id}, Screen Name: {screen_name} \n"
    posts = []
    replies = []
    for tweet in tweets:
        #posts
        if tweet.get('reply_to_id') == None:
            posts.append(f"{tweet.get('full_text')}")
        else:
            reply= tweet.get('full_text')
            previous_message= tweet.get('previous_message')
            if previous_message is not None:
                full_text = f"Previous message: {previous_message} \n {screen_name} replied:{reply} \n" 
                replies.append(f"{full_text}")
    
    formatted_posts = header + "Posts from User"+"\n" +"-"+ "\n - ".join(posts) +"\n" +"\n"+"Replies from User: \n -" + "\n -".join(replies)
    #print(f"Formatted posts: {formatted_posts}")
    return formatted_posts

#load Stimulus 
def load_stimulus(file_path):
    """
    Load user Stimulus from a JSON file.
    Uses preprocessed masked_text field when available, falls back to dynamic masking.
    
    :param file_path: Path to the JSON file containing user history.
    :return: a list of tuples containing formatted stimulus, whether it is a post or reply, and the tweet ID.
    """
    cache = get_file_cache()
    # Use cached data to avoid repeated file reads
    cached_data = cache.read_file_with_cache(file_path)
    if cached_data is None:
        return []
    
    # Get line 2 (index 1) data
    if 1 not in cached_data:
        logger.warning(f"File {file_path} has fewer than 2 lines")
        return []
    
    user_holdout = cached_data[1]
    if user_holdout is None:
        logger.error(f"Error parsing JSON from line 2 in {file_path}")
        return []
    
    logger.info(f"User stimulus loaded from {file_path}")
    all_stimuli = []
    
    # Check if it is a post or a reply
    user_holdout_tweets = user_holdout.get('tweets', [])
    
    for stimulus in user_holdout_tweets:
        post_id = stimulus.get('tweet_id')
        
        if stimulus.get('reply_to_id') is None and stimulus.get('previous_message') is None:
            is_post = True
            logger.info(f"Stimulus is a post: {stimulus.get('full_text')}")
        else:
            is_post = False
        
        if is_post:
            # Use preprocessed masked_text if available, otherwise fall back to dynamic masking
            if 'masked_text' in stimulus and stimulus['masked_text']:
                stimulus_post = stimulus['masked_text']
                logger.info(f"Using preprocessed masked text for post: {stimulus_post}")
            else:
                # Fallback to dynamic masking if no preprocessed text available
                stimulus_post = f"{stimulus.get('full_text')}"
                logger.warning(f"No preprocessed masked_text found for tweet {post_id}, using dynamic masking")
                
                try:
                    stimulus_post = mask_text(stimulus_post)
                    if stimulus_post:
                        logger.info(f"Dynamically masked post: {stimulus_post}")
                    else:
                        logger.error("No valid post text found in the stimulus.")
                        stimulus_post = "Post text could not be processed."
                except Exception as e:
                    logger.error(f"Error masking post text: {e}")
                    stimulus_post = "Post text could not be processed."
            
            stimulus_formatted = f"Post: {stimulus_post}"
        else:
            # Replies don't need masking according to your comment
            reply_stimulus = stimulus.get('previous_message')
            stimulus_formatted = f"Previous message: {reply_stimulus}"
        
        all_stimuli.append((stimulus_formatted, is_post, post_id))
    
    return all_stimuli        
def load_predictions(run_id, file_path):
    """
    Load predictions from a JSON Lines file based on run_id.
    Data is expected to be on line 3 (index 2).
    
    :param run_id: Run ID to filter the predictions.
    :param file_path: Path to the JSONL file containing user history.
    :return: List of predictions for the specified run_id.
    """
    cache = get_file_cache()
    
    # Use cached data to avoid repeated file reads
    cached_data = cache.read_file_with_cache(file_path)
    if cached_data is None:
        return []
    
    # Check if file has at least 3 lines (line 3 is index 2)
    if 2 not in cached_data:
        logger.error(f"File {file_path} has fewer than 3 lines")
        return []
    
    # Get line 3 (index 2) data
    line_2_data = cached_data[2]
    if line_2_data is None:
        logger.error(f"Failed to parse line 3 in {file_path}")
        return []
    
    # Extract predictions for the specified run_id
    if 'runs' not in line_2_data or not isinstance(line_2_data['runs'], list):
        logger.warning(f"No valid 'runs' found in line 3 of {file_path}")
        return []
    
    predictions = []
    for run in line_2_data['runs']:
        if run.get('run_id') == run_id:
            predictions.extend(run.get('imitations', []))
    
    return predictions

def load_orginal(file_path, tweet_id):
    """
    Load original tweets from a predefined file path.
    based on a given Tweet-id
    :return: Original tweet text or None if not found.
    """
    cache = get_file_cache()
    
    # Use cached data to avoid repeated file reads
    cached_data = cache.read_file_with_cache(file_path)
    if cached_data is None:
        return None
    
    # Check if file has at least 3 lines (line 2 is index 1)
    if 1 not in cached_data:
        logger.error(f"File {file_path} has fewer than 2 lines")
        return None
    
    # Get line 2 (index 1) data
    line_1_data = cached_data[1]
    if line_1_data is None:
        logger.error(f"Failed to parse line 2 in {file_path}")
        return None
    
    # Extract tweets and find the one with matching tweet_id
    holdout_tweets = line_1_data.get('tweets', [])
    for tweet in holdout_tweets:
        if tweet.get('tweet_id') == tweet_id:
            original_tweet = tweet.get('full_text')
            if original_tweet is not None:
                return original_tweet
    
    # Tweet not found
    return None
    
def load_predictions_orginales_formated(run_id, file_path):
    """
    Wrapper for loading_predictions and load_orginals
    :param run_id: Run ID to filter the predictions.
    :param file_path: Path to the JSONL file containing user history.
    :return: List of formatted predictions and orginals for the specified run_id.
    """

    predictions = load_predictions(run_id, file_path)
    if not predictions:
        logger.error(f"No predictions found for run_id {run_id} in {file_path}")
        return []
    
    # Process each prediction and add original tweet
    # The caching in load_orginal() will prevent redundant file reads
    for prediction in predictions:
        tweet_id = prediction.get('tweet_id')
        original_tweet = load_orginal(file_path, tweet_id)
        if original_tweet:
            # Add the original tweet to the prediction
            prediction['original'] = original_tweet
        else:
            logger.error(f"No original tweet found for tweet_id {tweet_id}")
            
    return predictions


def load_results_for_reflection(run_id, file_path):
    """
    Load results for reflection based on run_id.
    
    :param run_id: Run ID to filter the results.
    :param file_path: Path to the JSONL file containing the user data.
    :return: Dictionary with user_id, run_id, persona, timestamp, best_preds, original of best_preds,
             worst_preds, original of worst_preds, bleu scores, rouge scores
    """
    cache = get_file_cache()
    
    # Use cached data to avoid repeated file reads
    cached_data = cache.read_file_with_cache(file_path)
    if cached_data is None:
        return None
    
    # Check if file has at least 4 lines
    if 2 not in cached_data or 3 not in cached_data:
        logger.error(f"File {file_path} has fewer than 4 lines")
        return None
    
    # Get line 3 (index 2) for persona data
    persona_data = cached_data[2]
    if persona_data is None:
        logger.error(f"Error parsing JSON from line 3 in {file_path}")
        return None
    
    # Get line 4 (index 3) for evaluation data
    eval_data = cached_data[3]
    if eval_data is None:
        logger.error(f"Error parsing JSON from line 4 in {file_path}")
        return None
    
    # Get user_id (should be in both, but let's prioritize eval_data)
    user_id = eval_data.get('user_id') or persona_data.get('user_id')
    
    # Find the specific run in persona data
    runs = persona_data.get('runs', [])
    target_run = None
    for run in runs:
        if run.get('run_id') == run_id:
            target_run = run
            break
    
    if not target_run:
        logger.error(f"Run ID {run_id} not found in persona data")
        return None
    
    # Find the evaluation for this run
    evaluations = eval_data.get('evaluations', [])
    target_evaluation = None
    for evaluation in evaluations:
        if evaluation.get('run_id') == run_id:
            target_evaluation = evaluation
            break
    
    if not target_evaluation:
        logger.error(f"Evaluation for run ID {run_id} not found")
        return None
    
    # Extract evaluation results
    eval_results = target_evaluation.get('evaluation_results', {})
    overall = eval_results.get('overall', {})
    best_predictions = eval_results.get('best_predictions', [])
    worst_predictions = eval_results.get('worst_predictions', [])
    
    # Extract best predictions and their originals
    best_preds = [pred.get('prediction', '') for pred in best_predictions]
    best_originals = [pred.get('reference', '') for pred in best_predictions]
    
    # Extract worst predictions and their originals
    worst_preds = [pred.get('prediction', '') for pred in worst_predictions]
    worst_originals = [pred.get('reference', '') for pred in worst_predictions]
    
    # Dynamically add all metrics from the 'overall' dictionary
    reflection_results = {
        'user_id': user_id,
        'run_id': run_id,
        'persona': target_run.get('persona', ''),
        'timestamp': target_evaluation.get('timestamp', ''),
        'best_preds': best_preds,
        'best_originals': best_originals,
        'worst_preds': worst_preds,
        'worst_originals': worst_originals,
    }
    
    # Add all scores from the 'overall' dictionary to the results
    if overall:
        reflection_results.update(overall)
        
    return reflection_results
    

def load_latest_improved_persona(run_id, file_path):
    """
    Load the improved persona from the latest iteration for a given run_id.
    
    :param run_id: Run ID to filter the reflections.
    :param file_path: Path to the JSONL file containing the reflections.
    :return: String containing the improved persona from the latest iteration, or None if not found.
    """
    cache = get_file_cache()
    
    # Use cached data to avoid repeated file reads
    cached_data = cache.read_file_with_cache(file_path)
    if cached_data is None:
        return None
    
    # Check if file has at least 5 lines (reflections are on line 5, index 4)
    if 4 not in cached_data:
        logger.error(f"File {file_path} has fewer than 5 lines")
        return None
    
    # Get line 5 (index 4) for reflection data
    reflections_data = cached_data[4]
    if reflections_data is None:
        logger.error(f"Error parsing JSON from line 5 in {file_path}")
        return None
    
    # Filter reflections by run_id
    reflections = reflections_data.get('reflections', [])
    run_reflections = [r for r in reflections if r.get('run_id') == run_id]
    
    if not run_reflections:
        logger.warning(f"No reflections found for run_id {run_id}")
        return None
    
    # Find the reflection with the highest iteration number
    latest_reflection = max(run_reflections, key=lambda x: x.get('iteration', 0))
    
    # Extract improved_persona
    improved_persona = latest_reflection.get('reflection_results', {}).get('improved_persona', '')
    
    if improved_persona:
        logger.info(f"Loaded improved persona from run_id {run_id}, iteration {latest_reflection.get('iteration')}")
        return improved_persona
    else:
        logger.warning(f"No improved_persona found for run_id {run_id}")
        return None


if __name__ == "__main__":
    # Example usage
    test_file_path = r'C:\Users\Christoph.Hau\Experimente\ha\data\raw\users\534023.0.jsonl'
    
    user_history = get_formatted_user_historie(test_file_path)
    print(user_history)
