import os
import json
import pathlib
from logging_config import logger
import pandas as pd
import math

from masking import mask_text


def load_user_history(file_path):
    """
    Load user history from a JSON file.
    
    :param file_path: Path to the JSON file containing user history.
    :return: Dictionary containing user history data.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return {}

    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            data = pd.read_json(path_or_buf=file_path, lines=True, nrows=1)
            histories=data.iloc[0].to_dict()
            #print(histories)
            #data= json.load(file)
            if data.empty:
                logger.warning(f"File {file_path} is empty")
                return {}
            logger.info(f"User history loaded from {file_path}")
            #histories = data.get('histories', {})
            if not isinstance(histories, dict):
                logger.warning(f"Invalid format in {file_path}, expected a dictionary for 'histories'.")
                #print(f"Invalid format in {file_path}, expected a dictionary for 'histories'.")
                return {}
            return histories
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {file_path}: {e}")
            return {}

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
    
    :param file_path: Path to the JSON file containing user history.
    :return: a list of tuples containing formatted stimulus, whether it is a post or reply, and the tweet ID.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return {}

    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            data = pd.read_json(path_or_buf=file_path, lines=True, nrows=2)
            user_holdout=data.iloc[1].to_dict()
            
            if data.empty:
                logger.warning(f"File {file_path} is empty")
                return {}
            logger.info(f"User stimulus loaded from {file_path}")
            all_stimuli = []
            #check if it is a post or a reply
            user_holdout_tweets = user_holdout.get('tweets')
            for stimulus in user_holdout_tweets:
                post_id = stimulus.get('tweet_id') 
                if stimulus.get('reply_to_id') is None and stimulus.get('previous_message') is None:
                    is_post = True
                    logger.info(f"Stimulus is a post: {stimulus.get('full_text')}")
                else:
                    is_post = False
            
            
                if is_post:
                    stimulus_post = f"{stimulus.get('full_text')}"
                    try:
                        stimulus_post = mask_text(stimulus_post)
                        if stimulus_post:
                            logger.info(f"Stimulus is a post: {stimulus_post}")
                        else:
                            logger.error("No valid post text found in the stimulus.")
                    except Exception as e:
                        logger.error(f"Error masking post text: {e}")
                        stimulus_post = "Post text could not be processed."
                    stimulus_formatted = f"Post: {stimulus_post}"
                else:
                    reply_stimulus =  stimulus.get('previous_message')
                    stimulus_formatted = f"Previous message: {reply_stimulus}"
                all_stimuli.append((stimulus_formatted, is_post, post_id))
            return all_stimuli 
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {file_path}: {e}")
            return {}
        
def load_predictions(run_id, file_path):
    """
    Load predictions from a JSON Lines file based on run_id.
    Data is expected to be on line 3 (index 2).
    
    :param run_id: Run ID to filter the predictions.
    :param file_path: Path to the JSONL file containing user history.
    :return: List of predictions for the specified run_id.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            
            # Check if file has at least 3 lines
            if len(lines) < 3:
                logger.error(f"File {file_path} has fewer than 3 lines")
                return []
            
            # Parse line 3 (index 2)
            try:
                data = json.loads(lines[2].strip())
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from line 3 in {file_path}: {e}")
                return []
            
            # Extract predictions for the specified run_id
            if 'runs' not in data or not isinstance(data['runs'], list):
                logger.warning(f"No valid 'runs' found in line 3 of {file_path}")
                return []
            
            predictions = []
            for run in data['runs']:
                if run.get('run_id') == run_id:
                    predictions.extend(run.get('imitations'))
            
            return predictions
            
    except IOError as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return []

def load_orginal(file_path, tweet_id):
    """
    Load original tweets from a predefined file path.
    based on a given Tweet-id
    :return: List of original tweets.
    """

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return []

    try:
        original_tweet = None
        with open(file_path, 'r', encoding='utf-8') as file:
            # Read all lines from the file
            lines = file.readlines()
            # Check if file has at least 3 lines
            if len(lines) < 3:
                logger.error(f"File {file_path} has fewer than 3 lines")
                return []
            # Parse line 2 (index 1)
            try:
                data = json.loads(lines[1].strip())
            
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from line 3 in {file_path}: {e}")
                return []
            
            # Extract predictions for the specified run_id
            holdout_tweets = data.get('tweets')
            for tweet in holdout_tweets:
                if tweet.get('tweet_id') == tweet_id:
                    original_tweet = tweet.get('full_text')
                if original_tweet is not None:
                    return original_tweet

    except IOError as e:
        logger.error(f"Error reading file {file_path}: {e}")
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
    
    for prediction in predictions:
        tweet_id=prediction.get('tweet_id')
        load_orginal(file_path, tweet_id)


    formatted_predictions = []
    formatted_orginals = []
    for prediction in predictions:
        tweet_id = prediction.get('tweet_id')
        original_tweet = load_orginal(file_path, tweet_id)
        if original_tweet:
            # Append the formatted prediction with original tweet
            prediction['original'] = original_tweet
            prediction_tweet = prediction.get('imitation')
            formatted_predictions.append(prediction_tweet)
            formatted_orginals.append(original_tweet)
        else:
            logger.error(f"No original tweet found for tweet_id {tweet_id}")
            
    #return formatted_orginals, formatted_predictions
    return predictions      



if __name__ == "__main__":
    # Example usage
    test_file_path = r'C:\Users\Christoph.Hau\Experimente\ha\data\raw\users\534023.0.jsonl'
    
    user_history = get_formatted_user_historie(test_file_path)
    print(user_history)
