# JSONL Data Structure Documentation

## Overview

This document describes the structure of a multi-line JSONL (JSON Lines) file containing user data, imitations, evaluations, and reflections for a persona-based text generation system. Each line represents a different type of data in the processing pipeline.

## File Structure

The JSONL file contains exactly 5 lines, each with a specific purpose:

1. **Line 1**: User History
2. **Line 2**: User Holdset
3. **Line 3**: User Imitations
4. **Line 4**: Evaluation Results
5. **Line 5**: Reflections and Improvements

---

## Line 1: User History

**Purpose**: Contains historical tweet data for a user, used for training/analysis.

```json
{
  "user_id": "string",
  "set": "string",
  "tweets": []
}
```

### Fields

- **user_id** (string): Unique identifier for the user
- **set** (string): Dataset classification (e.g., "history", "training")
- **tweets** (array): Collection of historical tweets from the user

---

## Line 2: User Holdset

**Purpose**: Contains holdout tweet data for a user, used for validation/testing.

```json
{
  "user_id": "string",
  "set": "string", 
  "tweets": []
}
```

### Fields

- **user_id** (string): Unique identifier for the user
- **set** (string): Dataset classification (e.g., "holdset", "validation")
- **tweets** (array): Collection of holdout tweets from the user

---

## Line 3: User Imitations

**Purpose**: Contains generated imitations of user tweets using different personas.

```json
{
  "user_id": "string",
  "runs": [
    {
      "persona": "string",
      "imitations": [
        {
          "tweet_id": "string",
          "stimulus": "string",
          "imitation": "string",
          "original": "string"
        }
      ]
    }
  ]
}
```

### Fields

- **user_id** (string): Unique identifier for the user
- **runs** (array): Collection of imitation runs with different personas

#### Run Object Structure

- **persona** (string): Description of the persona used for generating imitations
- **imitations** (array): Collection of individual imitation attempts

#### Imitation Object Structure

- **tweet_id** (string): Unique identifier for the tweet being imitated
- **stimulus** (string): Input or prompt used to generate the imitation
- **imitation** (string): Generated text attempting to mimic the user's style
- **original** (string): The original tweet text being imitated

---

## Line 4: Evaluation Results

**Purpose**: Contains comprehensive evaluation metrics for imitation quality.

```json
{
  "user_id": "string",
  "evaluations": [
    {
      "run_id": "string",
      "timestamp": "string",
      "evaluation_results": {
        "overall": {
          "rouge": {
            "rouge1": "number",
            "rouge2": "number", 
            "rougeL": "number",
            "rougeLsum": "number"
          },
          "bleu": {
            "bleu": "number",
            "precisions": [],
            "brevity_penalty": "number",
            "length_ratio": "number",
            "translation_length": "number",
            "reference_length": "number"
          },
          "total_samples": "number"
        },
        "individual_scores": [],
        "best_predictions": [],
        "worst_predictions": [],
        "statistics": {
          "mean_combined_score": "number",
          "std_combined_score": "number",
          "best_score": "number",
          "worst_score": "number"
        }
      }
    }
  ]
}
```

### Fields

- **user_id** (string): Unique identifier for the user
- **evaluations** (array): Collection of evaluation runs

#### Evaluation Object Structure

- **run_id** (string): Unique identifier for the evaluation run
- **timestamp** (string): ISO 8601 formatted timestamp of when evaluation occurred
- **evaluation_results** (object): Contains all evaluation metrics and results

#### Overall Metrics

- **rouge** (object): ROUGE scores measuring text similarity
  - **rouge1**: ROUGE-1 score (unigram overlap)
  - **rouge2**: ROUGE-2 score (bigram overlap)
  - **rougeL**: ROUGE-L score (longest common subsequence)
  - **rougeLsum**: ROUGE-L sum score
- **bleu** (object): BLEU scores for translation quality
  - **bleu**: Overall BLEU score
  - **precisions**: Array of precision scores at different n-gram levels
  - **brevity_penalty**: Penalty for shorter translations
  - **length_ratio**: Ratio of translation to reference length
  - **translation_length**: Length of generated text
  - **reference_length**: Length of reference text
- **total_samples**: Total number of samples evaluated

#### Individual Score Object Structure

Used in `individual_scores`, `best_predictions`, and `worst_predictions` arrays:

```json
{
  "index": "number",
  "prediction": "string",
  "reference": "string", 
  "rouge": {
    "rouge1": "number",
    "rouge2": "number",
    "rougeL": "number",
    "rougeLsum": "number"
  },
  "bleu": {
    "bleu": "number",
    "precisions": [],
    "brevity_penalty": "number",
    "length_ratio": "number",
    "translation_length": "number",
    "reference_length": "number"
  },
  "combined_score": "number",
  "tweet_id": "string"
}
```

#### Statistics Object

- **mean_combined_score**: Average of all combined scores
- **std_combined_score**: Standard deviation of combined scores
- **best_score**: Highest combined score achieved
- **worst_score**: Lowest combined score achieved

---

## Line 5: Reflections and Improvements

**Purpose**: Contains iterative reflections on evaluation results and improved personas.

```json
{
  "reflections": [
    {
      "run_id": "string",
      "iteration": "number",
      "reflection_results": {
        "reflection_on_results": "string",
        "improved_persona": "string"
      }
    }
  ]
}
```

### Fields

- **reflections** (array): Collection of reflection iterations

#### Reflection Object Structure

- **run_id** (string): Identifier linking to the corresponding evaluation run
- **iteration** (number): Sequential number indicating the reflection iteration
- **reflection_results** (object): Contains the outcomes of the reflection process

#### Reflection Results Structure

- **reflection_on_results** (string): Qualitative analysis of imitation performance, typically discussing:
  - What worked well ("Best imitation")
  - What performed poorly ("Poor imitation")
  - Insights into model performance against the persona
  - Links between success/failure and persona definition aspects
- **improved_persona** (string): Refined user persona based on the reflection analysis

---

## Usage Notes

1. **Sequential Processing**: The lines represent a pipeline where each step builds on the previous ones
2. **Evaluation Metrics**: ROUGE and BLEU scores provide quantitative measures of text similarity and quality
3. **Iterative Improvement**: The reflection process allows for continuous persona refinement
4. **Data Separation**: History and holdset provide proper train/test separation for evaluation

## Example Workflow

1. Load user history and holdset data (Lines 1-2)
2. Generate imitations using initial persona (Line 3)
3. Evaluate imitation quality using multiple metrics (Line 4)
4. Reflect on results and improve persona (Line 5)
5. Repeat process with improved persona for iterative enhancement