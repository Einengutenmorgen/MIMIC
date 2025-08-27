# batch_config.py - Configuration and utilities for batch processing

import os
import json
from pathlib import Path
from typing import Dict, Any

# --- OpenAI Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

# --- Batch Processing Settings ---
BATCH_CONFIG = {
    # Maximum requests per batch (OpenAI limit is 50,000)
    "max_batch_size": 1000,
    
    # Model to use for batch processing (gpt-4o-mini is cost-effective)
    "model": "gpt-4o-mini",
    
    # Check interval for monitoring batch jobs (seconds)
    "check_interval": 60,
    
    # Temperature for LLM responses (lower = more consistent)
    "temperature": 0.1,
    
    # Maximum tokens per response
    "max_tokens": 1000,
    
    # Completion window (always "24h" for batch API)
    "completion_window": "24h",
}

# --- Command Line Processing Settings ---
CLI_CONFIG = {
    # Default batch size for command line processing
    "default_batch_size": 1000,
    
    # File extensions to process
    "file_extensions": [".jsonl"],
    
    # Whether to process subdirectories recursively
    "recursive": True,
    
    # Whether to backup original files
    "create_backup": True,
    
    # Backup file suffix
    "backup_suffix": ".backup",
    
    # Whether to show progress bars
    "show_progress": True,
    
    # Maximum concurrent batch jobs (to avoid overwhelming the API)
    "max_concurrent_jobs": 3,
    
    # Whether to skip files that already have masked_text fields
    "skip_already_processed": True,
    
    # Minimum number of texts to justify batch processing
    # (files with fewer texts will use deterministic masking)
    "min_batch_threshold": 10,
}

# --- File Paths ---
PATHS = {
    "batch_files": Path("batch_files"),
    "results": Path("batch_results"),
    "logs": Path("logs"),
}

# Create directories if they don't exist
for path in PATHS.values():
    path.mkdir(exist_ok=True)

# --- Masking Prompt Template ---
MASKING_SYSTEM_PROMPT = """You are a text masking expert. Your task is to identify ALL opinion-bearing, subjective, or emotionally charged words in the given text that should be masked to remove bias or sentiment.

Instructions:
- Identify adjectives, adverbs, verbs, and nouns that express opinions, emotions, judgments, or subjective viewpoints
- Focus on words that could influence the reader's perception or contain bias
- Include both positive and negative sentiment words
- Look for words like: awful, terrible, amazing, brilliant, disgusting, wonderful, horrible, fantastic, stupid, genius, etc.
- Also consider comparative and superlative forms: better, best, worse, worst, etc.
- Consider emotional verbs: love, hate, adore, despise, etc.
- Consider loaded nouns: hero, villain, genius, idiot, etc.

Response format:
- Respond ONLY with the original text where opinion-bearing words are replaced with [MASKED]
- Preserve the exact structure, spacing, and punctuation of the original text
- If no opinion-bearing words are found, return the original text unchanged
- Do not add any explanations or additional text"""

MASKING_USER_PROMPT_TEMPLATE = "Mask all opinion-bearing words in this text:\n\n{text}"

# --- Utility Functions ---
def get_batch_filepath(base_name: str, batch_start: int, batch_end: int) -> Path:
    """Generate standardized batch file path."""
    return PATHS["batch_files"] / f"batch_{base_name}_{batch_start}_{batch_end}.jsonl"

def get_results_filepath(base_name: str, batch_start: int, batch_end: int) -> Path:
    """Generate standardized results file path."""
    return PATHS["results"] / f"results_{base_name}_{batch_start}_{batch_end}.jsonl"

def cleanup_batch_files(base_name: str = None):
    """Clean up temporary batch files."""
    pattern = f"batch_{base_name}_*" if base_name else "batch_*"
    for file_path in PATHS["batch_files"].glob(pattern):
        try:
            file_path.unlink()
            print(f"Cleaned up: {file_path}")
        except Exception as e:
            print(f"Failed to clean up {file_path}: {e}")

def estimate_batch_cost(num_requests: int, avg_tokens_per_request: int = 100) -> float:
    """
    Estimate the cost of batch processing.
    
    Args:
        num_requests: Number of requests in the batch
        avg_tokens_per_request: Average tokens per request (input + output)
    
    Returns:
        Estimated cost in USD (approximate, based on gpt-4o-mini pricing)
    """
    # gpt-4o-mini batch pricing (approximate, check current pricing)
    cost_per_1k_tokens = 0.00015  # 50% of regular price
    total_tokens = num_requests * avg_tokens_per_request
    estimated_cost = (total_tokens / 1000) * cost_per_1k_tokens
    return estimated_cost

def print_processing_summary(
    directory: str,
    total_files: int,
    total_texts: int,
    batch_size: int,
    estimated_tokens: int = 100
):
    """Print a summary of the directory processing plan."""
    num_batches = (total_texts + batch_size - 1) // batch_size if total_texts > 0 else 0
    estimated_cost = estimate_batch_cost(total_texts, estimated_tokens)
    
    print(f"\n{'='*60}")
    print(f"DIRECTORY PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Directory: {directory}")
    print(f"Files to process: {total_files}")
    print(f"Total texts to process: {total_texts:,}")
    print(f"Batch size: {batch_size:,}")
    print(f"Number of batches: {num_batches}")
    print(f"Model: {BATCH_CONFIG['model']}")
    print(f"Estimated cost: ${estimated_cost:.4f}")
    print(f"Expected completion time: Within 24 hours")
    print(f"Create backups: {CLI_CONFIG['create_backup']}")
    print(f"Skip processed: {CLI_CONFIG['skip_already_processed']}")
    print(f"{'='*60}\n")

def get_processing_stats(directory_path: Path) -> Dict[str, Any]:
    """Get statistics about files in a directory that need processing."""
    stats = {
        "total_files": 0,
        "files_to_process": [],
        "files_already_processed": [],
        "files_skipped": [],
        "total_texts": 0,
        "total_processed_texts": 0,
        "errors": []
    }
    
    try:
        # Find all matching files
        pattern = "**/*" if CLI_CONFIG["recursive"] else "*"
        all_files = []
        
        for ext in CLI_CONFIG["file_extensions"]:
            all_files.extend(directory_path.glob(f"{pattern}{ext}"))
        
        stats["total_files"] = len(all_files)
        
        for file_path in all_files:
            try:
                # Check if file has the expected format
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                if len(lines) < 2:
                    stats["files_skipped"].append(str(file_path))
                    continue
                
                holdout_data = json.loads(lines[1])
                if "tweets" not in holdout_data:
                    stats["files_skipped"].append(str(file_path))
                    continue
                
                tweets = holdout_data["tweets"]
                unprocessed_texts = 0
                processed_texts = 0
                
                for tweet in tweets:
                    if "full_text" in tweet:
                        if "masked_text" in tweet:
                            processed_texts += 1
                        else:
                            unprocessed_texts += 1
                
                if unprocessed_texts > 0:
                    if unprocessed_texts >= CLI_CONFIG["min_batch_threshold"]:
                        stats["files_to_process"].append({
                            "path": str(file_path),
                            "unprocessed_texts": unprocessed_texts,
                            "processed_texts": processed_texts
                        })
                        stats["total_texts"] += unprocessed_texts
                    else:
                        # Too few texts, will use deterministic masking
                        stats["files_to_process"].append({
                            "path": str(file_path),
                            "unprocessed_texts": unprocessed_texts,
                            "processed_texts": processed_texts,
                            "use_deterministic": True
                        })
                else:
                    stats["files_already_processed"].append(str(file_path))
                
                stats["total_processed_texts"] += processed_texts
                
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                stats["errors"].append(f"{file_path}: {str(e)}")
            except Exception as e:
                stats["errors"].append(f"{file_path}: {str(e)}")
        
    except Exception as e:
        stats["errors"].append(f"Directory scan error: {str(e)}")
    
    return stats

# --- Environment Check ---
def check_environment():
    """Check if the environment is properly configured."""
    issues = []
    
    # Check API key
    if not OPENAI_API_KEY:
        issues.append("OPENAI_API_KEY environment variable is not set")
    
    # Check required packages
    try:
        import openai
        import spacy
        import nltk
    except ImportError as e:
        issues.append(f"Missing required package: {e.name}")
    
    # Check spaCy model
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        issues.append("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
    
    # Check NLTK data
    try:
        from nltk.corpus import opinion_lexicon
        opinion_lexicon.words()
    except LookupError:
        issues.append("NLTK opinion_lexicon not found. Run: import nltk; nltk.download('opinion_lexicon')")
    
    if issues:
        print("❌ Environment Issues Found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ Environment is properly configured!")
        return True

if __name__ == "__main__":
    # Test environment
    check_environment()
    
    # Test cost estimation
    print(f"\nCost estimation examples:")
    print(f"1,000 requests: ${estimate_batch_cost(1000):.4f}")
    print(f"10,000 requests: ${estimate_batch_cost(10000):.4f}")
    print(f"50,000 requests: ${estimate_batch_cost(50000):.4f}")