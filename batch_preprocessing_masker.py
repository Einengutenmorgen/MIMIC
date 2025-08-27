# batch_preprocessing_masker.py
import json
import time
import argparse
import sys
from pathlib import Path
import spacy
import nltk
from nltk.corpus import opinion_lexicon
from openai import OpenAI
from logging_config import logger
from typing import Union, List, Dict, Optional
import re
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# --- Configuration ---
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.error("spaCy model 'en_core_web_sm' not found. Run 'python -m spacy download en_core_web_sm'")
    nlp = None

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

HIGH_PRIORITY_LEXICON = {
    "stupid", "idiot", "moron", "clown", "liar", "scumbag", "traitor", "criminal",
    "disgraceful", "shameful", "lunatic", "awful", "terrible", "horrible", "worst",
    "pos", "warrior", "hero", "villain", "disgusting", "pathetic", "failed", "conned"
}

try:
    NLTK_LEXICON = set(opinion_lexicon.words())
    logger.info(f"NLTK opinion lexicon loaded with {len(NLTK_LEXICON)} words.")
except LookupError:
    logger.error("NLTK opinion_lexicon not found. Run: import nltk; nltk.download('opinion_lexicon')")
    NLTK_LEXICON = set()

URL_REGEX = re.compile(r"^\s*https?://[^\s]+\s*$")

# --- Batch Processing Functions ---

def create_batch_requests(texts: List[str], custom_id_prefix: str = "mask") -> List[Dict]:
    """
    Create batch requests for OpenAI API in the required JSONL format.
    Each request asks the LLM to identify opinion-bearing words to mask.
    """
    batch_requests = []
    
    for i, text in enumerate(texts):
        if not text or URL_REGEX.match(text):
            # Skip empty texts or pure URLs
            continue
            
        request = {
            "custom_id": f"{custom_id_prefix}-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",  # Cost-effective model for batch processing
                "messages": [
                    {
                        "role": "system",
                        "content": """You are a text masking expert. Your task is to identify ALL opinion-bearing, subjective, or emotionally charged words in the given text that should be masked to remove bias or sentiment.

Instructions:
- Identify adjectives, adverbs, verbs, and nouns that express opinions, emotions, judgments, or subjective viewpoints
- Focus on words that could influence the reader's perception or contain bias
- Include both positive and negative sentiment words
- Respond ONLY with the original text where opinion-bearing words are replaced with [MASKED]
- Preserve the exact structure and spacing of the original text
- If no opinion-bearing words are found, return the original text unchanged"""
                    },
                    {
                        "role": "user", 
                        "content": f"Mask all opinion-bearing words in this text:\n\n{text}"
                    }
                ],
                "temperature": 0.1,  # Low temperature for consistent results
                "max_tokens": 1000
            }
        }
        batch_requests.append(request)
    
    return batch_requests

def create_batch_file(batch_requests: List[Dict], filename: str) -> str:
    """Create JSONL file for batch processing."""
    filepath = Path(filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for request in batch_requests:
            f.write(json.dumps(request, ensure_ascii=False) + '\n')
    
    logger.info(f"Created batch file: {filepath} with {len(batch_requests)} requests")
    return str(filepath)

def upload_batch_file(filepath: str) -> str:
    """Upload JSONL file to OpenAI Files API."""
    try:
        with open(filepath, "rb") as file:
            batch_file = client.files.create(
                file=file,
                purpose="batch"
            )
        logger.info(f"Uploaded batch file: {batch_file.id}")
        return batch_file.id
    except Exception as e:
        logger.error(f"Failed to upload batch file: {e}")
        raise

def create_batch_job(file_id: str) -> str:
    """Create batch job with uploaded file."""
    try:
        batch_job = client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        logger.info(f"Created batch job: {batch_job.id}")
        return batch_job.id
    except Exception as e:
        logger.error(f"Failed to create batch job: {e}")
        raise

def monitor_batch_job(batch_job_id: str, check_interval: int = 60) -> Dict:
    """Monitor batch job until completion."""
    logger.info(f"Monitoring batch job: {batch_job_id}")
    
    while True:
        try:
            batch_job = client.batches.retrieve(batch_job_id)
            status = batch_job.status
            
            logger.info(f"Batch job status: {status}")
            
            if status == "completed":
                logger.info("Batch job completed successfully!")
                return {
                    "status": status,
                    "output_file_id": batch_job.output_file_id,
                    "error_file_id": batch_job.error_file_id
                }
            elif status in ["failed", "expired", "cancelled"]:
                logger.error(f"Batch job failed with status: {status}")
                return {
                    "status": status,
                    "error_file_id": batch_job.error_file_id if hasattr(batch_job, 'error_file_id') else None
                }
            elif status in ["validating", "in_progress"]:
                logger.info(f"Batch job {status}. Checking again in {check_interval} seconds...")
                time.sleep(check_interval)
            else:
                logger.warning(f"Unknown batch job status: {status}")
                time.sleep(check_interval)
                
        except Exception as e:
            logger.error(f"Error monitoring batch job: {e}")
            time.sleep(check_interval)

def download_batch_results(output_file_id: str, save_path: str = None) -> List[Dict]:
    """Download and parse batch job results."""
    try:
        # Download the results file
        result_content = client.files.content(output_file_id)
        result_text = result_content.content.decode('utf-8')
        
        # Save raw results if path provided
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(result_text)
            logger.info(f"Saved raw results to: {save_path}")
        
        # Parse JSONL results
        results = []
        for line in result_text.strip().split('\n'):
            if line:
                results.append(json.loads(line))
        
        logger.info(f"Downloaded {len(results)} batch results")
        return results
        
    except Exception as e:
        logger.error(f"Failed to download batch results: {e}")
        raise

def parse_batch_results(results: List[Dict]) -> Dict[str, str]:
    """Parse batch results and extract masked texts."""
    masked_texts = {}
    
    for result in results:
        custom_id = result.get('custom_id', '')
        
        if result.get('response', {}).get('status_code') == 200:
            # Successful response
            content = result['response']['body']['choices'][0]['message']['content']
            masked_texts[custom_id] = content.strip()
        else:
            # Error response
            error = result.get('response', {}).get('body', {}).get('error', {})
            logger.error(f"Error in batch result {custom_id}: {error}")
            masked_texts[custom_id] = "[MASKED]"  # Fallback
    
    return masked_texts

# --- Deterministic masking (fallback) ---
def deterministic_mask(text: str) -> str:
    """Deterministic masking using spaCy and NLTK as fallback."""
    if not text or not nlp:
        return "[MASKED]"
    if URL_REGEX.match(text):
        return "[MASKED]"
    
    doc = nlp(text)
    words = [token.text for token in doc]
    
    # Apply deterministic rules
    word_types_to_check = {"ADJ", "ADV", "VERB", "NOUN"}
    for token in doc:
        if token.lower_ in HIGH_PRIORITY_LEXICON:
            words[token.i] = "[MASKED]"
        elif token.pos_ in word_types_to_check and token.lower_ in NLTK_LEXICON:
            words[token.i] = "[MASKED]"
    
    return " ".join(words)

# --- Main batch processing function ---
def process_user_file_batch(file_path: Union[str, Path], batch_size: int = 1000):
    """
    Process user file using batch API for masking.
    
    Args:
        file_path: Path to the JSONL file to process
        batch_size: Number of texts to process in each batch (max 50,000)
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return
    
    logger.info(f"Starting batch processing for: {file_path.name}")
    
    # Read file
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if len(lines) < 2:
        logger.warning(f"File {file_path.name} has less than 2 lines. Skipping.")
        return
    
    try:
        holdout_data = json.loads(lines[1])
        if "tweets" not in holdout_data:
            logger.warning(f"'tweets' not found in line 2 of {file_path.name}")
            return
        
        tweets = holdout_data["tweets"]
        texts_to_mask = []
        tweet_indices = []
        
        # Collect texts that need masking
        for i, tweet in enumerate(tweets):
            if "full_text" in tweet and "masked_text" not in tweet:
                texts_to_mask.append(tweet["full_text"])
                tweet_indices.append(i)
        
        if not texts_to_mask:
            logger.info("No texts to mask found")
            return
        
        logger.info(f"Found {len(texts_to_mask)} texts to mask")
        
        # Process in batches
        all_masked_results = {}
        
        for batch_start in range(0, len(texts_to_mask), batch_size):
            batch_end = min(batch_start + batch_size, len(texts_to_mask))
            batch_texts = texts_to_mask[batch_start:batch_end]
            batch_prefix = f"batch_{batch_start}_{batch_end}"
            
            logger.info(f"Processing batch {batch_start}-{batch_end}")
            
            try:
                # Create batch requests
                batch_requests = create_batch_requests(batch_texts, batch_prefix)
                
                if not batch_requests:
                    logger.warning(f"No valid requests in batch {batch_start}-{batch_end}")
                    continue
                
                # Create and upload batch file
                batch_filename = f"batch_{file_path.stem}_{batch_start}_{batch_end}.jsonl"
                batch_filepath = create_batch_file(batch_requests, batch_filename)
                
                # Upload file and create job
                file_id = upload_batch_file(batch_filepath)
                job_id = create_batch_job(file_id)
                
                # Monitor job
                job_result = monitor_batch_job(job_id)
                
                if job_result["status"] == "completed":
                    # Download and parse results
                    results_filename = f"results_{file_path.stem}_{batch_start}_{batch_end}.jsonl"
                    results = download_batch_results(job_result["output_file_id"], results_filename)
                    batch_masked_results = parse_batch_results(results)
                    all_masked_results.update(batch_masked_results)
                    
                    # Clean up temporary files
                    Path(batch_filepath).unlink(missing_ok=True)
                    
                else:
                    logger.error(f"Batch job failed for batch {batch_start}-{batch_end}")
                    # Apply deterministic masking as fallback
                    for i, text in enumerate(batch_texts):
                        custom_id = f"{batch_prefix}-{i}"
                        all_masked_results[custom_id] = deterministic_mask(text)
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_start}-{batch_end}: {e}")
                # Apply deterministic masking as fallback
                for i, text in enumerate(batch_texts):
                    custom_id = f"{batch_prefix}-{i}"
                    all_masked_results[custom_id] = deterministic_mask(text)
        
        # Apply results back to tweets
        text_index = 0
        for tweet_idx in tweet_indices:
            # Find the corresponding result
            masked_text = None
            for custom_id, result in all_masked_results.items():
                if custom_id.endswith(f"-{text_index}"):
                    masked_text = result
                    break
            
            if masked_text:
                tweets[tweet_idx]["masked_text"] = masked_text
            else:
                # Fallback to deterministic masking
                tweets[tweet_idx]["masked_text"] = deterministic_mask(tweets[tweet_idx]["full_text"])
            
            text_index += 1
        
        # Save updated file
        lines[1] = json.dumps(holdout_data, ensure_ascii=False) + '\n'
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        logger.info(f"Batch processing completed for {file_path.name}")
        
    except json.JSONDecodeError:
        logger.error(f"Error parsing JSON in line 2 of {file_path.name}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

# --- Simple batch processing function for testing ---
def simple_batch_mask(texts: List[str]) -> List[str]:
    """
    Simple function to mask a list of texts using batch API.
    Useful for testing or smaller datasets.
    """
    if not texts:
        return []
    
    logger.info(f"Processing {len(texts)} texts with batch API")
    
    try:
        # Create batch requests
        batch_requests = create_batch_requests(texts, "simple_mask")
        
        # Create and upload batch file
        batch_filepath = create_batch_file(batch_requests, "simple_batch.jsonl")
        file_id = upload_batch_file(batch_filepath)
        
        # Create and monitor job
        job_id = create_batch_job(file_id)
        job_result = monitor_batch_job(job_id, check_interval=30)
        
        if job_result["status"] == "completed":
            # Download and parse results
            results = download_batch_results(job_result["output_file_id"])
            masked_results = parse_batch_results(results)
            
            # Sort results by custom_id to maintain order
            sorted_results = []
            for i in range(len(texts)):
                custom_id = f"simple_mask-{i}"
                if custom_id in masked_results:
                    sorted_results.append(masked_results[custom_id])
                else:
                    # Fallback to deterministic masking
                    sorted_results.append(deterministic_mask(texts[i]))
            
            # Clean up
            Path(batch_filepath).unlink(missing_ok=True)
            
            return sorted_results
        else:
            logger.error("Batch job failed, falling back to deterministic masking")
            return [deterministic_mask(text) for text in texts]
            
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        return [deterministic_mask(text) for text in texts]

# --- Directory Processing Functions ---

def create_backup(file_path: Path) -> Optional[Path]:
    """Create a backup of the original file."""
    try:
        from batch_config import CLI_CONFIG
        
        if not CLI_CONFIG["create_backup"]:
            return None
        
        backup_path = file_path.with_suffix(file_path.suffix + CLI_CONFIG["backup_suffix"])
        
        # Only create backup if it doesn't exist
        if not backup_path.exists():
            import shutil
            shutil.copy2(file_path, backup_path)
            logger.info(f"Created backup: {backup_path}")
            return backup_path
    except Exception as e:
        logger.warning(f"Failed to create backup for {file_path}: {e}")
    
    return None

def process_file_with_stats(file_info: Dict) -> Dict:
    """Process a single file and return statistics."""
    file_path = Path(file_info["path"])
    unprocessed_texts = file_info["unprocessed_texts"]
    use_deterministic = file_info.get("use_deterministic", False)
    
    result = {
        "file": str(file_path),
        "success": False,
        "processed_texts": 0,
        "error": None,
        "use_deterministic": use_deterministic
    }
    
    try:
        # Import config here to avoid issues
        from batch_config import CLI_CONFIG
        
        # Create backup
        backup_path = create_backup(file_path)
        
        if use_deterministic:
            # Use deterministic masking for small files
            logger.info(f"Using deterministic masking for {file_path} ({unprocessed_texts} texts)")
            process_user_file_deterministic(file_path)
            result["processed_texts"] = unprocessed_texts
            result["success"] = True
        else:
            # Use batch processing
            logger.info(f"Processing {file_path} with batch API ({unprocessed_texts} texts)")
            process_user_file_batch(file_path, batch_size=CLI_CONFIG["default_batch_size"])
            result["processed_texts"] = unprocessed_texts
            result["success"] = True
        
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        result["error"] = str(e)
    
    return result

def process_user_file_deterministic(file_path: Union[str, Path]):
    """Process file using only deterministic masking (faster for small files)."""
    file_path = Path(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if len(lines) < 2:
        return
    
    holdout_data = json.loads(lines[1])
    if "tweets" not in holdout_data:
        return
    
    for tweet in holdout_data.get("tweets", []):
        if "full_text" in tweet and "masked_text" not in tweet:
            tweet["masked_text"] = deterministic_mask(tweet["full_text"])
    
    lines[1] = json.dumps(holdout_data, ensure_ascii=False) + '\n'
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def process_directory(
    directory_path: Union[str, Path], 
    batch_size: Optional[int] = None,
    dry_run: bool = False,
    max_workers: int = 1
) -> Dict:
    """
    Process all files in a directory using batch API.
    
    Args:
        directory_path: Path to directory containing files to process
        batch_size: Batch size for processing (uses config default if None)
        dry_run: If True, only analyze files without processing
        max_workers: Number of concurrent file processors (1 = sequential)
    
    Returns:
        Dictionary with processing statistics
    """
    directory_path = Path(directory_path)
    
    if not directory_path.exists():
        logger.error(f"Directory not found: {directory_path}")
        return {"error": "Directory not found"}
    
    if not directory_path.is_dir():
        logger.error(f"Path is not a directory: {directory_path}")
        return {"error": "Path is not a directory"}
    
    # Import config here to avoid circular imports
    try:
        from batch_config import CLI_CONFIG, get_processing_stats, print_processing_summary
    except ImportError:
        logger.error("Could not import batch_config. Make sure batch_config.py is available.")
        return {"error": "Configuration import failed"}
    
    if batch_size is None:
        batch_size = CLI_CONFIG["default_batch_size"]
    
    logger.info(f"Scanning directory: {directory_path}")
    stats = get_processing_stats(directory_path)
    
    # Print summary
    print_processing_summary(
        str(directory_path),
        len(stats["files_to_process"]), 
        stats["total_texts"],
        batch_size
    )
    
    # Show file breakdown
    if stats["files_to_process"]:
        print("Files to process:")
        for i, file_info in enumerate(stats["files_to_process"][:10]):  # Show first 10
            use_det = " (deterministic)" if file_info.get("use_deterministic", False) else ""
            print(f"  {i+1}. {Path(file_info['path']).name}: {file_info['unprocessed_texts']} texts{use_det}")
        
        if len(stats["files_to_process"]) > 10:
            print(f"  ... and {len(stats['files_to_process']) - 10} more files")
    
    if stats["files_already_processed"]:
        print(f"\nFiles already processed: {len(stats['files_already_processed'])}")
    
    if stats["files_skipped"]:
        print(f"\nFiles skipped (invalid format): {len(stats['files_skipped'])}")
    
    if stats["errors"]:
        print(f"\nErrors encountered:")
        for error in stats["errors"][:5]:  # Show first 5 errors
            print(f"  - {error}")
    
    if dry_run:
        print(f"\n🔍 DRY RUN - No files were processed")
        return stats
    
    if not stats["files_to_process"]:
        print(f"\n✅ No files need processing")
        return stats
    
    # Ask for confirmation unless in non-interactive mode
    if sys.stdin.isatty():  # Only ask if running interactively
        estimated_cost = len(stats["files_to_process"]) * 0.01  # Rough estimate
        response = input(f"\nEstimated cost: ~${estimated_cost:.2f}. Continue? (y/N): ")
        if response.lower() != 'y':
            print("Processing cancelled")
            return stats
    
    # Process files
    print(f"\n🚀 Processing {len(stats['files_to_process'])} files...")
    
    processing_results = []
    start_time = time.time()
    
    if max_workers == 1:
        # Sequential processing
        for i, file_info in enumerate(stats["files_to_process"], 1):
            print(f"\nProcessing file {i}/{len(stats['files_to_process'])}: {Path(file_info['path']).name}")
            result = process_file_with_stats(file_info)
            processing_results.append(result)
    else:
        # Concurrent processing (be careful not to overwhelm the API)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(process_file_with_stats, file_info): file_info 
                for file_info in stats["files_to_process"]
            }
            
            for i, future in enumerate(as_completed(future_to_file), 1):
                file_info = future_to_file[future]
                try:
                    result = future.result()
                    processing_results.append(result)
                    print(f"Completed {i}/{len(stats['files_to_process'])}: {Path(file_info['path']).name}")
                except Exception as e:
                    logger.error(f"Error processing {file_info['path']}: {e}")
                    processing_results.append({
                        "file": file_info["path"],
                        "success": False,
                        "error": str(e)
                    })
    
    # Summary
    end_time = time.time()
    successful = sum(1 for r in processing_results if r["success"])
    failed = len(processing_results) - successful
    total_processed = sum(r.get("processed_texts", 0) for r in processing_results if r["success"])
    
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Files processed successfully: {successful}")
    print(f"Files failed: {failed}")
    print(f"Total texts processed: {total_processed:,}")
    print(f"Processing time: {end_time - start_time:.1f} seconds")
    
    if failed > 0:
        print(f"\nFailed files:")
        for result in processing_results:
            if not result["success"]:
                print(f"  - {Path(result['file']).name}: {result.get('error', 'Unknown error')}")
    
    print(f"{'='*60}")
    
    # Update stats with results
    stats["processing_results"] = processing_results
    stats["processing_time"] = end_time - start_time
    
    return stats

# --- Command Line Interface ---

def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(
        description="Batch process text files for opinion masking using OpenAI API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single file
  python batch_preprocessing_masker.py -f data/test_user/file.jsonl
  
  # Process entire directory
  python batch_preprocessing_masker.py -d data/test_user/
  
  # Dry run to see what would be processed
  python batch_preprocessing_masker.py -d data/test_user/ --dry-run
  
  # Process with custom batch size
  python batch_preprocessing_masker.py -d data/test_user/ --batch-size 2000
  
  # Process multiple files in parallel (be careful with API limits)
  python batch_preprocessing_masker.py -d data/test_user/ --max-workers 2
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-f", "--file", 
        type=str,
        help="Process a single file"
    )
    input_group.add_argument(
        "-d", "--directory",
        type=str, 
        help="Process all files in directory"
    )
    
    # Processing options
    parser.add_argument(
        "--batch-size",
        type=int,
        help=f"Batch size for API calls (default: 100)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze files without processing them"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum number of files to process concurrently (default: 1)"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating backup files"
    )
    parser.add_argument(
        "--force-batch",
        action="store_true",
        help="Use batch API even for small files"
    )
    
    # Utility options
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Check environment configuration and exit"
    )
    parser.add_argument(
        "--estimate-cost",
        action="store_true",
        help="Only estimate processing cost and exit"
    )
    
    args = parser.parse_args()
    
    # Import config (do this after argument parsing to avoid import errors in help)
    try:
        from batch_config import check_environment, CLI_CONFIG
    except ImportError as e:
        print(f"❌ Error importing configuration: {e}")
        sys.exit(1)
    
    # Check environment if requested
    if args.check_env:
        if check_environment():
            print("✅ Environment is ready for batch processing!")
            sys.exit(0)
        else:
            print("❌ Environment has issues. Please fix before processing.")
            sys.exit(1)
    
    # Check environment automatically
    if not check_environment():
        print("❌ Environment check failed. Use --check-env for details.")
        sys.exit(1)
    
    # Apply command line overrides to config
    if args.no_backup:
        CLI_CONFIG["create_backup"] = False
    
    if args.force_batch:
        CLI_CONFIG["min_batch_threshold"] = 1
    
    batch_size = args.batch_size or CLI_CONFIG["default_batch_size"]
    
    try:
        if args.file:
            # Process single file
            file_path = Path(args.file)
            if not file_path.exists():
                print(f"❌ File not found: {file_path}")
                sys.exit(1)
            
            print(f"Processing file: {file_path}")
            
            if args.estimate_cost:
                # Estimate cost for single file
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                if len(lines) >= 2:
                    holdout_data = json.loads(lines[1])
                    texts_count = len([t for t in holdout_data.get("tweets", []) 
                                     if "full_text" in t and "masked_text" not in t])
                    from batch_config import estimate_batch_cost
                    cost = estimate_batch_cost(texts_count)
                    print(f"Estimated cost: ${cost:.4f} for {texts_count} texts")
                sys.exit(0)
            
            if args.dry_run:
                print("🔍 DRY RUN - File would be processed")
                sys.exit(0)
            
            # Actually process the file
            process_user_file_batch(file_path, batch_size=batch_size)
            print("✅ File processing complete!")
            
        elif args.directory:
            # Process directory
            directory_path = Path(args.directory)
            if not directory_path.exists():
                print(f"❌ Directory not found: {directory_path}")
                sys.exit(1)
            
            # Process directory
            stats = process_directory(
                directory_path,
                batch_size=batch_size,
                dry_run=args.dry_run or args.estimate_cost,
                max_workers=args.max_workers
            )
            
            if "error" in stats:
                print(f"❌ Error: {stats['error']}")
                sys.exit(1)
            
            if args.estimate_cost:
                from batch_config import estimate_batch_cost
                cost = estimate_batch_cost(stats["total_texts"])
                print(f"\n💰 Estimated total cost: ${cost:.4f}")
                sys.exit(0)
            
            if not args.dry_run:
                success_count = sum(1 for r in stats.get("processing_results", []) if r["success"])
                if success_count > 0:
                    print("✅ Directory processing complete!")
                else:
                    print("❌ No files were successfully processed")
                    sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()