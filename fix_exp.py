import os
import json
import copy
from pathlib import Path
from typing import Dict, List, Any, Set
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def detect_metrics_used(evaluation_data: Dict) -> Set[str]:
    """
    Detects which metrics are actually present in evaluation data.
    
    Args:
        evaluation_data: The evaluation data dictionary
        
    Returns:
        Set of metric names found in the data
    """
    metrics_found = set()
    
    if 'evaluations' in evaluation_data:
        for evaluation in evaluation_data['evaluations']:
            if 'evaluation_results' in evaluation:
                eval_results = evaluation['evaluation_results']
                
                # Check overall metrics
                if 'overall' in eval_results:
                    overall = eval_results['overall']
                    if 'rouge' in overall and overall['rouge']:
                        metrics_found.add('rouge')
                    if 'bleu' in overall and overall['bleu']:
                        metrics_found.add('bleu')
                    if 'perplexity' in overall:
                        metrics_found.add('perplexity')
                    if 'bertscore' in overall:
                        metrics_found.add('bertscore')
                
                # Also check individual scores for additional confirmation
                if 'individual_scores' in eval_results and eval_results['individual_scores']:
                    sample = eval_results['individual_scores'][0]
                    if 'rouge' in sample and sample['rouge']:
                        metrics_found.add('rouge')
                    if 'bleu' in sample and sample['bleu']:
                        metrics_found.add('bleu')
                    if 'perplexity' in sample:
                        metrics_found.add('perplexity')
                    if 'bertscore' in sample:
                        metrics_found.add('bertscore')
    
    return metrics_found

def determine_correct_run_id(original_run_id: str, metrics_used: Set[str]) -> str:
    """
    Determines the correct run_id based on metrics used.
    
    Args:
        original_run_id: The original (possibly incorrect) run_id
        metrics_used: Set of metrics actually used
        
    Returns:
        Corrected run_id
    """
    all_metrics = {'rouge', 'bleu', 'perplexity', 'bertscore'}
    
    # If all metrics are present, keep the original name
    if metrics_used == all_metrics:
        return original_run_id
    
    # Determine the correct suffix based on metrics
    if metrics_used == {'rouge'}:
        return original_run_id.replace('all_metrics', 'rouge_only')
    elif metrics_used == {'bleu'}:
        return original_run_id.replace('all_metrics', 'bleu_only')
    elif metrics_used == {'perplexity'}:
        return original_run_id.replace('all_metrics', 'perplexity_only')
    elif metrics_used == {'rouge', 'bleu'}:
        return original_run_id.replace('all_metrics', 'rouge_bleu')
    elif metrics_used == {'rouge', 'perplexity'}:
        return original_run_id.replace('all_metrics', 'rouge_perplexity')
    elif metrics_used == {'bleu', 'perplexity'}:
        return original_run_id.replace('all_metrics', 'bleu_perplexity')
    else:
        # For any other combination, create a descriptive name
        metrics_str = '_'.join(sorted(metrics_used))
        return original_run_id.replace('all_metrics', metrics_str)

def process_jsonl_file(file_path: Path, target_prefix: str = "run_r50_all_metrics") -> Dict[str, Any]:
    """
    Processes a JSONL file and fixes run IDs based on actual metrics used.
    
    Args:
        file_path: Path to the JSONL file
        target_prefix: The run prefix to look for and fix
        
    Returns:
        Dictionary with statistics about the fixes applied
    """
    logger.info(f"Processing file: {file_path}")
    
    stats = {
        'file_path': str(file_path),
        'total_lines': 0,
        'lines_with_target_runs': 0,
        'runs_processed': 0,
        'corrections_made': {},
        'errors': []
    }
    
    try:
        # Read all lines
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        stats['total_lines'] = len(lines)
        modified_lines = []
        file_modified = False
        
        for line_idx, line in enumerate(lines):
            try:
                data = json.loads(line.strip())
                line_modified = False
                
                # Process Line 4: Evaluation Results (0-indexed, so line 3)
                if line_idx == 3 and 'evaluations' in data:
                    logger.info(f"Processing evaluation results (line {line_idx + 1})")
                    
                    for eval_idx, evaluation in enumerate(data['evaluations']):
                        if 'run_id' in evaluation and target_prefix in evaluation['run_id']:
                            stats['lines_with_target_runs'] += 1
                            original_run_id = evaluation['run_id']
                            
                            # Detect metrics used
                            metrics_used = detect_metrics_used({'evaluations': [evaluation]})
                            logger.info(f"Run {original_run_id}: detected metrics = {metrics_used}")
                            
                            # Determine correct run_id
                            correct_run_id = determine_correct_run_id(original_run_id, metrics_used)
                            
                            if correct_run_id != original_run_id:
                                evaluation['run_id'] = correct_run_id
                                line_modified = True
                                file_modified = True
                                stats['runs_processed'] += 1
                                
                                correction_key = f"{original_run_id} -> {correct_run_id}"
                                if correction_key not in stats['corrections_made']:
                                    stats['corrections_made'][correction_key] = 0
                                stats['corrections_made'][correction_key] += 1
                                
                                logger.info(f"Corrected: {original_run_id} -> {correct_run_id}")
                            else:
                                logger.info(f"Run {original_run_id} is correctly named (uses all metrics)")
                
                # Process Line 5: Reflections (0-indexed, so line 4)
                elif line_idx == 4 and 'reflections' in data:
                    logger.info(f"Processing reflections (line {line_idx + 1})")
                    
                    for refl_idx, reflection in enumerate(data['reflections']):
                        if 'run_id' in reflection and target_prefix in reflection['run_id']:
                            original_run_id = reflection['run_id']
                            
                            # For reflections, we need to match with the corrected evaluation run_ids
                            # This requires checking the evaluation results from line 4
                            if len(lines) > 3:  # Ensure line 4 exists
                                try:
                                    eval_data = json.loads(lines[3].strip())
                                    if 'evaluations' in eval_data:
                                        for evaluation in eval_data['evaluations']:
                                            # Find matching evaluation by timestamp or other criteria
                                            eval_run_id = evaluation.get('run_id', '')
                                            if target_prefix in eval_run_id:
                                                metrics_used = detect_metrics_used({'evaluations': [evaluation]})
                                                correct_run_id = determine_correct_run_id(original_run_id, metrics_used)
                                                
                                                if correct_run_id != original_run_id:
                                                    reflection['run_id'] = correct_run_id
                                                    line_modified = True
                                                    file_modified = True
                                                    logger.info(f"Corrected reflection: {original_run_id} -> {correct_run_id}")
                                                break
                                except json.JSONDecodeError:
                                    logger.warning("Could not parse evaluation data for reflection correction")
                
                # Add the (possibly modified) line
                if line_modified:
                    modified_lines.append(json.dumps(data, ensure_ascii=False))
                else:
                    modified_lines.append(line.strip())
                    
            except json.JSONDecodeError as e:
                error_msg = f"JSON decode error in line {line_idx + 1}: {e}"
                logger.error(error_msg)
                stats['errors'].append(error_msg)
                modified_lines.append(line.strip())  # Keep original line if it can't be parsed
        
        # Write back to file if modifications were made
        if file_modified:
            # Create backup first
            backup_path = file_path.with_suffix(file_path.suffix + '.backup')
            logger.info(f"Creating backup: {backup_path}")
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join([line.strip() for line in lines]))
            
            # Write corrected file
            logger.info(f"Writing corrected file: {file_path}")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(modified_lines))
        
    except Exception as e:
        error_msg = f"Error processing file {file_path}: {e}"
        logger.error(error_msg)
        stats['errors'].append(error_msg)
    
    return stats

def process_directory(directory_path: str, target_prefix: str = "run_r50_all_metrics") -> List[Dict[str, Any]]:
    """
    Processes all JSONL files in a directory and fixes run IDs.
    
    Args:
        directory_path: Path to the directory containing JSONL files
        target_prefix: The run prefix to look for and fix
        
    Returns:
        List of statistics dictionaries for each processed file
    """
    dir_path = Path(directory_path)
    
    if not dir_path.exists():
        logger.error(f"Directory does not exist: {directory_path}")
        return []
    
    if not dir_path.is_dir():
        logger.error(f"Path is not a directory: {directory_path}")
        return []
    
    # Find all JSONL files
    jsonl_files = list(dir_path.glob("*.jsonl")) + list(dir_path.glob("*.JSONL"))
    
    if not jsonl_files:
        logger.warning(f"No JSONL files found in directory: {directory_path}")
        return []
    
    logger.info(f"Found {len(jsonl_files)} JSONL files")
    
    all_stats = []
    
    for jsonl_file in jsonl_files:
        file_stats = process_jsonl_file(jsonl_file, target_prefix)
        all_stats.append(file_stats)
        
        # Log summary for this file
        if file_stats['runs_processed'] > 0:
            logger.info(f"File {jsonl_file.name}: {file_stats['runs_processed']} runs corrected")
            for correction, count in file_stats['corrections_made'].items():
                logger.info(f"  {correction} (x{count})")
        else:
            logger.info(f"File {jsonl_file.name}: No corrections needed")
    
    return all_stats

def print_summary(all_stats: List[Dict[str, Any]]):
    """Print a summary of all corrections made."""
    total_files = len(all_stats)
    total_runs_corrected = sum(stats['runs_processed'] for stats in all_stats)
    total_errors = sum(len(stats['errors']) for stats in all_stats)
    
    print("\n" + "="*60)
    print("CORRECTION SUMMARY")
    print("="*60)
    print(f"Files processed: {total_files}")
    print(f"Total runs corrected: {total_runs_corrected}")
    print(f"Total errors: {total_errors}")
    
    if total_runs_corrected > 0:
        print("\nCorrections made:")
        all_corrections = {}
        for stats in all_stats:
            for correction, count in stats['corrections_made'].items():
                if correction in all_corrections:
                    all_corrections[correction] += count
                else:
                    all_corrections[correction] = count
        
        for correction, count in all_corrections.items():
            print(f"  {correction} (x{count})")
    
    if total_errors > 0:
        print("\nErrors encountered:")
        for stats in all_stats:
            for error in stats['errors']:
                print(f"  {error}")

# Main execution
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python experiment_fix.py <directory_path>")
        print("Example: python experiment_fix.py ./experiment_results/")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    target_prefix = "run_r50_all_metrics"
    
    print("="*80)
    print("🔧 EXPERIMENT RUN ID CORRECTION TOOL")
    print("="*80)
    print(f"📂 Processing directory: {directory_path}")
    print(f"🎯 Target prefix: {target_prefix}")
    print(f"🔍 Looking for incorrectly named runs that should be separated by metrics")
    print("-" * 80)
    
    all_stats = process_directory(directory_path, target_prefix)
    print_summary(all_stats)
    
    print("\n" + "="*80)
    print("✅ PROCESSING COMPLETE!")
    print("="*80)
    print("💾 Backups were created for all modified files (.backup extension)")
    print("🔍 Check the detailed output above for all corrections made")
    print("🔄 You can restore original files using: mv file.jsonl.backup file.jsonl")
    print("="*80)