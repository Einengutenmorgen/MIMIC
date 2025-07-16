# Data Loading and Analysis Framework

A comprehensive Python framework for loading, analyzing, and visualizing Twitter persona imitation experiment data.

## Overview

This framework provides a complete solution for analyzing run progress data from Twitter persona imitation experiments. It includes classes for data loading, performance analysis, persona evolution tracking, and comprehensive reporting.

## Features

### Core Functionality
- **Data Loading**: Load and parse JSONL files with robust error handling and caching
- **Performance Analysis**: Calculate and aggregate evaluation metrics (ROUGE, BLEU, combined scores)
- **Persona Analysis**: Track persona evolution across iterations and runs
- **Run Comparison**: Compare performance across different runs
- **Data Validation**: Validate data integrity and identify issues
- **Export Capabilities**: Export data to JSON, CSV, and visualization formats

### Key Components

#### 1. Data Structures (`data_framework.py`)
- `Tweet`: Represents individual tweets with metadata
- `Imitation`: Represents stimulus-response pairs
- `EvaluationResult`: Individual evaluation results with scores
- `RunEvaluation`: Complete evaluation data for a run
- `Run`: Run data with persona and imitations
- `UserData`: Complete user data container

#### 2. Core Classes
- `DataLoader`: Handles JSONL file loading with caching
- `RunAnalyzer`: Main analysis class for run-specific operations
- `EvaluationMetrics`: Metric calculation and aggregation
- `PersonaAnalyzer`: Persona evolution and comparison analysis

#### 3. Utility Functions (`data_utils.py`)
- File discovery and run ID extraction
- Data filtering and sorting
- Performance ranking and comparison
- Data validation and quality checks
- Export and reporting functions

## Installation

The framework requires the following dependencies:

```bash
pip install pandas numpy matplotlib seaborn
```

Ensure you have the existing project files:
- `logging_config.py` - for logging configuration
- `file_cache.py` - for caching functionality

## Usage

### Basic Usage

```python
from data_framework import RunAnalyzer
from data_utils import get_available_run_ids

# Initialize analyzer
analyzer = RunAnalyzer("data/users")

# Get available runs
available_runs = get_available_run_ids("data/users")
print(f"Available runs: {available_runs}")

# Analyze specific run
run_id = "20250702_114023"
stats = analyzer.get_run_statistics(run_id)
print(f"Total users: {stats['total_users']}")
print(f"Average scores: {stats['average_scores']}")
```

### User Performance Analysis

```python
# Analyze individual user performance
user_analysis = analyzer.analyze_user_performance("534023.0", run_id)
print(f"User persona: {user_analysis['persona'][:200]}...")
print(f"Performance metrics: {user_analysis['performance_metrics']}")
```

### Performance Ranking

```python
from data_utils import aggregate_scores_by_user, get_top_bottom_performers

# Load user data and aggregate scores
user_data = analyzer.load_run_data(run_id)
user_scores = aggregate_scores_by_user(user_data, run_id)

# Get top and bottom performers
top_performers, bottom_performers = get_top_bottom_performers(user_scores)
print(f"Top performer: {top_performers[0]}")
```

### Persona Analysis

```python
from data_framework import PersonaAnalyzer

persona_analyzer = PersonaAnalyzer()

# Analyze persona evolution
user_runs = [run for run in user_data['534023.0'].runs if run.run_id == run_id]
evolution = persona_analyzer.analyze_persona_evolution(user_runs)
print(f"Persona evolution: {evolution}")
```

### Data Export

```python
# Export complete run data
export_file = analyzer.export_run_data(run_id, 'json')
print(f"Data exported to: {export_file}")

# Export performance data to CSV
from data_utils import export_to_csv
export_to_csv(user_data, run_id, "performance_data.csv")
```

### Comprehensive Reporting

```python
from data_utils import generate_data_report

# Generate comprehensive report
report = generate_data_report("data/users", run_id)
print(f"Report generated for {report['data_overview']['total_users']} users")
```

## Data Structure

The framework expects JSONL files with the following structure:

```
Line 0: Historical tweets data
{
  "user_id": "534023.0",
  "set": "history",
  "tweets": [...]
}

Line 1: Holdout tweets data
{
  "user_id": "534023.0",
  "set": "holdout", 
  "tweets": [...]
}

Line 2: Runs data
{
  "user_id": "534023.0",
  "runs": [
    {
      "run_id": "20250702_114023",
      "persona": "...",
      "imitations": [...]
    }
  ]
}

Line 3: Evaluations data
{
  "user_id": "534023.0",
  "evaluations": [
    {
      "run_id": "20250702_114023",
      "evaluation_results": {
        "overall": {"bleu": 0.5, "rouge_1": 0.6, ...},
        "individual_results": [...],
        "best_predictions": [...],
        "worst_predictions": [...]
      }
    }
  ]
}

Line 4: Reflections data (optional)
{
  "user_id": "534023.0",
  "reflections": [...]
}
```

## API Reference

### RunAnalyzer

Main class for run analysis operations.

#### Methods

- `__init__(data_dir: str)`: Initialize with data directory
- `find_files_with_run_id(run_id: str) -> List[str]`: Find files containing run ID
- `load_run_data(run_id: str) -> Dict[str, UserData]`: Load all user data for run
- `get_run_statistics(run_id: str) -> Dict[str, Any]`: Get comprehensive run statistics
- `analyze_user_performance(user_id: str, run_id: str) -> Dict[str, Any]`: Analyze individual user
- `compare_runs(run_id1: str, run_id2: str) -> Dict[str, Any]`: Compare two runs
- `export_run_data(run_id: str, format: str) -> str`: Export run data

### EvaluationMetrics

Handles metric calculations and aggregations.

#### Methods

- `calculate_combined_score(bleu, rouge_1, rouge_2, rouge_l) -> float`: Calculate combined score
- `aggregate_scores(evaluations: List[RunEvaluation]) -> Dict[str, float]`: Aggregate scores
- `get_best_worst_predictions(evaluations, n=5) -> Tuple[List, List]`: Get best/worst predictions

### PersonaAnalyzer

Analyzes persona evolution and characteristics.

#### Methods

- `extract_persona_keywords(persona: str) -> List[str]`: Extract key characteristics
- `analyze_persona_evolution(runs: List[Run]) -> Dict[str, Any]`: Analyze evolution
- `compare_personas(persona1: str, persona2: str) -> Dict[str, Any]`: Compare personas

### Utility Functions

#### Data Discovery
- `find_run_files(data_dir: str, run_id: str) -> List[str]`: Find files with run ID
- `get_available_run_ids(data_dir: str) -> List[str]`: Get all available run IDs
- `get_user_files(data_dir: str) -> List[str]`: Get all user files

#### Data Filtering
- `filter_tweets_by_date(tweets, start_date, end_date) -> List[Tweet]`: Filter by date
- `filter_tweets_by_type(tweets, tweet_type) -> List[Tweet]`: Filter by type

#### Performance Analysis
- `aggregate_scores_by_user(user_data, run_id) -> Dict`: Aggregate user scores
- `sort_users_by_performance(user_scores, metric) -> List`: Sort by performance
- `get_top_bottom_performers(user_scores, metric, n) -> Tuple`: Get top/bottom performers

#### Reporting
- `create_performance_summary(user_data, run_id) -> Dict`: Create performance summary
- `generate_data_report(data_dir, run_id) -> Dict`: Generate comprehensive report
- `validate_data_integrity(user_data) -> Dict`: Validate data integrity

## Demo Script

Run the demo script to see the framework in action:

```bash
python demo_analysis.py
```

The demo script demonstrates:
- Basic usage patterns
- User performance analysis
- Performance ranking
- Data export functionality
- Persona analysis
- Comprehensive reporting
- Basic visualization

## Examples

### Example 1: Find Best Performing Users

```python
from data_framework import RunAnalyzer
from data_utils import aggregate_scores_by_user, sort_users_by_performance

analyzer = RunAnalyzer("data/users")
user_data = analyzer.load_run_data("20250702_114023")
user_scores = aggregate_scores_by_user(user_data, "20250702_114023")

# Sort by combined score
ranked_users = sort_users_by_performance(user_scores, 'combined', reverse=True)
print(f"Best performer: {ranked_users[0]}")
```

### Example 2: Compare Runs

```python
comparison = analyzer.compare_runs("20250702_114023", "20250708_125947")
print(f"Score differences: {comparison['score_differences']}")
```

### Example 3: Export Data for Visualization

```python
# Export performance data
export_to_csv(user_data, "20250702_114023", "performance.csv")

# Load in pandas for analysis
import pandas as pd
df = pd.read_csv("performance.csv")
print(df.describe())
```

## Error Handling

The framework includes comprehensive error handling:

- **File Not Found**: Graceful handling of missing files
- **JSON Parsing Errors**: Skip malformed lines with logging
- **Data Validation**: Identify and report data integrity issues
- **Missing Data**: Handle incomplete user data gracefully

## Performance Considerations

- **Caching**: Uses file caching to avoid repeated reads
- **Lazy Loading**: Loads data only when needed
- **Memory Efficient**: Processes data in chunks where possible
- **Parallel Processing**: Ready for parallel processing extensions

## Contributing

To extend the framework:

1. Add new data structures to `data_framework.py`
2. Implement new analysis functions in `data_utils.py`
3. Add visualization functions to `demo_analysis.py`
4. Update documentation and examples

## License

This framework is part of the Twitter persona imitation experiment project.