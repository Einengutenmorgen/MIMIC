# OpenAI Batch API Text Masking Setup

This guide will help you set up and use the OpenAI Batch API for text masking in your project.

## 🔧 Setup Instructions

### 1. Install Required Packages

```bash
# Upgrade OpenAI library to latest version (required for Batch API)
pip install openai --upgrade

# Install other dependencies
pip install spacy nltk

# Download spaCy model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('opinion_lexicon')"
```

### 2. Set Environment Variables

Create a `.env` file or set environment variables:

```bash
# Your OpenAI API key
export OPENAI_API_KEY="sk-your-api-key-here"
```

Or create a `.env` file:
```
OPENAI_API_KEY=sk-your-api-key-here
```

### 3. Directory Structure

The system will automatically create these directories:
```
your_project/
├── batch_files/          # Temporary batch files
├── batch_results/        # Downloaded results
├── logs/                 # Log files
├── data/
│   └── test_user/       # Your data files
│       └── *.jsonl
└── your_scripts.py
```

## 🚀 Quick Start

### Option 1: Simple Text Masking

```python
from batch_preprocessing_masker import simple_batch_mask

texts = [
    "This movie is absolutely terrible!",
    "The weather is beautiful today.",
    "What a disgusting display of poor sportsmanship.",
]

masked_texts = simple_batch_mask(texts)
for original, masked in zip(texts, masked_texts):
    print(f"Original: {original}")
    print(f"Masked:   {masked}\n")
```

### Option 2: Process Your Data File

```python
from batch_preprocessing_masker import process_user_file_batch
from pathlib import Path

# Process your existing file
file_path = Path("data/test_user/1.347616040797266e+18.jsonl")
process_user_file_batch(file_path, batch_size=1000)
```

### Option 3: Run Examples

```python
# Run the comprehensive examples
python example_batch_usage.py
```

## 📊 Key Benefits of Batch API

- **50% Cost Reduction**: Half the price of regular API calls
- **No Rate Limits**: Process thousands of requests without throttling  
- **24-Hour Guarantee**: Results delivered within 24 hours
- **Separate Quota**: Doesn't affect your real-time API usage
- **Automatic Fallback**: Falls back to deterministic masking if batch fails

## 🔍 How It Works

### 1. Batch Request Format

Each request in the JSONL file looks like this:
```json
{
  "custom_id": "mask-0",
  "method": "POST", 
  "url": "/v1/chat/completions",
  "body": {
    "model": "gpt-4o-mini",
    "messages": [
      {"role": "system", "content": "You are a text masking expert..."},
      {"role": "user", "content": "Mask opinion words in: This is terrible!"}
    ],
    "temperature": 0.1
  }
}
```

### 2. Processing Workflow

1. **Prepare**: Convert your texts to JSONL batch requests
2. **Upload**: Upload JSONL file to OpenAI Files API  
3. **Submit**: Create batch job with uploaded file
4. **Monitor**: Check job status until completion (up to 24h)
5. **Download**: Retrieve results file
6. **Parse**: Extract masked texts and apply to your data

### 3. Error Handling

- **Batch Fails**: Automatically falls back to deterministic masking
- **Invalid Responses**: Uses fallback masking for individual failures
- **Network Issues**: Retries with exponential backoff
- **Partial Results**: Processes completed work, uses fallback for failed items

## 💰 Cost Estimation

| Dataset Size | Requests | Est. Tokens | Est. Cost | Time |
|--------------|----------|-------------|-----------|------|
| Small        | 100      | 5,000       | $0.0008   | < 1h |
| Medium       | 1,000    | 75,000      | $0.0113   | < 2h |  
| Large        | 10,000   | 1,000,000   | $0.1500   | < 12h |
| Very Large   | 50,000   | 6,000,000   | $0.9000   | < 24h |

*Estimates based on gpt-4o-mini batch pricing (50% discount)*

## 🛠️ Configuration Options

### Batch Settings (batch_config.py)
```python
BATCH_CONFIG = {
    "max_batch_size": 1000,      # Requests per batch
    "model": "gpt-4o-mini",      # Model to use  
    "check_interval": 60,        # Status check interval (seconds)
    "temperature": 0.1,          # LLM temperature
    "max_tokens": 1000,          # Max tokens per response
}
```

### Masking Prompt
The system uses a carefully crafted prompt to identify:
- Opinion adjectives (terrible, amazing, awful, brilliant)
- Emotional verbs (love, hate, adore, despise)
- Loaded nouns (hero, villain, genius, idiot)  
- Comparative forms (better, worse, best, worst)
- Subjective adverbs (extremely, absolutely, completely)

## 🔧 Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY not found"**
   - Set your API key: `export OPENAI_API_KEY="sk-..."`
   - Or create `.env` file with your key

2. **"spaCy model not found"**
   - Run: `python -m spacy download en_core_web_sm`

3. **"NLTK opinion_lexicon not found"**
   - Run: `python -c "import nltk; nltk.download('opinion_lexicon')"`

4. **"Batch job failed"**
   - Check your API key validity
   - Verify internet connection
   - Check OpenAI API status
   - System will fall back to deterministic masking

5. **"Invalid JSONL format"**
   - Check for malformed JSON in your batch file
   - Ensure each line is valid JSON
   - Remove empty lines from JSONL files

### Environment Check
```python
from batch_config import check_environment
check_environment()  # Will show any configuration issues
```

## 📁 File Structure

### Input File Format
Your JSONL file should have this structure:
```
Line 1: Metadata (optional)
Line 2: {"tweets": [{"full_text": "...", "tweet_id": "..."}, ...]}
Line 3+: Additional data (optional)
```

### Output
The system adds `"masked_text"` fields to each tweet:
```json
{
  "tweets": [
    {
      "full_text": "This movie is absolutely terrible!",
      "masked_text": "This movie is [MASKED] [MASKED]!",
      "tweet_id": "123456789"
    }
  ]
}
```

## 🎯 Usage Tips

1. **Start Small**: Test with a few hundred texts first
2. **Monitor Costs**: Use cost estimation before large batches
3. **Batch Size**: 500-2000 requests per batch works well
4. **Peak Times**: Batch jobs may complete faster during off-peak hours
5. **Cleanup**: The system automatically cleans up temporary files
6. **Monitoring**: Batch jobs can take minutes to hours, be patient
7. **Fallback**: Always have deterministic masking as backup

## 🔗 Resources

- [OpenAI Batch API Documentation](https://platform.openai.com/docs/guides/batch)
- [OpenAI API Pricing](https://openai.com/pricing)
- [OpenAI Cookbook - Batch Processing](https://cookbook.openai.com/examples/batch_processing)

## 🆘 Support

If you encounter issues:
1. Check this README for troubleshooting steps
2. Run environment check: `python batch_config.py`
3. Test with simple examples first: `python example_batch_usage.py`
4. Check OpenAI API status and your account limits
