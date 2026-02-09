# SFT Data Generation - Summary

## What Was Fixed

### Problem Diagnosis
The original code used the REST API with `requests.post()` which had issues:
1. Manual JSON payload construction
2. Manual error handling for HTTP requests
3. Less reliable connection handling

### Solution
Replaced REST API calls with the official **Ollama Python package**:
- Simpler API: `ollama.chat()` instead of manual HTTP requests
- Better error handling built-in
- More reliable connection management
- Cleaner code

### Changes Made

**File: src/sft_data.py**
1. Replaced `import requests` with `import ollama`
2. Removed `OLLAMA_URL` configuration (not needed with Python API)
3. Simplified `ollama_chat()` function to use `ollama.chat()`
4. Added handling for both `{"pairs": [...]}` and direct array JSON responses
5. Improved error messages to be less verbose

## Test Results

### Test 1: First 40 Pages
```bash
python src\sft_data.py data\books\Deep-Learning-Book--MIT-Press--2016.pdf \
  --model llama3.1:8b \
  --pairs-per-paragraph 2 \
  --max-pages 40 \
  --out results\deep-learning-book-sft-test.jsonl
```

**Result**: ✅ Success
- Processed: 32 paragraphs
- Generated: 66 chat examples
- Time: ~2 minutes
- No errors

### Test 2: Full Book (In Progress)
```bash
python src\sft_data.py data\books\Deep-Learning-Book--MIT-Press--2016.pdf \
  --model llama3.1:8b \
  --pairs-per-paragraph 4 \
  --out results\deep-learning-book-sft.jsonl
```

**Status**: Running successfully
- Processed: 287/788 paragraphs (36%)
- Generated: 867 examples so far
- Time: ~31 minutes
- Performance: ~6.5 seconds per paragraph
- Estimated total time: ~85 minutes

## Example Output

Each line in the JSONL file contains a chat example:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Using ONLY the excerpt below, answer the question.\n\nEXCERPT:\n[paragraph text]\n\nQUESTION:\n[question]"
    },
    {
      "role": "assistant",
      "content": "[answer grounded in the excerpt]"
    }
  ]
}
```

## How It Works

1. **Extract PDF Text**: Reads PDF and normalizes text
2. **Split Paragraphs**: Creates 300-1800 character chunks
3. **Generate Q&A Pairs**: Uses Ollama to create questions and answers from each paragraph
4. **Verify Support**: Optional verification pass to check answers are grounded in the text
5. **Save to JSONL**: Each example saved in chat format for fine-tuning

## Usage

### Basic Usage
```bash
python src\sft_data.py path/to/book.pdf --model llama3.1:8b
```

### Full Options
```bash
python src\sft_data.py path/to/book.pdf \
  --model llama3.1:8b \
  --pairs-per-paragraph 4 \
  --temperature 0.7 \
  --top-p 0.9 \
  --seed 42 \
  --max-pages 100 \
  --max-paragraphs 50 \
  --out output.jsonl \
  --no-verify
```

### Options
- `--model`: Ollama model to use (default: llama3.1:8b)
- `--pairs-per-paragraph`: Number of Q&A pairs per paragraph (default: 3)
- `--temperature`: Model temperature (default: 0.7)
- `--top-p`: Top-p sampling (default: 0.9)
- `--seed`: Random seed for reproducibility
- `--max-pages`: Limit number of PDF pages to process
- `--max-paragraphs`: Limit number of paragraphs to process
- `--out`: Output JSONL file path
- `--no-verify`: Disable answer verification pass (faster but less accurate)

## Available Ollama Models

Models detected on your system:
- llama3.1:8b ✅ (recommended, used in tests)
- llama3.1:70B
- mistral:7b
- qwen3:8b
- qwen3:30b
- gemma3:4b
- gemma3:12b
- gemma3:27b
- ministral-3:8b
- ministral-3:14b
- deepseek-r1:32b
- gpt-oss:20b

## Performance Estimates

With llama3.1:8b:
- **Small books** (~100 pages): 10-20 minutes
- **Medium books** (~300 pages): 30-60 minutes
- **Large books** (~600 pages): 60-120 minutes

Factors affecting speed:
- Paragraphs per page
- Pairs per paragraph
- Model size (larger = slower)
- Verification enabled/disabled

## Next Steps

To use this data for fine-tuning:
1. Wait for full generation to complete
2. Review generated examples for quality
3. Use the JSONL file with your fine-tuning pipeline
4. The format is compatible with OpenAI fine-tuning format

## Files Modified

1. **src/sft_data.py**: Main script
   - Replaced REST API with Ollama Python package
   - Improved error handling
   - Added support for both JSON response formats

2. **pyproject.toml**: Dependencies
   - Added `ollama>=0.6.1` package

3. **test_ollama.py**: New test script
   - Tests Ollama connection
   - Lists available models
   - Verifies chat functionality
