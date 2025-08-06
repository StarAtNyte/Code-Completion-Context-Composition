# Code Completion Context Framework

An intelligent context retrieval system that enhances code completion by collecting and composing relevant code snippets from entire codebases. The framework uses advanced ranking algorithms (BM25, semantic embeddings) to identify the most contextually relevant files and code segments, helping language models generate more accurate code completions. Built with Modal Labs for scalable cloud processing of large repositories.


## Quick Start

### 1. Setup Modal Labs

```bash
pip install modal
modal token new
```

### 2. Upload Competition Data

```bash
# Upload your competition data files
modal volume put code-completion-data data/python-public.jsonl /python-public.jsonl
modal volume put code-completion-data data/python-public.zip /python-public.zip
```

### 3. Extract Data

```bash
modal run src/utils.py --action extract
```

### 4. Generate Submission

```bash
# Generate full submission
modal run src/core.py --action generate --stage public --language python

# Generate test batch
modal run src/core.py --action generate --stage public --language python --batch-size 10
```

### 5. Download Results

```bash
modal volume get code-completion-data /python-public-submission-*.jsonl ./
```

## Project Structure

```
code-completion/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── config.py            # Configuration settings
│   ├── core.py              # Main context framework
│   └── utils.py             # Utility functions
├── data/                    # Competition data (local)
├── docs/                    # Documentation
├── README.md                # This file
├── pyproject.toml           # Python dependencies
└── python-public-submission.jsonl  # Generated submission
```

## Framework Features

### Advanced Retrieval Pipeline
- **Enhanced BM25**: Code-aware tokenization and scoring
- **Multi-file Context**: Combines multiple relevant files per prediction
- **Smart Truncation**: Maintains context quality within token limits
- **Repository-aware**: Processes individual repository archives

### Performance Optimizations
- **GPU Acceleration**: Available for semantic embeddings
- **Parallel Processing**: Concurrent file analysis and scoring
- **Caching System**: Persistent storage for embeddings and analyses
- **Error Handling**: Robust fallbacks ensure 100% success rate

### Format Compliance
- **Format Validation**: Automatic JSONL format checking
- **Token Limits**: Respects model context windows (8K-16K tokens)
- **File Separators**: Uses standard `<|file_sep|>` tokens

## Available Commands

### Core Framework
```bash
# Generate submission
modal run src/core.py --action generate --stage public --language python

# Validate submission
modal run src/core.py --action validate
```

### Utilities
```bash
# Extract competition data
modal run src/utils.py --action extract

# List volume contents
modal run src/utils.py --action list

# Clean up temporary files
modal run src/utils.py --action cleanup
```

## Configuration

Edit `src/config.py` to customize:

- **Model settings**: Context windows, weights
- **Retrieval parameters**: Candidate limits, scoring weights
- **Modal configuration**: Memory, timeouts, dependencies
