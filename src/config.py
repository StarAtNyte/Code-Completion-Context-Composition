"""
Configuration settings for the SOTA Context Collection Framework
"""

# Competition configuration
COMPETITION_CONFIG = {
    "stages": ["practice", "public", "private"],
    "languages": ["python", "kotlin"],
    "models": {
        "mellum": {"context_window": 8192, "weight": 0.33},
        "codestral": {"context_window": 16384, "weight": 0.33}, 
        "qwen": {"context_window": 16384, "weight": 0.34}
    },
    "file_separator": "<|file_sep|>",
    "max_context_chars": {
        "mellum": 32000,    # ~8K tokens
        "codestral": 64000, # ~16K tokens
        "qwen": 64000       # ~16K tokens
    }
}

# Modal configuration
MODAL_CONFIG = {
    "app_name": "code-completion-sota",
    "volume_name": "code-completion-data",
    "image_python_version": "3.11",
    "dependencies": [
        "jsonlines==4.0.0",
        "rank-bm25==0.2.2", 
        "numpy==1.24.0",
        "scikit-learn==1.3.0",
        "tqdm==4.67.1",
        "sentence-transformers>=2.2.2",
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "faiss-cpu>=1.7.4",
    ],
    "system_packages": ["unzip", "git", "curl"],
    "timeouts": {
        "data_processing": 3600,  # 1 hour
        "context_generation": 7200,  # 2 hours
        "validation": 300  # 5 minutes
    },
    "memory": {
        "small": 4096,   # 4GB
        "medium": 16384, # 16GB
        "large": 32768   # 32GB
    }
}

# SOTA Framework configuration
SOTA_CONFIG = {
    "scoring_weights": {
        "semantic": 0.40,
        "structural": 0.25,
        "recency": 0.15,
        "dependency": 0.20
    },
    "retrieval": {
        "max_candidates": 15,
        "max_files_per_repo": 50,
        "min_file_size": 20,
        "max_context_length": 10000
    },
    "bm25": {
        "k1": 1.5,
        "b": 0.75
    }
}