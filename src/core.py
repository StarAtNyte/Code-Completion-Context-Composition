"""
Core SOTA Context Collection Framework
Main implementation of the state-of-the-art context retrieval system
"""

import modal
import os
import json
import jsonlines
import zipfile
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional
import time
from tqdm import tqdm
import numpy as np
import signal

import sys
import os

# Handle imports for both local and Modal environments
try:
    from src.config import COMPETITION_CONFIG, MODAL_CONFIG, SOTA_CONFIG
except ImportError:
    # Try relative import for Modal environment
    try:
        from config import COMPETITION_CONFIG, MODAL_CONFIG, SOTA_CONFIG
    except ImportError:
        # Fallback configuration if imports fail
        COMPETITION_CONFIG = {
            "file_separator": "<|file_sep|>",
            "max_context_chars": {"mellum": 32000, "codestral": 64000, "qwen": 64000}
        }
        MODAL_CONFIG = {
            "app_name": "code-completion-sota",
            "volume_name": "code-completion-data",
            "image_python_version": "3.11",
            "dependencies": [
                "jsonlines==4.0.0", "rank-bm25==0.2.2", "numpy==1.24.0",
                "scikit-learn==1.3.0", "tqdm==4.67.1", "sentence-transformers>=2.2.2",
                "transformers>=4.30.0", "torch>=2.0.0", "faiss-cpu>=1.7.4"
            ],
            "system_packages": ["unzip", "git", "curl"],
            "timeouts": {"data_processing": 3600, "context_generation": 7200, "validation": 300},
            "memory": {"small": 4096, "medium": 16384, "large": 32768}
        }
        SOTA_CONFIG = {
            "scoring_weights": {"semantic": 0.40, "structural": 0.25, "recency": 0.15, "dependency": 0.20},
            "retrieval": {"max_candidates": 15, "max_files_per_repo": 50, "min_file_size": 20, "max_context_length": 10000},
            "bm25": {"k1": 1.5, "b": 0.75}
        }

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modal App Configuration
app = modal.App(MODAL_CONFIG["app_name"])
volume = modal.Volume.from_name(MODAL_CONFIG["volume_name"], create_if_missing=True)

# Production image with optimized dependencies
image = (
    modal.Image.debian_slim(python_version=MODAL_CONFIG["image_python_version"])
    .pip_install(MODAL_CONFIG["dependencies"])
    .apt_install(MODAL_CONFIG["system_packages"])
)

class SOTAContextRetriever:
    """Main SOTA context retrieval system with research-backed enhancements."""
    
    def __init__(self, language: str = "python"):
        self.language = language
        self.config = SOTA_CONFIG
        self.file_separator = COMPETITION_CONFIG["file_separator"]
        
        # Research-backed scoring weights from Framework.md (Hierarchical Reasoning)
        self.primary_weights = {
            "semantic": 0.40,
            "structural": 0.25, 
            "recency": 0.15,
            "dependency": 0.20
        }
        self.scoring_weights = {
            "semantic": 0.40,      # Semantic similarity (primary)
            "structural": 0.25,    # Code structure similarity  
            "recency": 0.15,       # Recent file modifications
            "dependency": 0.20     # Import/dependency relevance
        }
        
        # Multi-factor approach inspired by "mf3" in winning team name
        self.multi_factors = {
            "prefix_driven": 0.4,   # Prefix-focused scoring (pd in team name)
            "hypothetic_lines": 0.3, # Generated completion hypothetics
            "analogy_context": 0.3   # Similar code patterns
        }
        self.scoring_weights = {
            "semantic": 0.40,      # Semantic similarity (primary)
            "structural": 0.25,    # Code structure similarity  
            "recency": 0.15,       # Recent file modifications
            "dependency": 0.20     # Import/dependency relevance
        }
        
        # Multi-factor approach inspired by "mf3" in winning team name
        self.multi_factors = {
            "prefix_driven": 0.4,   # Prefix-focused scoring (pd in team name)
            "hypothetic_lines": 0.3, # Generated completion hypothetics
            "analogy_context": 0.3   # Similar code patterns
        }
        
    def retrieve_context(self, prefix: str, suffix: str, repo_files: List[Path]) -> str:
        """
        Retrieve optimal context using research-backed dual context approach.
        
        Implements findings from:
        - Dual Context Approach (2402.14323v2): AC + RC combination
        - Hypothetic Line Generation (2405.07530v1): Enhanced retrieval queries
        - Hierarchical Reasoning: Multi-factor scoring system
        
        Args:
            prefix: Code before the completion point
            suffix: Code after the completion point
            repo_files: List of repository file paths
            
        Returns:
            Formatted context string ready for submission
        """
        from rank_bm25 import BM25Okapi
        import re
        
        # Early return for empty repo_files to prevent hanging
        if not repo_files:
            return f"{self.file_separator}empty.py\n# No repository files found\npass"
        
        # Limit repo_files to prevent excessive processing
        if len(repo_files) > 50:
            repo_files = repo_files[:50]
        
        try:
            # Step 1: Generate hypothetic completions for enhanced retrieval
            hypothetic_lines = self._generate_hypothetic_completions(prefix, suffix)
            
            # Step 2: Dual context retrieval with timeout protection
            analogy_context = self._retrieve_analogy_context(prefix, suffix, repo_files, hypothetic_lines)
            rationale_context = self._retrieve_rationale_context(prefix, suffix, repo_files)
            
            # Step 3: Intelligent context composition based on research findings
            return self._compose_dual_context(analogy_context, rationale_context, prefix, suffix)
            
        except Exception as e:
            logger.warning(f"Context retrieval failed: {e}, using fallback")
            return f"{self.file_separator}fallback.py\n# Context retrieval failed: {str(e)}\npass"
    
    def _generate_hypothetic_completions(self, prefix: str, suffix: str) -> List[str]:
        """Generate hypothetic completion lines based on prefix/suffix patterns."""
        import re
        
        hypothetics = []
        
        # Pattern 1: Function call completion
        if re.search(r'\w+\($', prefix.strip()):
            func_match = re.search(r'(\w+)\($', prefix.strip())
            if func_match:
                func_name = func_match.group(1)
                hypothetics.extend([
                    f"{func_name}()",
                    f"{func_name}(self)",
                    f"{func_name}(args)",
                    f"def {func_name}(self):"
                ])
        
        # Pattern 2: Variable assignment
        if re.search(r'\w+\s*=$', prefix.strip()):
            var_match = re.search(r'(\w+)\s*=$', prefix.strip())
            if var_match:
                var_name = var_match.group(1)
                hypothetics.extend([
                    f"{var_name} = None",
                    f"{var_name} = []",
                    f"{var_name} = {{}}",
                    f"self.{var_name}"
                ])
        
        # Pattern 3: Import completion
        if 'import' in prefix.lower():
            hypothetics.extend([
                "import os",
                "import sys", 
                "from typing import",
                "import json"
            ])
        
        # Pattern 4: Class/method definition
        if re.search(r'(class|def)\s+\w*$', prefix.strip()):
            hypothetics.extend([
                "def __init__(self):",
                "def __str__(self):",
                "class MyClass:",
                "def process(self):"
            ])
        
        return hypothetics[:10]  # Limit to top 10 hypothetics
    
    def _retrieve_analogy_context(self, prefix: str, suffix: str, repo_files: List[Path], hypothetics: List[str]) -> List[Dict]:
        """Retrieve analogy context: similar code patterns and usage examples."""
        from rank_bm25 import BM25Okapi
        import re
        
        # Enhanced code-aware tokenization
        def tokenize_code(text):
            # Split on whitespace and punctuation but keep identifiers
            tokens = re.findall(r'\w+|[^\w\s]', text.lower())
            
            # Extract identifiers separately for better matching
            identifiers = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', text)
            tokens.extend([id.lower() for id in identifiers])
            
            # Extract function/class names with higher weight
            func_classes = re.findall(r'(?:def|class)\s+([a-zA-Z_][a-zA-Z0-9_]*)', text)
            tokens.extend([name.lower() + '_definition' for name in func_classes])
            
            # Extract import statements
            imports = re.findall(r'(?:import|from)\s+([a-zA-Z_][a-zA-Z0-9_.]*)', text)
            tokens.extend([imp.lower() + '_import' for imp in imports])
            
            return tokens
        
        # Prepare file contents with enhanced tokenization and filtering
        files_content = []
        file_paths = []
        file_raw_content = []
        
        # Pre-filter files by relevance to avoid processing irrelevant files
        relevant_files = self._prefilter_files(repo_files, prefix, suffix)
        
        for file_path in relevant_files[:self.config["retrieval"]["max_files_per_repo"]]:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if len(content.strip()) > self.config["retrieval"]["min_file_size"]:
                        files_content.append(tokenize_code(content))
                        file_paths.append(file_path)
                        file_raw_content.append(content)
            except Exception as e:
                logger.warning(f"Could not read {file_path}: {e}")
        
        if not files_content:
            return self._create_fallback_context("No readable files found")
        
        # Enhanced query with multiple strategies
        prefix_tokens = tokenize_code(prefix)
        suffix_tokens = tokenize_code(suffix)
        
        # Strategy 1: Combined prefix+suffix query
        combined_query = prefix_tokens + suffix_tokens
        
        # Strategy 2: Extract key identifiers and keywords
        key_terms = []
        function_calls = []
        
        for token in prefix_tokens + suffix_tokens:
            if len(token) > 2 and token.isalpha():
                key_terms.append(token)
            # Detect function calls (tokens followed by parentheses in original text)
            if token + '(' in prefix + suffix:
                function_calls.append(token)
        
        # Strategy 3: Extract context from surrounding code structure
        context_patterns = []
        for pattern in ['def ', 'class ', 'import ', 'from ', 'return ', 'if ', 'for ', 'while ']:
            if pattern in prefix.lower() or pattern in suffix.lower():
                context_patterns.append(pattern.strip())
        
        # Use BM25 with enhanced parameters
        bm25 = BM25Okapi(files_content, k1=1.5, b=0.75)
        
        # Multi-strategy scoring
        combined_scores = bm25.get_scores(combined_query)
        
        # Score with key terms (focused identifiers)
        key_scores = bm25.get_scores(key_terms) if key_terms else np.zeros_like(combined_scores)
        
        # Score with function calls (higher weight for function definitions)
        func_scores = bm25.get_scores(function_calls) if function_calls else np.zeros_like(combined_scores)
        
        # Score with context patterns
        pattern_scores = bm25.get_scores(context_patterns) if context_patterns else np.zeros_like(combined_scores)
        
        # Generate hypothetic completions (RAG approach from 2405.07530v1)
        hypothetic_scores = self._score_with_hypothetics(prefix, suffix, files_content)
        
        # Multi-threshold scoring system (inspired by "mt6" - 6 thresholds)
        final_scores = self._apply_multi_threshold_scoring(
            combined_scores, key_scores, func_scores, pattern_scores, 
            prefix, suffix, file_paths, file_raw_content
        )
        
        # Apply hierarchical ranking with research-backed weights
        hierarchical_scores = self._apply_hierarchical_ranking(
            final_scores, file_paths, file_raw_content, prefix, suffix
        )
        
        # Get top files with adaptive selection based on hierarchical scores
        sorted_indices = np.argsort(hierarchical_scores)[::-1]
        
        # Multi-threshold selection inspired by "mt6" in winning team name
        thresholds = self._calculate_adaptive_thresholds(hierarchical_scores)
        top_indices = self._select_with_multiple_thresholds(sorted_indices, hierarchical_scores, thresholds)
        
        # Ensure we have at least one file
        if not top_indices:
            top_indices = sorted_indices[:1]
        
        # Research-backed dual context composition
        return self._compose_dual_context(top_indices, file_paths, file_raw_content, final_scores, prefix, suffix)
    
    def _generate_hypothetic_lines(self, prefix: str, suffix: str) -> List[str]:
        """Generate hypothetic completion lines using heuristic patterns (RAG approach)."""
        hypothetics = []
        
        # Extract patterns from prefix for completion prediction
        import re
        
        # Pattern 1: Function calls - predict return statements
        if 'def ' in prefix and 'return' not in prefix.split('\n')[-3:]:
            # Look for variable assignments that might be returned
            vars_assigned = re.findall(r'(\w+)\s*=', prefix)
            if vars_assigned:
                hypothetics.append(f"return {vars_assigned[-1]}")
        
        # Pattern 2: If statements - predict common conditions
        if prefix.strip().endswith('if '):
            hypothetics.extend([
                "if not ", "if len(", "if isinstance(", "if hasattr("
            ])
        
        # Pattern 3: For loops - predict iteration patterns  
        if 'for ' in prefix.split('\n')[-1]:
            hypothetics.extend([
                "for item in items:", "for i, item in enumerate(", "for key, value in "
            ])
        
        # Pattern 4: Import statements
        if prefix.strip().endswith('import ') or prefix.strip().endswith('from '):
            common_imports = ['os', 'sys', 'json', 'logging', 'pathlib', 'typing']
            hypothetics.extend(common_imports)
        
        # Pattern 5: Class methods - predict common method patterns
        if 'class ' in prefix and 'def ' in prefix.split('\n')[-2:]:
            hypothetics.extend([
                "def __init__(self", "def __str__(self", "def __repr__(self"
            ])
        
        return hypothetics[:5]  # Limit to top 5 hypothetics
    
    def _compose_dual_context(self, indices: List[int], file_paths: List[Path], 
                                file_contents: List[str], scores: np.ndarray, 
                                prefix: str, suffix: str) -> str:
        """Compose dual context using Analogy Context (AC) + Rationale Context (RC) approach."""
        
        # Generate hypothetic lines for enhanced retrieval
        hypothetics = self._generate_hypothetic_lines(prefix, suffix)
        
        # Separate files into Analogy Context (code patterns) and Rationale Context (docs/tests)
        analogy_files = []
        rationale_files = []
        
        for idx in indices:
            if idx >= len(file_paths):
                continue
                
            file_path = file_paths[idx]
            content = file_contents[idx]
            score = scores[idx]
            
            if score <= 0:
                continue
            
            # Classify file type for dual context approach
            filename = file_path.name.lower()
            
            # Rationale Context: documentation, tests, comments
            if any(keyword in filename for keyword in ['test', 'doc', 'readme', 'example', 'demo']):
                rationale_files.append((idx, file_path, content, score))
            # Analogy Context: implementation files with similar patterns
            else:
                analogy_files.append((idx, file_path, content, score))
        
        # Return analogy candidates for dual context composition
        analogy_candidates = []
        for idx, file_path, content, score in analogy_files:
            analogy_candidates.append({
                'file_path': file_path,
                'content': content,
                'score': score,
                'type': 'analogy'
            })
        
        return analogy_candidates[:5]  # Top 5 analogy files
    
    def _build_interleaved_context(self, analogy_files: List, rationale_files: List, 
                                 prefix: str, suffix: str, hypothetics: List[str]) -> str:
        """Build interleaved dual context with research-backed optimization."""
        file_separator = COMPETITION_CONFIG["file_separator"]
        context_parts = []
        total_length = 0
        max_length = min(self.config["retrieval"]["max_context_length"], 6000)
        
        # Sort files by score within each category
        analogy_files.sort(key=lambda x: x[3], reverse=True)  # Sort by score
        rationale_files.sort(key=lambda x: x[3], reverse=True)
        
        # Interleave Analogy and Rationale contexts (research-backed approach)
        # Start with Analogy Context (more important for code completion)
        analogy_count = 0
        rationale_count = 0
        
        while (analogy_count < len(analogy_files) or rationale_count < len(rationale_files)) and total_length < max_length * 0.9:
            
            # Add Analogy Context (2:1 ratio based on research)
            if analogy_count < len(analogy_files) and analogy_count < 3:
                idx, file_path, content, score = analogy_files[analogy_count]
                optimized_content = self._optimize_file_content(content, prefix, suffix)
                
                # Enhance with hypothetic line matching
                enhanced_content = self._enhance_with_hypothetics(optimized_content, hypothetics)
                
                context_part = f"{file_separator}{file_path.name}\n{enhanced_content}"
                
                if total_length + len(context_part) < max_length:
                    context_parts.append(context_part)
                    total_length += len(context_part)
                
                analogy_count += 1
            
            # Add Rationale Context (documentation/tests)
            if rationale_count < len(rationale_files) and rationale_count < 1:
                idx, file_path, content, score = rationale_files[rationale_count]
                doc_content = self._extract_documentation(content)
                
                if doc_content:
                    context_part = f"{file_separator}{file_path.name}\n{doc_content}"
                    
                    if total_length + len(context_part) < max_length:
                        context_parts.append(context_part)
                        total_length += len(context_part)
                
                rationale_count += 1
            
            # Break if we've processed available files
            if analogy_count >= len(analogy_files) and rationale_count >= len(rationale_files):
                break
        
        if context_parts:
            return "\n\n".join(context_parts)
        else:
            return self._create_fallback_context("No relevant files found")
    
    def _enhance_with_hypothetics(self, content: str, hypothetics: List[str]) -> str:
        """Enhance content by highlighting sections matching hypothetic completions."""
        if not hypothetics:
            return content
        
        lines = content.split('\n')
        enhanced_lines = []
        
        for line in lines:
            line_lower = line.lower()
            # Boost lines that match hypothetic patterns
            for hyp in hypothetics:
                if hyp.lower() in line_lower:
                    # Add a comment to highlight relevance
                    enhanced_lines.append(f"{line}  # <- Relevant pattern")
                    break
            else:
                enhanced_lines.append(line)
        
        return '\n'.join(enhanced_lines)
    
    def _extract_documentation(self, content: str) -> str:
        """Extract documentation, comments, and docstrings from content."""
        lines = content.split('\n')
        doc_lines = []
        
        in_docstring = False
        docstring_delimiter = None
        
        for line in lines:
            stripped = line.strip()
            
            # Detect docstring start/end
            if '"""' in stripped or "'''" in stripped:
                if not in_docstring:
                    in_docstring = True
                    docstring_delimiter = '"""' if '"""' in stripped else "'''"
                    doc_lines.append(line)
                elif docstring_delimiter in stripped:
                    doc_lines.append(line)
                    in_docstring = False
                else:
                    doc_lines.append(line)
            elif in_docstring:
                doc_lines.append(line)
            # Include comments
            elif stripped.startswith('#'):
                doc_lines.append(line)
            # Include function/class signatures with docstrings
            elif stripped.startswith(('def ', 'class ')) and len(doc_lines) < 20:
                doc_lines.append(line)
        
        return '\n'.join(doc_lines[:30])  # Limit documentation length
    
    def _apply_multi_threshold_scoring(self, combined_scores: np.ndarray, key_scores: np.ndarray, 
                                     func_scores: np.ndarray, pattern_scores: np.ndarray,
                                     prefix: str, suffix: str, file_paths: List[Path], 
                                     file_contents: List[str]) -> np.ndarray:
        """Apply multi-threshold scoring with 6 different thresholds (mt6 approach)."""
        
        # Initialize final scores
        final_scores = np.zeros_like(combined_scores)
        
        # Threshold 1: High semantic similarity (top 20%)
        semantic_threshold = np.percentile(combined_scores, 80)
        high_semantic_mask = combined_scores >= semantic_threshold
        final_scores[high_semantic_mask] += combined_scores[high_semantic_mask] * 0.4
        
        # Threshold 2: Strong key term matches (above median)
        key_threshold = np.median(key_scores) if len(key_scores) > 0 else 0
        strong_key_mask = key_scores >= key_threshold
        final_scores[strong_key_mask] += key_scores[strong_key_mask] * 0.25
        
        # Threshold 3: Function definition matches (any positive score)
        func_threshold = 0.1
        func_match_mask = func_scores >= func_threshold
        final_scores[func_match_mask] += func_scores[func_match_mask] * 0.2
        
        # Threshold 4: Pattern similarity (above average)
        pattern_threshold = np.mean(pattern_scores) if len(pattern_scores) > 0 else 0
        pattern_mask = pattern_scores >= pattern_threshold
        final_scores[pattern_mask] += pattern_scores[pattern_mask] * 0.1
        
        # Threshold 5: Prefix-driven scoring (files containing prefix patterns)
        prefix_boost = self._calculate_prefix_driven_scores(prefix, file_paths, file_contents)
        prefix_threshold = np.percentile(prefix_boost, 70)
        prefix_mask = prefix_boost >= prefix_threshold
        final_scores[prefix_mask] += prefix_boost[prefix_mask] * 0.3
        
        # Threshold 6: Dependency relevance (import relationships)
        dependency_boost = self._calculate_dependency_scores(prefix, suffix, file_paths, file_contents)
        dep_threshold = np.percentile(dependency_boost, 60)
        dep_mask = dependency_boost >= dep_threshold
        final_scores[dep_mask] += dependency_boost[dep_mask] * 0.15
        
        return final_scores
    
    def _calculate_prefix_driven_scores(self, prefix: str, file_paths: List[Path], 
                                      file_contents: List[str]) -> np.ndarray:
        """Calculate prefix-driven scores focusing on completion context."""
        scores = np.zeros(len(file_paths))
        
        # Extract the immediate context around completion point
        prefix_lines = prefix.split('\n')
        current_line = prefix_lines[-1] if prefix_lines else ""
        context_lines = prefix_lines[-3:] if len(prefix_lines) >= 3 else prefix_lines
        
        import re
        
        for i, content in enumerate(file_contents):
            score = 0
            content_lower = content.lower()
            
            # Boost files containing similar line patterns
            for context_line in context_lines:
                if context_line.strip() and context_line.strip().lower() in content_lower:
                    score += 2
            
            # Boost files with similar indentation patterns
            current_indent = len(current_line) - len(current_line.lstrip())
            content_lines = content.split('\n')
            for line in content_lines:
                if line.strip():
                    line_indent = len(line) - len(line.lstrip())
                    if abs(line_indent - current_indent) <= 2:  # Similar indentation
                        score += 0.1
            
            # Boost files with similar variable/function naming patterns
            prefix_identifiers = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', current_line)
            for identifier in prefix_identifiers:
                if len(identifier) > 2:
                    score += content_lower.count(identifier.lower()) * 0.5
            
            scores[i] = score
        
        return scores
    
    def _calculate_dependency_scores(self, prefix: str, suffix: str, file_paths: List[Path], 
                                   file_contents: List[str]) -> np.ndarray:
        """Calculate dependency-based relevance scores."""
        scores = np.zeros(len(file_paths))
        
        # Extract imports from prefix/suffix
        import re
        all_text = prefix + suffix
        imports = re.findall(r'(?:from\s+(\S+)\s+import|import\s+(\S+))', all_text)
        imported_modules = set()
        for imp in imports:
            imported_modules.update([m for m in imp if m])
        
        for i, (file_path, content) in enumerate(zip(file_paths, file_contents)):
            score = 0
            
            # Boost files that are imported modules
            file_module = file_path.stem
            if file_module in imported_modules:
                score += 5
            
            # Boost files that import similar modules
            file_imports = re.findall(r'(?:from\s+(\S+)\s+import|import\s+(\S+))', content)
            file_modules = set()
            for imp in file_imports:
                file_modules.update([m for m in imp if m])
            
            # Calculate import overlap
            overlap = len(imported_modules.intersection(file_modules))
            score += overlap * 2
            
            # Boost files in same directory (likely related)
            if len(file_paths) > 1:
                current_dir = file_path.parent.name
                for other_path in file_paths:
                    if other_path != file_path and other_path.parent.name == current_dir:
                        score += 0.5
            
            scores[i] = score
        
        return scores
    
    def _optimize_file_content(self, content: str, prefix: str, suffix: str) -> str:
        """Optimize file content by selecting most relevant sections."""
        if len(content) <= 1500:
            return content
        
        lines = content.split('\n')
        
        # Extract key terms from prefix/suffix for relevance scoring
        import re
        key_terms = set()
        function_names = set()
        
        for text in [prefix, suffix]:
            terms = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', text.lower())
            key_terms.update([t for t in terms if len(t) > 2])
            
            # Extract function calls (look for patterns like "func(")
            func_calls = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', text)
            function_names.update([f.lower() for f in func_calls])
        
        # Score lines by relevance with improved scoring
        line_scores = []
        for i, line in enumerate(lines):
            score = 0
            line_lower = line.lower()
            line_stripped = line.strip()
            
            # High score for exact key term matches
            for term in key_terms:
                if term in line_lower:
                    score += 3
            
            # Very high score for function definitions that match calls in prefix/suffix
            for func_name in function_names:
                if re.match(rf'^\s*def\s+{re.escape(func_name)}\s*\(', line):
                    score += 10
            
            # Boost important code structures
            if re.match(r'^\s*(def|class)\s+', line):
                score += 5
            elif re.match(r'^\s*(import|from)\s+', line):
                score += 2
            elif re.match(r'^\s*(return|yield)\s+', line):
                score += 3
            
            # Boost lines with control structures
            if any(pattern in line_lower for pattern in ['if ', 'for ', 'while ', 'try:', 'except']):
                score += 2
            
            # Boost lines with assignments to key terms
            for term in key_terms:
                if re.search(rf'\b{re.escape(term)}\s*=', line_lower):
                    score += 4
            
            # Penalize very long lines (likely not as relevant)
            if len(line) > 120:
                score -= 1
            
            # Penalize empty lines and comments (but don't exclude entirely)
            if not line_stripped or line_stripped.startswith('#'):
                score -= 0.5
            
            line_scores.append((i, score, line))
        
        # Select top scoring lines with context preservation
        line_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take top scoring lines but ensure we maintain some context
        selected_indices = set()
        max_lines = min(40, len(lines))
        
        for i, score, line in line_scores[:max_lines]:
            if score > 0:
                selected_indices.add(i)
                # Add context lines around high-scoring lines
                if score > 5:
                    selected_indices.update(range(max(0, i-1), min(len(lines), i+2)))
        
        if not selected_indices:
            # Fallback: take first meaningful part
            selected_indices = set(range(min(30, len(lines))))
        
        # Sort selected lines back to original order and build content
        selected_lines = sorted([(i, lines[i]) for i in selected_indices])
        
        # Build optimized content with gap indicators
        optimized_lines = []
        last_idx = -2
        
        for idx, line in selected_lines:
            if idx > last_idx + 1:
                if optimized_lines:  # Don't add gap at the beginning
                    optimized_lines.append("# ...")
            optimized_lines.append(line)
            last_idx = idx
        
        return '\n'.join(optimized_lines)
    
    def _prefilter_files(self, repo_files: List[Path], prefix: str, suffix: str) -> List[Path]:
        """Pre-filter files by basic relevance before expensive processing."""
        import re
        
        # Extract key identifiers from prefix/suffix
        key_identifiers = set()
        for text in [prefix, suffix]:
            identifiers = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', text)
            key_identifiers.update([id.lower() for id in identifiers if len(id) > 2])
        
        # Score files by filename and quick content scan
        file_scores = []
        
        for file_path in repo_files:
            score = 0
            filename = file_path.name.lower()
            
            # Boost files with relevant names
            for identifier in key_identifiers:
                if identifier in filename:
                    score += 5
            
            # Quick content scan for key identifiers
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    # Read first 1000 chars for quick scan
                    quick_content = f.read(1000).lower()
                    for identifier in key_identifiers:
                        score += quick_content.count(identifier)
            except:
                continue
            
            file_scores.append((file_path, score))
        
        # Sort by score and return top files
        file_scores.sort(key=lambda x: x[1], reverse=True)
        return [file_path for file_path, score in file_scores if score > 0] or [fp for fp, _ in file_scores[:10]]
    
    def _apply_hierarchical_ranking(self, base_scores: np.ndarray, file_paths: List[Path], 
                                  file_contents: List[str], prefix: str, suffix: str) -> np.ndarray:
        """Apply hierarchical ranking with research-backed scoring factors."""
        # Primary ranking factors (from Framework.md)
        semantic_scores = base_scores  # Already computed semantic similarity
        structural_scores = self._compute_structural_similarity(file_contents, prefix, suffix)
        recency_scores = self._compute_recency_scores(file_paths)
        dependency_scores = self._compute_dependency_relevance(file_paths, prefix, suffix)
        
        # Combine with research-backed weights
        hierarchical_scores = (
            self.scoring_weights["semantic"] * semantic_scores +
            self.scoring_weights["structural"] * structural_scores +
            self.scoring_weights["recency"] * recency_scores +
            self.scoring_weights["dependency"] * dependency_scores
        )
        
        return hierarchical_scores
    
    def _compute_structural_similarity(self, file_contents: List[str], prefix: str, suffix: str) -> np.ndarray:
        """Compute structural similarity based on code patterns."""
        import re
        scores = np.zeros(len(file_contents))
        
        # Extract structural patterns from prefix/suffix
        prefix_patterns = self._extract_code_patterns(prefix)
        suffix_patterns = self._extract_code_patterns(suffix)
        all_patterns = prefix_patterns + suffix_patterns
        
        for i, content in enumerate(file_contents):
            content_patterns = self._extract_code_patterns(content)
            
            # Count matching patterns
            pattern_matches = 0
            for pattern in all_patterns:
                if pattern in content_patterns:
                    pattern_matches += 1
            
            scores[i] = pattern_matches / max(len(all_patterns), 1)
        
        return scores
    
    def _extract_code_patterns(self, code: str) -> List[str]:
        """Extract structural code patterns."""
        import re
        patterns = []
        
        # Function definitions
        patterns.extend(re.findall(r'def\s+(\w+)', code))
        # Class definitions  
        patterns.extend(re.findall(r'class\s+(\w+)', code))
        # Control structures
        if 'if ' in code: patterns.append('if_statement')
        if 'for ' in code: patterns.append('for_loop')
        if 'while ' in code: patterns.append('while_loop')
        if 'try:' in code: patterns.append('try_except')
        # Return patterns
        if 'return ' in code: patterns.append('return_statement')
        
        return patterns
    
    def _compute_recency_scores(self, file_paths: List[Path]) -> np.ndarray:
        """Compute recency scores based on file modification times."""
        import os
        import time
        
        scores = np.zeros(len(file_paths))
        current_time = time.time()
        
        for i, file_path in enumerate(file_paths):
            try:
                # Get file modification time
                mod_time = os.path.getmtime(file_path)
                # Convert to recency score (more recent = higher score)
                age_hours = (current_time - mod_time) / 3600
                scores[i] = 1.0 / (1.0 + age_hours / 24)  # Decay over days
            except:
                scores[i] = 0.5  # Default score if can't get mod time
        
        return scores
    
    def _compute_dependency_relevance(self, file_paths: List[Path], prefix: str, suffix: str) -> np.ndarray:
        """Compute dependency relevance based on imports and relationships."""
        import re
        scores = np.zeros(len(file_paths))
        
        # Extract imports from prefix/suffix
        prefix_imports = re.findall(r'(?:import|from)\s+([a-zA-Z_][a-zA-Z0-9_.]*)', prefix + suffix)
        
        for i, file_path in enumerate(file_paths):
            filename = file_path.stem  # filename without extension
            
            # Score based on import relationships
            for imp in prefix_imports:
                if filename in imp or imp in filename:
                    scores[i] += 1.0
            
            # Score based on filename similarity to imports
            for imp in prefix_imports:
                if any(part in filename for part in imp.split('.')):
                    scores[i] += 0.5
        
        # Normalize scores
        max_score = max(scores) if max(scores) > 0 else 1
        return scores / max_score
    
    def _calculate_adaptive_thresholds(self, scores: np.ndarray) -> List[float]:
        """Calculate multiple adaptive thresholds (inspired by mt6 in team name)."""
        if len(scores) == 0:
            return [0.0]
        
        # Calculate 6 different thresholds based on score distribution
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        max_score = np.max(scores)
        
        thresholds = [
            max_score * 0.9,           # Very high threshold
            mean_score + 2 * std_score, # High threshold  
            mean_score + std_score,     # Above average + 1 std
            mean_score,                 # Average threshold
            mean_score - std_score,     # Below average threshold
            np.percentile(scores, 25)   # Lower quartile threshold
        ]
        
        return sorted(thresholds, reverse=True)
    
    def _select_with_multiple_thresholds(self, sorted_indices: np.ndarray, 
                                       scores: np.ndarray, thresholds: List[float]) -> List[int]:
        """Select files using multiple thresholds for robust selection."""
        selected_indices = []
        
        # Try each threshold until we have enough files
        for threshold in thresholds:
            candidates = [idx for idx in sorted_indices if scores[idx] >= threshold]
            
            if len(candidates) >= 2:  # Minimum 2 files
                selected_indices = candidates[:5]  # Maximum 5 files
                break
        
        # Fallback: take top 3 if no threshold worked
        if not selected_indices:
            selected_indices = sorted_indices[:3].tolist()
        
        return selected_indices
    
    def _retrieve_rationale_context(self, prefix: str, suffix: str, repo_files: List[Path]) -> List[Dict]:
        """Retrieve rationale context: documentation, test cases, error handling patterns."""
        rationale_candidates = []
        
        for file_path in repo_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Score file for rationale content
                rationale_score = 0
                
                # Look for documentation patterns
                if any(pattern in content.lower() for pattern in ['"""', "'''", '# ', 'docstring', 'doc:']):
                    rationale_score += 3
                
                # Look for test patterns
                if any(pattern in content.lower() for pattern in ['test_', 'assert', 'unittest', 'pytest']):
                    rationale_score += 2
                
                # Look for error handling
                if any(pattern in content.lower() for pattern in ['try:', 'except', 'raise', 'error', 'exception']):
                    rationale_score += 2
                
                # Look for examples and usage
                if any(pattern in content.lower() for pattern in ['example', 'usage', 'demo', 'sample']):
                    rationale_score += 1
                
                if rationale_score > 0:
                    rationale_candidates.append({
                        'file_path': file_path,
                        'content': content,
                        'score': rationale_score,
                        'type': 'rationale'
                    })
            except Exception as e:
                logger.warning(f"Could not read {file_path}: {e}")
        
        # Sort by score and return top candidates
        rationale_candidates.sort(key=lambda x: x['score'], reverse=True)
        return rationale_candidates[:3]  # Top 3 rationale files
    
    def _compose_dual_context(self, analogy_context: List[Dict], rationale_context: List[Dict], 
                            prefix: str, suffix: str) -> str:
        """Compose dual context using research-backed interleaving strategy."""
        try:
            context_parts = []
            total_length = 0
            max_length = min(self.config["retrieval"]["max_context_length"], 6000)
            
            # Safety check for empty contexts
            if not analogy_context and not rationale_context:
                return f"{self.file_separator}empty.py\n# No context available\npass"
            
            # Research finding: Interleave analogy and rationale contexts for optimal performance
            # Start with analogy context (code patterns) as it's more directly relevant
            
            # Add top analogy files first
            for i, candidate in enumerate(analogy_context[:2]):  # Top 2 analogy files
                if total_length >= max_length * 0.7:  # Reserve space for rationale
                    break
                    
                file_path = candidate['file_path']
                content = candidate['content']
                
                # Optimize content for analogy context
                optimized_content = self._optimize_file_content(content, prefix, suffix)
                
                context_part = f"{self.file_separator}{file_path.name}\n{optimized_content}"
                
                if total_length + len(context_part) <= max_length:
                    context_parts.append(context_part)
                    total_length += len(context_part)
            
            # Add rationale context (documentation, tests, examples)
            for candidate in rationale_context[:1]:  # Top 1 rationale file
                if total_length >= max_length * 0.9:
                    break
                    
                file_path = candidate['file_path']
                content = candidate['content']
                
                # Extract most relevant documentation/examples
                rationale_content = self._extract_rationale_content(content, prefix, suffix)
                
                context_part = f"{self.file_separator}{file_path.name}\n{rationale_content}"
                
                if total_length + len(context_part) <= max_length:
                    context_parts.append(context_part)
                    total_length += len(context_part)
            
            # Fill remaining space with more analogy context if available
            for candidate in analogy_context[2:]:
                if total_length >= max_length * 0.95:
                    break
                    
                file_path = candidate['file_path']
                content = candidate['content']
                
                optimized_content = self._optimize_file_content(content, prefix, suffix)
                context_part = f"{self.file_separator}{file_path.name}\n{optimized_content}"
                
                if total_length + len(context_part) <= max_length:
                    context_parts.append(context_part)
                    total_length += len(context_part)
            
            if context_parts:
                final_context = "\n\n".join(context_parts)
            else:
                final_context = self._create_fallback_context("No relevant context found")
            
            return final_context
            
        except Exception as e:
            logger.warning(f"Context composition failed: {e}, using fallback")
            return f"{self.file_separator}composition_error.py\n# Context composition failed: {str(e)}\npass"
    
    def _extract_rationale_content(self, content: str, prefix: str, suffix: str) -> str:
        """Extract most relevant rationale content (docs, examples, tests)."""
        lines = content.split('\n')
        relevant_lines = []
        
        # Extract key terms for relevance
        import re
        key_terms = set()
        for text in [prefix, suffix]:
            terms = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', text.lower())
            key_terms.update([t for t in terms if len(t) > 2])
        
        # Score lines for rationale relevance
        for i, line in enumerate(lines):
            line_lower = line.strip().lower()
            
            # High priority for documentation
            if any(pattern in line for pattern in ['"""', "'''", '# ']):
                # Check if documentation mentions key terms
                if any(term in line_lower for term in key_terms):
                    relevant_lines.extend(lines[max(0, i-1):min(len(lines), i+5)])  # Include context
            
            # Medium priority for examples and tests
            elif any(pattern in line_lower for pattern in ['example', 'test_', 'assert', 'usage']):
                if any(term in line_lower for term in key_terms):
                    relevant_lines.extend(lines[max(0, i-1):min(len(lines), i+3)])
        
        # If no specific rationale found, return first part of file
        if not relevant_lines:
            relevant_lines = lines[:20]
        
        return '\n'.join(relevant_lines[:30])  # Limit length
    
    def _create_fallback_context(self, reason: str) -> str:
        """Create fallback context when retrieval fails."""
        return f"{self.file_separator}fallback.py\n# {reason}\n# Context retrieval failed, using minimal fallback\npass"
    
    def _score_with_hypothetics(self, prefix: str, suffix: str, files_content: List[List[str]]) -> np.ndarray:
        """Generate hypothetic completions and score files based on similarity (RAG approach)."""
        # Generate potential completion patterns based on prefix/suffix
        hypothetic_completions = self._generate_hypothetic_lines(prefix, suffix)
        
        if not hypothetic_completions:
            return np.zeros(len(files_content))
        
        # Score files based on similarity to hypothetic completions
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi(files_content)
        
        # Combine scores from all hypothetic completions
        combined_scores = np.zeros(len(files_content))
        for hyp_completion in hypothetic_completions:
            hyp_tokens = hyp_completion.lower().split()
            scores = bm25.get_scores(hyp_tokens)
            combined_scores += scores
        
        return combined_scores / len(hypothetic_completions) if hypothetic_completions else combined_scores
    
    def _generate_hypothetic_lines(self, prefix: str, suffix: str) -> List[str]:
        """Generate hypothetic completion lines based on code patterns."""
        import re
        hypothetics = []
        
        # Pattern 1: Function call completion
        if '(' in prefix and ')' not in prefix.split('\n')[-1]:
            # Extract function name and generate potential completions
            func_match = re.search(r'(\w+)\s*\($', prefix.strip())
            if func_match:
                func_name = func_match.group(1)
                hypothetics.extend([
                    f"{func_name}()",
                    f"{func_name}(self)",
                    f"{func_name}(args)",
                    f"return {func_name}()"
                ])
        
        # Pattern 2: Variable assignment completion
        if '=' in prefix.split('\n')[-1] and '=' not in suffix.split('\n')[0]:
            hypothetics.extend([
                "= None",
                "= []",
                "= {}",
                "= self.",
                "= get_",
                "= create_"
            ])
        
        # Pattern 3: Control structure completion
        last_line = prefix.strip().split('\n')[-1] if prefix.strip() else ""
        if last_line.strip().endswith(':'):
            if 'if ' in last_line:
                hypothetics.extend(["return True", "return False", "pass", "raise"])
            elif 'def ' in last_line:
                hypothetics.extend(["return", "pass", "raise NotImplementedError"])
            elif 'class ' in last_line:
                hypothetics.extend(["def __init__(self)", "pass"])
        
        # Pattern 4: Import completion
        if prefix.strip().endswith('import') or 'from ' in prefix.split('\n')[-1]:
            hypothetics.extend([
                "import os",
                "import sys", 
                "from typing import",
                "import json"
            ])
        
        return hypothetics[:5]  # Limit to top 5 hypothetics
    
    def _compose_dual_context(self, indices: List[int], file_paths: List[Path], 
                            file_contents: List[str], scores: np.ndarray, 
                            prefix: str, suffix: str) -> str:
        """Compose dual context: Analogy Context (AC) + Rationale Context (RC)."""
        file_separator = COMPETITION_CONFIG["file_separator"]
        
        # Separate files into analogy and rationale contexts
        analogy_files, rationale_files = self._categorize_context_files(
            indices, file_paths, file_contents, scores, prefix, suffix
        )
        
        # Build Analogy Context (AC) - Similar code patterns and usage examples
        analogy_context = self._build_analogy_context(analogy_files, prefix, suffix)
        
        # Build Rationale Context (RC) - Documentation, tests, and explanations  
        rationale_context = self._build_rationale_context(rationale_files, prefix, suffix)
        
        # Optimal interleaving based on research findings (+6.68 to +7.32 EM improvement)
        return self._interleave_dual_contexts(analogy_context, rationale_context)
    
    def _categorize_context_files(self, indices: List[int], file_paths: List[Path],
                                file_contents: List[str], scores: np.ndarray,
                                prefix: str, suffix: str) -> tuple:
        """Categorize files into analogy vs rationale contexts."""
        analogy_files = []
        rationale_files = []
        
        for idx in indices:
            if idx >= len(file_paths):
                continue
                
            file_path = file_paths[idx]
            content = file_contents[idx]
            filename = file_path.name.lower()
            
            # Rationale context: tests, docs, examples
            if any(keyword in filename for keyword in ['test_', '_test', 'example', 'doc', 'readme']):
                rationale_files.append((file_path, content, scores[idx]))
            # Analogy context: similar code patterns
            else:
                analogy_files.append((file_path, content, scores[idx]))
        
        # Sort by relevance scores
        analogy_files.sort(key=lambda x: x[2], reverse=True)
        rationale_files.sort(key=lambda x: x[2], reverse=True)
        
        return analogy_files[:3], rationale_files[:2]  # Limit context size
    
    def _build_analogy_context(self, analogy_files: List[tuple], prefix: str, suffix: str) -> str:
        """Build analogy context with similar code patterns and usage examples."""
        file_separator = COMPETITION_CONFIG["file_separator"]
        context_parts = []
        
        for file_path, content, score in analogy_files:
            if score <= 0:
                continue
                
            clean_name = file_path.name
            optimized_content = self._optimize_file_content(content, prefix, suffix)
            
            context_part = f"{file_separator}{clean_name}\n{optimized_content}"
            context_parts.append(context_part)
        
        return "\n\n".join(context_parts)
    
    def _build_rationale_context(self, rationale_files: List[tuple], prefix: str, suffix: str) -> str:
        """Build rationale context with documentation and explanations."""
        file_separator = COMPETITION_CONFIG["file_separator"]
        context_parts = []
        
        for file_path, content, score in rationale_files:
            if score <= 0:
                continue
                
            clean_name = file_path.name
            # Extract documentation, comments, and docstrings
            doc_content = self._extract_documentation(content)
            
            if doc_content:
                context_part = f"{file_separator}{clean_name}\n{doc_content}"
                context_parts.append(context_part)
        
        return "\n\n".join(context_parts)
    
    def _extract_documentation(self, content: str) -> str:
        """Extract documentation, comments, and docstrings from code."""
        import re
        lines = content.split('\n')
        doc_lines = []
        
        for line in lines:
            stripped = line.strip()
            # Include docstrings
            if '"""' in line or "'''" in line:
                doc_lines.append(line)
            # Include comments
            elif stripped.startswith('#'):
                doc_lines.append(line)
            # Include function/class definitions with docstrings
            elif re.match(r'^\s*(def|class)\s+', line):
                doc_lines.append(line)
        
        return '\n'.join(doc_lines) if doc_lines else content[:500]  # Fallback to first 500 chars
    
    def _interleave_dual_contexts(self, analogy_context: str, rationale_context: str) -> str:
        """Optimally interleave analogy and rationale contexts based on research."""
        max_length = min(self.config["retrieval"]["max_context_length"], 6000)
        
        # Research shows analogy context should come first, then rationale
        if analogy_context and rationale_context:
            combined = f"{analogy_context}\n\n{rationale_context}"
        elif analogy_context:
            combined = analogy_context
        elif rationale_context:
            combined = rationale_context
        else:
            return self._create_fallback_context("No context available")
        
        # Trim to fit length constraints
        if len(combined) > max_length:
            # Preserve analogy context priority
            analogy_len = min(len(analogy_context), int(max_length * 0.7))
            rationale_len = max_length - analogy_len
            
            trimmed_analogy = analogy_context[:analogy_len] if analogy_context else ""
            trimmed_rationale = rationale_context[:rationale_len] if rationale_context else ""
            
            combined = f"{trimmed_analogy}\n\n{trimmed_rationale}".strip()
        
        return combined
    
    def _generate_hypothetic_lines(self, prefix: str, suffix: str) -> List[str]:
        """Generate hypothetic completion lines based on prefix/suffix patterns (2405.07530v1)."""
        import re
        
        hypothetics = []
        
        # Pattern 1: Function call completion
        if re.search(r'\w+\s*\($', prefix.strip()):
            func_match = re.search(r'(\w+)\s*\($', prefix.strip())
            if func_match:
                func_name = func_match.group(1)
                hypothetics.extend([
                    f"{func_name}()",
                    f"{func_name}(self)",
                    f"{func_name}(args)",
                    f"return {func_name}()"
                ])
        
        # Pattern 2: Variable assignment completion
        if re.search(r'\w+\s*=\s*$', prefix.strip()):
            var_match = re.search(r'(\w+)\s*=\s*$', prefix.strip())
            if var_match:
                var_name = var_match.group(1)
                hypothetics.extend([
                    f"{var_name} = None",
                    f"{var_name} = []",
                    f"{var_name} = {{}}",
                    f"{var_name} = self.{var_name}"
                ])
        
        # Pattern 3: Method definition completion
        if 'def ' in prefix and '(' in prefix:
            hypothetics.extend([
                "def method(self):",
                "def __init__(self):",
                "def process(self, data):",
                "return result"
            ])
        
        # Pattern 4: Import statement completion
        if prefix.strip().endswith('import '):
            hypothetics.extend([
                "import os",
                "import sys", 
                "import json",
                "from typing import"
            ])
        
        # Pattern 5: Class definition completion
        if 'class ' in prefix:
            hypothetics.extend([
                "class MyClass:",
                "def __init__(self):",
                "super().__init__()"
            ])
        
        return hypothetics[:10]  # Limit to top 10 hypothetics
    
    def _separate_analogy_rationale_contexts(self, file_paths: List[Path], 
                                           file_contents: List[str], 
                                           scores: np.ndarray) -> tuple:
        """Separate files into Analogy Context (AC) and Rationale Context (RC) (2402.14323v2)."""
        
        analogy_candidates = []
        rationale_candidates = []
        
        for i, (file_path, content, score) in enumerate(zip(file_paths, file_contents, scores)):
            filename = file_path.name.lower()
            
            # Rationale Context: Documentation, tests, examples
            if any(keyword in filename for keyword in ['test', 'doc', 'readme', 'example', 'demo']):
                rationale_candidates.append((i, file_path, content, score))
            # Rationale Context: Files with high comment/docstring ratio
            elif self._has_high_documentation_ratio(content):
                rationale_candidates.append((i, file_path, content, score))
            # Analogy Context: Regular code files
            else:
                analogy_candidates.append((i, file_path, content, score))
        
        # Sort by score
        analogy_candidates.sort(key=lambda x: x[3], reverse=True)
        rationale_candidates.sort(key=lambda x: x[3], reverse=True)
        
        return analogy_candidates, rationale_candidates
    
    def _has_high_documentation_ratio(self, content: str) -> bool:
        """Check if file has high documentation/comment ratio."""
        lines = content.split('\n')
        doc_lines = 0
        code_lines = 0
        
        in_docstring = False
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
                
            # Check for docstring
            if '"""' in stripped or "'''" in stripped:
                in_docstring = not in_docstring
                doc_lines += 1
            elif in_docstring:
                doc_lines += 1
            elif stripped.startswith('#'):
                doc_lines += 1
            else:
                code_lines += 1
        
        total_lines = doc_lines + code_lines
        return total_lines > 0 and (doc_lines / total_lines) > 0.3
    
    def _compose_dual_context(self, analogy_candidates: List[tuple], 
                            rationale_candidates: List[tuple],
                            prefix: str, suffix: str) -> str:
        """Compose dual context with optimal interleaving (2402.14323v2)."""
        context_parts = []
        total_length = 0
        max_length = min(self.config["retrieval"]["max_context_length"], 6000)
        
        # Research finding: Start with analogy context for better performance
        analogy_count = 0
        rationale_count = 0
        
        # Interleave analogy and rationale contexts (2:1 ratio based on research)
        candidates_iter = []
        
        # Add analogy contexts (2 parts)
        for candidate in analogy_candidates[:4]:
            candidates_iter.append(('analogy', candidate))
        
        # Add rationale context (1 part)  
        for candidate in rationale_candidates[:2]:
            candidates_iter.append(('rationale', candidate))
        
        # Sort by score for final selection
        candidates_iter.sort(key=lambda x: x[1][3], reverse=True)
        
        for context_type, (idx, file_path, content, score) in candidates_iter:
            if total_length > max_length * 0.9:
                break
                
            clean_name = file_path.name
            
            if context_type == 'analogy':
                # Optimize for code patterns and usage examples
                optimized_content = self._optimize_file_content(content, prefix, suffix)
                context_part = f"{self.file_separator}{clean_name}\n{optimized_content}"
            else:
                # Extract documentation and explanations for rationale
                doc_content = self._extract_documentation_content(content)
                context_part = f"{self.file_separator}{clean_name}\n{doc_content}"
            
            if total_length + len(context_part) <= max_length:
                context_parts.append(context_part)
                total_length += len(context_part)
        
        if context_parts:
            return "\n\n".join(context_parts)
        else:
            return self._create_fallback_context("No dual context candidates found")
    
    def _extract_documentation_content(self, content: str) -> str:
        """Extract documentation, comments, and docstrings for rationale context."""
        lines = content.split('\n')
        doc_lines = []
        
        in_docstring = False
        docstring_delimiter = None
        
        for line in lines:
            stripped = line.strip()
            
            # Detect docstring start/end
            if '"""' in stripped or "'''" in stripped:
                if not in_docstring:
                    in_docstring = True
                    docstring_delimiter = '"""' if '"""' in stripped else "'''"
                    doc_lines.append(line)
                elif docstring_delimiter in stripped:
                    in_docstring = False
                    doc_lines.append(line)
                    docstring_delimiter = None
                else:
                    doc_lines.append(line)
            elif in_docstring:
                doc_lines.append(line)
            elif stripped.startswith('#'):
                doc_lines.append(line)
            elif any(keyword in stripped.lower() for keyword in ['todo', 'fixme', 'note', 'warning']):
                doc_lines.append(line)
        
        # If no documentation found, return first 20 lines as context
        if not doc_lines:
            return '\n'.join(lines[:20])
        
        return '\n'.join(doc_lines[:30])  # Limit documentation context
    
    def _create_fallback_context(self, reason: str) -> str:
        """Create fallback context when retrieval fails."""
        file_separator = COMPETITION_CONFIG["file_separator"]
        return f"{file_separator}fallback.py\n# {reason}\n# Context retrieval failed, using minimal fallback\npass"
    
    def _generate_hypothetic_completions(self, prefix: str, suffix: str) -> List[str]:
        """
        Generate hypothetic completion lines based on prefix/suffix patterns.
        
        Based on research from 2405.07530v1: RAG with hypothetic line generation
        significantly enhances code completion accuracy.
        """
        import re
        
        hypothetics = []
        
        # Pattern 1: Function call completion
        if re.search(r'\w+\s*\($', prefix.strip()):
            func_match = re.search(r'(\w+)\s*\($', prefix.strip())
            if func_match:
                func_name = func_match.group(1)
                hypothetics.extend([
                    f"{func_name}(self)",
                    f"{func_name}(self, *args)",
                    f"{func_name}(self, **kwargs)",
                    f"return {func_name}(",
                ])
        
        # Pattern 2: Variable assignment completion
        if re.search(r'\w+\s*=\s*$', prefix.strip()):
            var_match = re.search(r'(\w+)\s*=\s*$', prefix.strip())
            if var_match:
                var_name = var_match.group(1)
                hypothetics.extend([
                    f"{var_name} = None",
                    f"{var_name} = []",
                    f"{var_name} = {{}}",
                    f"{var_name} = self.",
                ])
        
        # Pattern 3: Import statement completion
        if 'import' in prefix.lower():
            hypothetics.extend([
                "import os",
                "import sys", 
                "from typing import",
                "import json",
                "import re"
            ])
        
        # Pattern 4: Control structure completion
        if re.search(r'(if|for|while|try)\s+.*:$', prefix.strip()):
            hypothetics.extend([
                "if condition:",
                "for item in items:",
                "while condition:",
                "try:",
                "except Exception as e:"
            ])
        
        # Pattern 5: Class/method definition completion
        if re.search(r'(def|class)\s+\w+', prefix):
            hypothetics.extend([
                "def __init__(self):",
                "def method(self):",
                "class ClassName:",
                "return self"
            ])
        
        return hypothetics[:10]  # Limit to top 10 hypothetics
    
    def _retrieve_analogy_context(self, prefix: str, suffix: str, repo_files: List[Path], hypothetics: List[str]) -> List[tuple]:
        """
        Retrieve Analogy Context (AC): Similar code patterns, usage examples, design patterns.
        
        Based on research from 2402.14323v2: Dual Context Approach
        """
        from rank_bm25 import BM25Okapi
        import re
        
        # Enhanced code-aware tokenization (from existing implementation)
        def tokenize_code(text):
            tokens = re.findall(r'\w+|[^\w\s]', text.lower())
            identifiers = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', text)
            tokens.extend([id.lower() for id in identifiers])
            func_classes = re.findall(r'(?:def|class)\s+([a-zA-Z_][a-zA-Z0-9_]*)', text)
            tokens.extend([name.lower() + '_definition' for name in func_classes])
            imports = re.findall(r'(?:import|from)\s+([a-zA-Z_][a-zA-Z0-9_.]*)', text)
            tokens.extend([imp.lower() + '_import' for imp in imports])
            return tokens
        
        # Pre-filter files for analogy context (code patterns)
        relevant_files = self._prefilter_files(repo_files, prefix, suffix)
        
        files_content = []
        file_paths = []
        file_raw_content = []
        
        for file_path in relevant_files[:self.config["retrieval"]["max_files_per_repo"]]:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if len(content.strip()) > self.config["retrieval"]["min_file_size"]:
                        # Filter for code-heavy files (analogy context)
                        if self._is_code_heavy(content):
                            files_content.append(tokenize_code(content))
                            file_paths.append(file_path)
                            file_raw_content.append(content)
            except Exception as e:
                logger.warning(f"Could not read {file_path}: {e}")
        
        if not files_content:
            return []
        
        # Enhanced query with hypothetics
        prefix_tokens = tokenize_code(prefix)
        suffix_tokens = tokenize_code(suffix)
        hypothetic_tokens = []
        for hyp in hypothetics:
            hypothetic_tokens.extend(tokenize_code(hyp))
        
        # Multi-strategy scoring for analogy context
        bm25 = BM25Okapi(files_content, k1=1.5, b=0.75)
        
        combined_query = prefix_tokens + suffix_tokens
        combined_scores = bm25.get_scores(combined_query)
        
        # Hypothetic enhancement (research-backed)
        hypothetic_scores = bm25.get_scores(hypothetic_tokens) if hypothetic_tokens else np.zeros_like(combined_scores)
        
        # Final scoring with hypothetic boost
        final_scores = 0.7 * combined_scores + 0.3 * hypothetic_scores
        
        # Return top analogy files with scores
        sorted_indices = np.argsort(final_scores)[::-1]
        analogy_files = []
        
        for idx in sorted_indices[:5]:  # Top 5 analogy files
            if final_scores[idx] > 0:
                analogy_files.append((idx, file_paths[idx], file_raw_content[idx], final_scores[idx]))
        
        return analogy_files
    
    def _retrieve_rationale_context(self, prefix: str, suffix: str, repo_files: List[Path]) -> List[tuple]:
        """
        Retrieve Rationale Context (RC): Documentation, test cases, error handling patterns.
        
        Based on research from 2402.14323v2: Dual Context Approach
        """
        rationale_files = []
        
        for file_path in repo_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                    # Score for rationale content (documentation, tests, comments)
                    rationale_score = self._score_rationale_content(content, prefix, suffix)
                    
                    if rationale_score > 0:
                        rationale_files.append((0, file_path, content, rationale_score))
                        
            except Exception as e:
                continue
        
        # Sort by rationale score and return top files
        rationale_files.sort(key=lambda x: x[3], reverse=True)
        return rationale_files[:3]  # Top 3 rationale files
    
    def _is_code_heavy(self, content: str) -> bool:
        """Check if file contains substantial code (for analogy context)."""
        lines = content.split('\n')
        code_lines = 0
        
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#') and not stripped.startswith('"""') and not stripped.startswith("'''"):
                code_lines += 1
        
        return code_lines > len(lines) * 0.3  # At least 30% code lines
    
    def _score_rationale_content(self, content: str, prefix: str, suffix: str) -> float:
        """Score content for rationale value (documentation, tests, etc.)."""
        score = 0.0
        content_lower = content.lower()
        
        # Documentation indicators
        if 'docstring' in content_lower or '"""' in content or "'''" in content:
            score += 2.0
        
        # Test file indicators
        if 'test' in content_lower or 'assert' in content_lower:
            score += 1.5
        
        # README or documentation files
        if 'readme' in content_lower or 'doc' in content_lower:
            score += 1.0
        
        # Error handling patterns
        if 'except' in content_lower or 'raise' in content_lower or 'error' in content_lower:
            score += 0.5
        
        # Comments ratio (higher = more documentation)
        lines = content.split('\n')
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        if lines:
            comment_ratio = comment_lines / len(lines)
            score += comment_ratio * 2
        
        return score
    
    def _compose_dual_context(self, analogy_files: List[tuple], rationale_files: List[tuple], 
                             prefix: str, suffix: str) -> str:
        """
        Compose dual context using research-backed interleaving strategy.
        
        Based on 2402.14323v2: Optimal AC/RC interleaving shows +6.68 to +7.32 EM improvement
        """
        context_parts = []
        file_separator = self.file_separator
        total_length = 0
        max_length = min(self.config["retrieval"]["max_context_length"], 6000)
        
        # Research-backed interleaving: Start with analogy, then rationale, 2:1 ratio
        analogy_count = 0
        rationale_count = 0
        
        while (analogy_count < len(analogy_files) or rationale_count < len(rationale_files)) and total_length < max_length * 0.9:
            
            # Add Analogy Context (2:1 ratio based on research)
            if analogy_count < len(analogy_files) and analogy_count < 3:
                idx, file_path, content, score = analogy_files[analogy_count]
                optimized_content = self._optimize_file_content(content, prefix, suffix)
                
                context_part = f"{file_separator}{file_path.name}\n{optimized_content}"
                
                if total_length + len(context_part) < max_length:
                    context_parts.append(context_part)
                    total_length += len(context_part)
                    analogy_count += 1
                else:
                    break
            
            # Add Rationale Context (every 2nd iteration)
            if rationale_count < len(rationale_files) and analogy_count % 2 == 0 and rationale_count < 2:
                idx, file_path, content, score = rationale_files[rationale_count]
                
                # Extract key documentation/comments
                rationale_content = self._extract_rationale_content(content, prefix, suffix)
                
                if rationale_content:
                    context_part = f"{file_separator}{file_path.name}\n{rationale_content}"
                    
                    if total_length + len(context_part) < max_length:
                        context_parts.append(context_part)
                        total_length += len(context_part)
                        rationale_count += 1
                    else:
                        break
            
            # Prevent infinite loop
            if analogy_count >= len(analogy_files) and rationale_count >= len(rationale_files):
                break
        
        if context_parts:
            return "\n\n".join(context_parts)
        else:
            return self._create_fallback_context("No relevant dual context found")
    
    def _extract_rationale_content(self, content: str, prefix: str, suffix: str) -> str:
        """Extract rationale content (documentation, comments, tests)."""
        lines = content.split('\n')
        rationale_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            # Include docstrings
            if '"""' in line or "'''" in line:
                rationale_lines.append(line)
            
            # Include comments
            elif stripped.startswith('#'):
                rationale_lines.append(line)
            
            # Include test assertions
            elif 'assert' in stripped.lower() or 'test' in stripped.lower():
                rationale_lines.append(line)
            
            # Include error handling
            elif any(keyword in stripped.lower() for keyword in ['except', 'raise', 'error', 'warning']):
                rationale_lines.append(line)
        
        # Limit rationale content length
        rationale_content = '\n'.join(rationale_lines[:30])  # Max 30 lines
        
        return rationale_content if rationale_content.strip() else "# No rationale content found"

@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=MODAL_CONFIG["timeouts"]["context_generation"] * 3,  # Triple timeout for safety
    memory=MODAL_CONFIG["memory"]["large"] * 2,  # Double memory allocation
    cpu=8.0,  # Maximum CPU allocation to prevent preemption
    retries=5,  # More retries for network issues
    min_containers=1  # Keep one instance warm to avoid cold starts
)
def generate_competition_submission(
    stage: str = "public", 
    language: str = "python",
    batch_size: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generate complete competition submission using SOTA framework.
    
    Args:
        stage: Competition stage (public, practice, etc.)
        language: Programming language (python, kotlin)
        batch_size: Number of datapoints to process (None for all)
        
    Returns:
        Dictionary with processing results and file paths
    """
    logger.info(f"Generating SOTA submission for {language}-{stage}")
    
    data_dir = Path("/data")
    
    # Load competition data
    jsonl_file = data_dir / f"{language}-{stage}.jsonl"
    if not jsonl_file.exists():
        return {"status": "error", "message": f"Data file not found: {jsonl_file}"}
    
    with jsonlines.open(jsonl_file, 'r') as reader:
        datapoints = list(reader)
    
    if batch_size:
        datapoints = datapoints[:batch_size]
    
    logger.info(f"Processing {len(datapoints)} datapoints")
    
    # Initialize SOTA retriever
    retriever = SOTAContextRetriever(language=language)
    
    # Process datapoints with checkpointing
    predictions = []
    results = []
    
    # Check for existing checkpoint
    checkpoint_file = data_dir / f"{language}-{stage}-checkpoint.json"
    start_index = 0
    
    # Graceful shutdown handler
    shutdown_requested = False
    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        logger.info(f"Shutdown signal {signum} received, saving checkpoint...")
        shutdown_requested = True
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                start_index = checkpoint_data.get('last_processed', 0) + 1
                predictions = checkpoint_data.get('predictions', [])
                results = checkpoint_data.get('results', [])
                logger.info(f"Resuming from checkpoint at datapoint {start_index}")
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")
            start_index = 0
    
    for i, dp in enumerate(tqdm(datapoints[start_index:], desc="Processing datapoints", initial=start_index, total=len(datapoints))):
        actual_index = start_index + i
        
        # Check for shutdown request
        if shutdown_requested:
            logger.info("Shutdown requested, saving final checkpoint...")
            checkpoint_data = {
                'last_processed': actual_index - 1,
                'predictions': predictions,
                'results': results,
                'total': len(datapoints),
                'timestamp': time.time(),
                'interrupted': True
            }
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f)
            volume.commit()
            break
        
        try:
            # Get repository files with timeout protection
            repo_files = _extract_repository_files(dp, data_dir, actual_index)
            
            # Generate context using SOTA method with timeout protection
            def timeout_handler(signum, frame):
                raise TimeoutError("Context retrieval timed out")
            
            # Set a 60-second timeout for context retrieval
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(60)
            
            try:
                context = retriever.retrieve_context(
                    prefix=dp.get('prefix', ''),
                    suffix=dp.get('suffix', ''),
                    repo_files=repo_files
                )
            finally:
                signal.alarm(0)  # Cancel the alarm
            
            predictions.append({"context": context})
            
            # Store processing info
            results.append({
                "datapoint": actual_index,
                "repo": dp.get('repo', ''),
                "files_found": len(repo_files),
                "context_length": len(context)
            })
            
            # Calculate completion percentage
            completion_percent = (actual_index + 1) / len(datapoints) * 100
            
            # Ultra-frequent checkpointing in final 2% to prevent data loss
            checkpoint_frequency = 1 if completion_percent >= 98.0 else 5
            
            if (actual_index + 1) % checkpoint_frequency == 0:
                try:
                    checkpoint_data = {
                        'last_processed': actual_index,
                        'predictions': predictions,
                        'results': results,
                        'total': len(datapoints),
                        'timestamp': time.time(),
                        'completion_percent': completion_percent
                    }
                    # Write to temporary file first, then rename for atomic operation
                    temp_checkpoint = checkpoint_file.with_suffix('.tmp')
                    with open(temp_checkpoint, 'w') as f:
                        json.dump(checkpoint_data, f)
                    temp_checkpoint.rename(checkpoint_file)
                    volume.commit()  # Commit to persistent storage
                    
                    # Extra logging near completion
                    if completion_percent >= 95.0:
                        logger.info(f"CRITICAL PHASE: {completion_percent:.1f}% complete ({actual_index + 1}/{len(datapoints)}) - Checkpoint saved")
                        
                except Exception as e:
                    logger.warning(f"Checkpoint save failed: {e}")
                
            # Progress logging with special attention to final phase
            if completion_percent >= 95.0:
                logger.info(f"FINAL PHASE: {actual_index + 1}/{len(datapoints)} datapoints ({completion_percent:.1f}%) - Processing datapoint {dp.get('repo', 'unknown')}")
            elif (actual_index + 1) % 25 == 0:
                logger.info(f"Processed {actual_index + 1}/{len(datapoints)} datapoints ({completion_percent:.1f}%)")
                
        except TimeoutError as e:
            logger.warning(f"Context retrieval timed out for datapoint {actual_index}, using fallback")
            fallback_context = f"{COMPETITION_CONFIG['file_separator']}timeout_fallback.py\n# Context retrieval timed out\n# Using minimal fallback context\npass"
            predictions.append({"context": fallback_context})
        except Exception as e:
            logger.error(f"Error processing datapoint {actual_index}: {str(e)}")
            fallback_context = f"{COMPETITION_CONFIG['file_separator']}error.py\n# Error: {str(e)}\npass"
            predictions.append({"context": fallback_context})
        
        # Memory optimization: clean up temp files periodically to prevent memory pressure
        if (actual_index + 1) % 10 == 0:
            try:
                _cleanup_temp_files(data_dir)
            except Exception as e:
                logger.warning(f"Temp cleanup failed: {e}")
        
        # Final phase optimization: more aggressive cleanup
        if completion_percent >= 95.0:
            try:
                import gc
                gc.collect()  # Force garbage collection in critical phase
            except Exception:
                pass
    
    # Final completion logging
    logger.info(f"COMPLETION: Processed all {len(predictions)}/{len(datapoints)} datapoints")
    
    # Save results with extra safety measures
    timestamp = int(time.time())
    results_file = data_dir / f"{language}-{stage}-sota-results-{timestamp}.json"
    predictions_file = data_dir / f"{language}-{stage}-sota-submission-{timestamp}.jsonl"
    
    # Save with retry logic for final write operations
    max_save_retries = 3
    for save_attempt in range(max_save_retries):
        try:
            # Save results
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Save predictions
            with jsonlines.open(predictions_file, 'w') as writer:
                for pred in predictions:
                    writer.write(pred)
            
            logger.info(f"SUCCESS: Files saved - {results_file.name}, {predictions_file.name}")
            break
            
        except Exception as e:
            logger.error(f"Save attempt {save_attempt + 1} failed: {e}")
            if save_attempt == max_save_retries - 1:
                # Last resort: save to checkpoint as backup
                logger.error("CRITICAL: Final save failed, preserving in checkpoint")
                emergency_checkpoint = {
                    'last_processed': len(datapoints) - 1,
                    'predictions': predictions,
                    'results': results,
                    'total': len(datapoints),
                    'timestamp': time.time(),
                    'emergency_save': True
                }
                with open(checkpoint_file.with_suffix('.emergency'), 'w') as f:
                    json.dump(emergency_checkpoint, f)
                raise
            time.sleep(1)
    
    # Clean up temporary files and checkpoint
    _cleanup_temp_files(data_dir)
    
    # Remove checkpoint file on successful completion
    checkpoint_file = data_dir / f"{language}-{stage}-checkpoint.json"
    if checkpoint_file.exists():
        checkpoint_file.unlink()
    
    volume.commit()
    
    summary = {
        "status": "success",
        "processed": len(predictions),
        "total_available": len(datapoints),
        "avg_context_length": sum(len(p["context"]) for p in predictions) / len(predictions),
        "results_file": str(results_file),
        "predictions_file": str(predictions_file),
        "timestamp": timestamp
    }
    
    logger.info(f"SOTA submission generated: {summary}")
    return summary

def _extract_repository_files(datapoint: Dict, data_dir: Path, index: int) -> List[Path]:
    """Extract repository files for a datapoint with timeout protection."""
    repo_info = datapoint.get('repo', '')
    repo_revision = datapoint.get('revision', '')
    
    # Repository ZIP file name
    repo_zip_name = f"{repo_info.replace('/', '__')}-{repo_revision}.zip"
    repo_zip_path = data_dir / repo_zip_name
    
    if not repo_zip_path.exists():
        logger.warning(f"Repository ZIP not found: {repo_zip_name}")
        return []
    
    # Extract to temporary directory
    temp_repo_dir = data_dir / "temp_repo" / f"repo_{index}"
    temp_repo_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Add timeout protection for ZIP extraction
        def extraction_timeout_handler(signum, frame):
            raise TimeoutError("ZIP extraction timed out")
        
        signal.signal(signal.SIGALRM, extraction_timeout_handler)
        signal.alarm(30)  # 30-second timeout for extraction
        
        try:
            with zipfile.ZipFile(repo_zip_path, 'r') as zip_ref:
                # Limit extraction to prevent huge repositories from hanging
                members = zip_ref.namelist()
                if len(members) > 1000:  # Limit to 1000 files max
                    logger.warning(f"Large repository {repo_zip_name} ({len(members)} files), limiting extraction")
                    members = members[:1000]
                
                for member in members:
                    try:
                        zip_ref.extract(member, temp_repo_dir)
                    except Exception as extract_error:
                        logger.warning(f"Failed to extract {member}: {extract_error}")
                        continue
        finally:
            signal.alarm(0)  # Cancel the alarm
        
        # Find relevant files with size limits
        extension = ".py" if datapoint.get('path', '').endswith('.py') else f".{datapoint.get('path', '').split('.')[-1]}"
        py_files = []
        
        for file_path in temp_repo_dir.rglob(f"*{extension}"):
            try:
                # Skip very large files that might cause issues
                if file_path.stat().st_size > 1024 * 1024:  # Skip files > 1MB
                    continue
                py_files.append(file_path)
                if len(py_files) >= 100:  # Limit to 100 files max
                    break
            except Exception:
                continue
        
        return py_files
        
    except TimeoutError:
        logger.warning(f"ZIP extraction timed out for {repo_zip_name}")
        return []
    except Exception as e:
        logger.error(f"Error extracting repository {repo_zip_name}: {e}")
        return []

def _cleanup_temp_files(data_dir: Path):
    """Clean up temporary extraction directories."""
    temp_base = data_dir / "temp_repo"
    if temp_base.exists():
        import shutil
        shutil.rmtree(temp_base, ignore_errors=True)

@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=MODAL_CONFIG["timeouts"]["validation"],
    memory=MODAL_CONFIG["memory"]["small"]
)
def validate_submission(predictions_file: str) -> Dict[str, Any]:
    """
    Validate competition submission format and quality.
    
    Args:
        predictions_file: Path to predictions JSONL file
        
    Returns:
        Validation results dictionary
    """
    logger.info(f"Validating submission: {predictions_file}")
    
    data_dir = Path("/data")
    pred_path = data_dir / predictions_file
    
    if not pred_path.exists():
        return {"status": "error", "message": f"Predictions file not found: {pred_path}"}
    
    validation = {
        "format_valid": True,
        "entry_count": 0,
        "context_stats": {},
        "issues": []
    }
    
    try:
        with jsonlines.open(pred_path, 'r') as reader:
            predictions = list(reader)
        
        validation["entry_count"] = len(predictions)
        context_lengths = []
        
        # Validate format
        for i, pred in enumerate(predictions):
            if not isinstance(pred, dict):
                validation["issues"].append(f"Entry {i}: Not a dictionary")
                validation["format_valid"] = False
                continue
            
            if "context" not in pred:
                validation["issues"].append(f"Entry {i}: Missing 'context' field")
                validation["format_valid"] = False
                continue
            
            context = pred["context"]
            if not isinstance(context, str):
                validation["issues"].append(f"Entry {i}: Context is not a string")
                validation["format_valid"] = False
                continue
            
            context_lengths.append(len(context))
            
            # Check for file separator
            if COMPETITION_CONFIG["file_separator"] not in context:
                validation["issues"].append(f"Entry {i}: No file separator found")
        
        # Context statistics
        if context_lengths:
            validation["context_stats"] = {
                "avg_length": sum(context_lengths) / len(context_lengths),
                "min_length": min(context_lengths),
                "max_length": max(context_lengths),
                "empty_contexts": sum(1 for length in context_lengths if length == 0)
            }
        
        logger.info(f"Validation complete: {validation}")
        return validation
        
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.local_entrypoint()
def main(
    action: str = "generate",
    stage: str = "public",
    language: str = "python",
    batch_size: Optional[int] = None
):
    """
    Main entry point for SOTA Context Collection Framework.
    
    Args:
        action: Action to perform (generate, validate)
        stage: Competition stage
        language: Programming language
        batch_size: Batch size for processing
    """
    
    print(f"SOTA Context Collection Framework")
    print(f"Action: {action}, Stage: {stage}, Language: {language}")
    
    if action == "generate":
        print(f"Generating SOTA submission...")
        
        max_retries = 5  # Increase retries for better resilience
        retry_delay = 30  # Start with 30 seconds
        
        for attempt in range(max_retries):
            try:
                print(f"Attempt {attempt + 1}/{max_retries}")
                result = generate_competition_submission.remote(stage, language, batch_size)
                print(f"Generation result: {result}")
                
                if result.get("status") == "success":
                    # Auto-validate
                    pred_file = Path(result["predictions_file"]).name
                    print(f"Auto-validating: {pred_file}")
                    validation = validate_submission.remote(pred_file)
                    print(f"Validation result: {validation}")
                    break
                elif result.get("status") == "error":
                    print(f"Generation failed: {result.get('message', 'Unknown error')}")
                    break
                    
            except modal.exception.InterruptedError as e:
                print(f"Modal worker interrupted on attempt {attempt + 1}/{max_retries}: {e}")
                print("This is likely due to worker preemption. Your progress is saved.")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 1.5, 300)  # Exponential backoff, max 5 minutes
                else:
                    print("Max retries exceeded due to worker preemption.")
                    print("Your progress is saved in checkpoint. Run the command again to resume.")
                    
            except (ConnectionError, TimeoutError) as e:
                print(f"Network/timeout error on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 1.2, 180)  # Slower exponential backoff
                else:
                    print("Max retries exceeded. Please check your network connection.")
                    print("You can resume from checkpoint by running the command again.")
                    
            except Exception as e:
                print(f"Unexpected error on attempt {attempt + 1}: {e}")
                if "preemption" in str(e).lower() or "interrupted" in str(e).lower():
                    print("Worker was preempted. Your progress is saved.")
                    if attempt < max_retries - 1:
                        print(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                print("Stopping due to unexpected error.")
                break
        
    elif action == "validate":
        pred_file = input("Enter predictions file name: ")
        result = validate_submission.remote(pred_file)
        print(f"Validation result: {result}")
        
    else:
        print(f"Unknown action: {action}")
        print("Available actions: generate, validate")

if __name__ == "__main__":
    print("SOTA Context Collection Framework")
    print("Usage: modal run src/core.py --action generate --stage public --language python")