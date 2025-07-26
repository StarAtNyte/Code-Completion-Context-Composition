"""
Utility functions for the SOTA Context Collection Framework
"""

import modal
import os
import zipfile
import jsonlines
from pathlib import Path
import logging
from typing import Dict, Any, List

import sys
import os

# Handle imports for both local and Modal environments
try:
    from src.config import MODAL_CONFIG
except ImportError:
    try:
        from config import MODAL_CONFIG
    except ImportError:
        # Fallback configuration
        MODAL_CONFIG = {
            "app_name": "code-completion-sota",
            "volume_name": "code-completion-data",
            "image_python_version": "3.11",
            "dependencies": ["jsonlines==4.0.0"],
            "system_packages": ["unzip", "git", "curl"],
            "timeouts": {"data_processing": 3600, "context_generation": 7200, "validation": 300},
            "memory": {"small": 4096, "medium": 16384, "large": 32768}
        }

logger = logging.getLogger(__name__)

# Modal setup for utilities
app = modal.App("sota-utils")
volume = modal.Volume.from_name(MODAL_CONFIG["volume_name"], create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version=MODAL_CONFIG["image_python_version"])
    .pip_install(["jsonlines==4.0.0"])
    .apt_install(["unzip"])
)

@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=MODAL_CONFIG["timeouts"]["data_processing"],
    memory=MODAL_CONFIG["memory"]["medium"]
)
def extract_competition_data(stage: str = "public", language: str = "python") -> Dict[str, Any]:
    """
    Extract uploaded competition data and prepare it for processing.
    
    Args:
        stage: Competition stage (public, practice, etc.)
        language: Programming language (python, kotlin)
        
    Returns:
        Extraction results dictionary
    """
    logger.info(f"Extracting {language}-{stage} competition data...")
    
    data_dir = Path("/data")
    
    # File names
    jsonl_file = f"{language}-{stage}.jsonl"
    zip_file = f"{language}-{stage}.zip"
    
    jsonl_path = data_dir / jsonl_file
    zip_path = data_dir / zip_file
    
    results = {"status": "success", "files": []}
    
    # Check if files exist
    if not jsonl_path.exists():
        return {"status": "error", "message": f"JSONL file not found: {jsonl_file}"}
    
    if not zip_path.exists():
        return {"status": "error", "message": f"ZIP file not found: {zip_file}"}
    
    try:
        # Verify JSONL file
        logger.info(f"Verifying {jsonl_file}...")
        with jsonlines.open(jsonl_path, 'r') as reader:
            datapoints = list(reader)
        
        logger.info(f"Loaded {len(datapoints)} datapoints")
        results["datapoints"] = len(datapoints)
        results["files"].append(jsonl_file)
        
        # Show sample datapoint
        if datapoints:
            sample = datapoints[0]
            logger.info(f"Sample keys: {list(sample.keys())}")
        
        # Extract ZIP file if not already extracted
        repo_dir_name = f"repositories-{language}-{stage}"
        repo_dir = data_dir / repo_dir_name
        
        if not repo_dir.exists():
            logger.info(f"Extracting {zip_file}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            logger.info(f"Extracted {zip_file}")
        
        results["files"].append(zip_file)
        
        # Verify extracted repositories
        if repo_dir.exists():
            repositories = list(repo_dir.iterdir())
            repo_count = len([r for r in repositories if r.is_dir()])
            logger.info(f"Found {repo_count} repositories")
            results["repositories"] = repo_count
        
        volume.commit()
        
        summary = {
            "stage": stage,
            "language": language,
            "datapoints": results.get("datapoints", 0),
            "repositories": results.get("repositories", 0),
            "extracted_files": results["files"]
        }
        
        logger.info(f"Extraction complete: {summary}")
        return summary
        
    except Exception as e:
        logger.error(f"Extraction failed: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=300,  # 5 minutes
    memory=MODAL_CONFIG["memory"]["small"]
)
def list_volume_contents() -> Dict[str, List]:
    """
    List all contents in the Modal volume.
    
    Returns:
        Dictionary with categorized file listings
    """
    logger.info("Listing volume contents...")
    
    data_dir = Path("/data")
    
    contents = {
        "jsonl_files": [],
        "zip_files": [],
        "repository_dirs": [],
        "prediction_files": [],
        "other_files": []
    }
    
    if data_dir.exists():
        for item in data_dir.iterdir():
            if item.is_file():
                file_info = {
                    "name": item.name,
                    "size": item.stat().st_size
                }
                
                if item.suffix == ".jsonl":
                    if "submission" in item.name or "prediction" in item.name:
                        contents["prediction_files"].append(file_info)
                    else:
                        contents["jsonl_files"].append(file_info)
                elif item.suffix == ".zip":
                    contents["zip_files"].append(file_info)
                else:
                    contents["other_files"].append(file_info)
                    
            elif item.is_dir() and "repositories" in item.name:
                repo_count = len(list(item.iterdir()))
                contents["repository_dirs"].append({
                    "name": item.name,
                    "repo_count": repo_count
                })
    
    logger.info(f"Found: {len(contents['jsonl_files'])} JSONL, {len(contents['zip_files'])} ZIP, {len(contents['repository_dirs'])} repo dirs, {len(contents['prediction_files'])} predictions")
    return contents

@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=300,  # 5 minutes
    memory=MODAL_CONFIG["memory"]["small"]
)
def cleanup_temp_files() -> Dict[str, Any]:
    """
    Clean up temporary files and directories.
    
    Returns:
        Cleanup results
    """
    logger.info("Cleaning up temporary files...")
    
    data_dir = Path("/data")
    cleaned = []
    
    # Clean up temp directories
    temp_dirs = ["temp_repo", "temp_extract"]
    for temp_dir in temp_dirs:
        temp_path = data_dir / temp_dir
        if temp_path.exists():
            import shutil
            shutil.rmtree(temp_path, ignore_errors=True)
            cleaned.append(temp_dir)
            logger.info(f"Cleaned: {temp_dir}")
    
    # Clean up old result files (keep only latest 5)
    result_files = list(data_dir.glob("*-results-*.json"))
    if len(result_files) > 5:
        result_files.sort(key=lambda x: x.stat().st_mtime)
        for old_file in result_files[:-5]:
            old_file.unlink()
            cleaned.append(old_file.name)
            logger.info(f"Cleaned old result: {old_file.name}")
    
    volume.commit()
    
    return {
        "status": "success",
        "cleaned_items": cleaned,
        "message": f"Cleaned {len(cleaned)} items"
    }

@app.local_entrypoint()
def main(action: str = "list"):
    """
    Utility functions entry point.
    
    Args:
        action: Action to perform (extract, list, cleanup)
    """
    
    print(f"SOTA Framework Utilities - Action: {action}")
    
    if action == "extract":
        stage = input("Enter stage (public/practice): ") or "public"
        language = input("Enter language (python/kotlin): ") or "python"
        result = extract_competition_data.remote(stage, language)
        print(f"Extraction result: {result}")
        
    elif action == "list":
        result = list_volume_contents.remote()
        print(f"Volume contents:")
        for category, items in result.items():
            if items:
                print(f"  {category}: {len(items)} items")
                for item in items[:3]:  # Show first 3 items
                    if isinstance(item, dict):
                        if "size" in item:
                            print(f"    - {item['name']} ({item['size']} bytes)")
                        else:
                            print(f"    - {item['name']} ({item.get('repo_count', 0)} repos)")
        
    elif action == "cleanup":
        result = cleanup_temp_files.remote()
        print(f"Cleanup result: {result}")
        
    else:
        print(f"Unknown action: {action}")
        print("Available actions: extract, list, cleanup")

if __name__ == "__main__":
    print("SOTA Framework Utilities")
    print("Usage: modal run src/utils.py --action list")