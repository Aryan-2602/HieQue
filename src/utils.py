"""
Utility functions for the HieQue framework
"""

import os
import re
import json
import hashlib
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import numpy as np
from datetime import datetime

from .logger import get_logger

logger = get_logger(__name__)

def load_environment_variables(env_file: str = ".env") -> Dict[str, str]:
    """
    Load environment variables from .env file
    
    Args:
        env_file: Path to .env file
        
    Returns:
        Dictionary of environment variables
    """
    env_vars = {}
    
    if os.path.exists(env_file):
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip()
                        os.environ[key.strip()] = value.strip()
            
            logger.info(f"Loaded {len(env_vars)} environment variables from {env_file}")
        except Exception as e:
            logger.error(f"Failed to load environment variables: {e}")
    
    return env_vars

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe file operations
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(' .')
    # Limit length
    if len(sanitized) > 200:
        sanitized = sanitized[:200]
    
    return sanitized

def generate_document_id(content: str, metadata: Dict[str, Any] = None) -> str:
    """
    Generate a unique document ID based on content and metadata
    
    Args:
        content: Document content
        metadata: Document metadata
        
    Returns:
        Unique document ID
    """
    # Create hash from content
    content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:16]
    
    # Add metadata to hash if available
    if metadata:
        metadata_str = json.dumps(metadata, sort_keys=True)
        metadata_hash = hashlib.md5(metadata_str.encode('utf-8')).hexdigest()[:8]
        return f"{content_hash}_{metadata_hash}"
    
    return content_hash

def chunk_text(text: str, 
               chunk_size: int = 1000, 
               overlap: int = 200,
               separator: str = "\n") -> List[str]:
    """
    Split text into overlapping chunks
    
    Args:
        text: Text to chunk
        chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks
        separator: Character to use as chunk boundary
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to find a good break point
        if end < len(text):
            # Look for separator near the end
            last_sep = text.rfind(separator, start, end)
            if last_sep > start + chunk_size // 2:
                end = last_sep + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks

def extract_metadata_from_filename(filename: str) -> Dict[str, Any]:
    """
    Extract metadata from filename
    
    Args:
        filename: Filename to extract metadata from
        
    Returns:
        Dictionary of extracted metadata
    """
    metadata = {}
    
    # Extract file extension
    file_ext = Path(filename).suffix.lower()
    metadata['file_type'] = file_ext
    
    # Extract filename without extension
    name_without_ext = Path(filename).stem
    metadata['filename'] = name_without_ext
    
    # Try to extract chapter/page information
    chapter_match = re.search(r'chapter[_\s]*(\d+)', name_without_ext.lower())
    if chapter_match:
        metadata['chapter'] = int(chapter_match.group(1))
    
    page_match = re.search(r'page[_\s]*(\d+)', name_without_ext.lower())
    if page_match:
        metadata['page'] = int(page_match.group(1))
    
    # Extract date if present
    date_match = re.search(r'(\d{4})[_\s]*(\d{2})[_\s]*(\d{2})', name_without_ext)
    if date_match:
        metadata['date'] = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
    
    return metadata

def calculate_text_similarity(text1: str, text2: str, method: str = "cosine") -> float:
    """
    Calculate similarity between two text strings
    
    Args:
        text1: First text string
        text2: Second text string
        method: Similarity method ('cosine', 'jaccard', 'levenshtein')
        
    Returns:
        Similarity score between 0 and 1
    """
    if method == "cosine":
        return _cosine_similarity(text1, text2)
    elif method == "jaccard":
        return _jaccard_similarity(text1, text2)
    elif method == "levenshtein":
        return _levenshtein_similarity(text1, text2)
    else:
        raise ValueError(f"Unsupported similarity method: {method}")

def _cosine_similarity(text1: str, text2: str) -> float:
    """Calculate cosine similarity between two texts"""
    # Simple character-based cosine similarity
    chars1 = set(text1.lower())
    chars2 = set(text2.lower())
    
    intersection = len(chars1.intersection(chars2))
    union = len(chars1.union(chars2))
    
    if union == 0:
        return 0.0
    
    return intersection / union

def _jaccard_similarity(text1: str, text2: str) -> float:
    """Calculate Jaccard similarity between two texts"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    if union == 0:
        return 0.0
    
    return intersection / union

def _levenshtein_similarity(text1: str, text2: str) -> float:
    """Calculate Levenshtein distance-based similarity"""
    def levenshtein_distance(s1, s2):
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    distance = levenshtein_distance(text1, text2)
    max_len = max(len(text1), len(text2))
    
    if max_len == 0:
        return 1.0
    
    return 1 - (distance / max_len)

def normalize_text(text: str) -> str:
    """
    Normalize text for better processing
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}]', '', text)
    
    # Normalize quotes
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"[''']", "'", text)
    
    # Normalize dashes
    text = re.sub(r'[–—]', '-', text)
    
    return text.strip()

def create_timestamp() -> str:
    """Create a timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_results_to_file(results: List[Dict[str, Any]], 
                        filename: str, 
                        format: str = "json") -> str:
    """
    Save search results to a file
    
    Args:
        results: List of result dictionaries
        filename: Output filename
        format: Output format ('json', 'csv', 'txt')
        
    Returns:
        Path to saved file
    """
    filename = sanitize_filename(filename)
    
    if format == "json":
        filepath = f"{filename}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    elif format == "csv":
        import csv
        filepath = f"{filename}.csv"
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
    
    elif format == "txt":
        filepath = f"{filename}.txt"
        with open(filepath, 'w', encoding='utf-8') as f:
            for i, result in enumerate(results, 1):
                f.write(f"Result {i}:\n")
                for key, value in result.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
    
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Results saved to {filepath}")
    return filepath

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return {}

def save_config(config: Dict[str, Any], config_path: str) -> bool:
    """
    Save configuration to JSON file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logger.info(f"Configuration saved to {config_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save configuration: {e}")
        return False

def get_file_size_mb(filepath: str) -> float:
    """Get file size in megabytes"""
    try:
        size_bytes = os.path.getsize(filepath)
        return size_bytes / (1024 * 1024)
    except OSError:
        return 0.0

def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def validate_api_key(api_key: str) -> bool:
    """
    Validate API key format
    
    Args:
        api_key: API key to validate
        
    Returns:
        True if valid format, False otherwise
    """
    if not api_key:
        return False
    
    # Basic validation for OpenAI API key
    if api_key.startswith('sk-') and len(api_key) > 20:
        return True
    
    # Add other API key format validations as needed
    return True
