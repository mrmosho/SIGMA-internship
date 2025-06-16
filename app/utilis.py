import json
import os
from typing import List, Dict, Any

def save_metadata(metadata: List[Dict[str, Any]], filepath: str = "metadata.json"):
    """Save metadata to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved {len(metadata)} metadata entries to {filepath}")

def load_metadata(filepath: str = "metadata.json") -> List[Dict[str, Any]]:
    """Load metadata from JSON file"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Metadata file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        metadata = json.load(f)
    print(f"Loaded {len(metadata)} metadata entries from {filepath}")
    return metadata

def check_files_exist(*filepaths):
    """Check if required files exist"""
    missing = [f for f in filepaths if not os.path.exists(f)]
    if missing:
        print(f"Missing files: {', '.join(missing)}")
        return False
    return True
