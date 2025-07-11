"""
Helper functions for tracking DAGs
"""
import os
import glob
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import sys
from pathlib import Path

# Add the root directory to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

def check_required_files(download_dir: str, file_patterns: List[str], max_age_hours: int = 24) -> Dict[str, Any]:
    """
    Checks whether files matching specified patterns exist in a directory and verifies that they are not older than a given maximum age.
    
    Parameters:
        download_dir (str): Path to the directory where files are checked.
        file_patterns (List[str]): List of filename patterns to search for.
        max_age_hours (int): Maximum allowed file age in hours.
    
    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'success': True if all required files are present and recent, False otherwise.
            - 'files_found': Mapping of patterns to the most recent matching file paths.
            - 'missing_patterns': List of patterns for which no files were found.
            - 'details': Additional information about each pattern's result.
    """
    result = {
        'success': True,
        'files_found': {},
        'missing_patterns': [],
        'details': {}
    }
    
    current_time = datetime.now()
    max_age = timedelta(hours=max_age_hours)
    
    for pattern in file_patterns:
        files = glob.glob(os.path.join(download_dir, pattern))
        
        if not files:
            result['success'] = False
            result['missing_patterns'].append(pattern)
            result['details'][pattern] = 'No files found'
            continue
            
        # Get the most recent file for this pattern
        latest_file = max(files, key=lambda x: os.path.getctime(x))
        file_age = current_time - datetime.fromtimestamp(os.path.getctime(latest_file))
        
        if file_age > max_age:
            result['success'] = False
            result['details'][pattern] = f'File too old: {file_age} (max age: {max_age})'
        else:
            result['files_found'][pattern] = latest_file
            result['details'][pattern] = f'File found: {os.path.basename(latest_file)} (age: {file_age})'
    
    return result

def get_latest_file_by_pattern(download_dir: str, pattern: str) -> Optional[str]:
    """
    Return the path to the most recently created file in a directory matching a given pattern.
    
    Parameters:
        download_dir (str): Directory to search for files.
        pattern (str): Filename pattern to match.
    
    Returns:
        str or None: Absolute path to the latest matching file, or None if no files are found.
    """
    files = glob.glob(os.path.join(download_dir, pattern))
    return max(files, key=lambda x: os.path.getctime(x)) if files else None

def setup_tracking_directories() -> str:
    """
    Create the 'data_lake/tracking' directory within the project if it does not exist and return its absolute path.
    
    Returns:
        The absolute path to the tracking download directory.
    """
    download_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', 'data_lake', 'tracking'
    ))
    
    if not os.path.exists(download_dir):
        os.makedirs(download_dir, exist_ok=True)
        
    return download_dir