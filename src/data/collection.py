"""
Data collection module for handling the Kaggle TBX11K dataset and assembling
generic image collections into standard folders.
"""
import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional, Union, List

from src.utils import validate_image, check_duplicate, standardize_filename, get_image_hash, quality_stats
from src.config import Config

collection_stats: Dict[str, Dict[str, int]] = {class_name: {} for class_name in Config.CLASS_NAMES}
image_hashes: set = set()

def find_tbx11k_metadata() -> Optional[str]:
    """Locate the TBX11K metadata CSV file across known Kaggle paths.
    
    Returns:
        Optional[str]: The absolute path to the metadata.csv if found, else None.
    """
    possible_paths = [
        "/kaggle/input/tbx11k-simplified/tbx11k-simplified/data.csv",
        "/kaggle/input/tbx11k/metadata.csv",
        "/kaggle/input/tbx11k-simplified/metadata.csv",
        "/kaggle/input/tbx11k/tbx11k_metadata.csv"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
            
    tbx_dirs = ["/kaggle/input/tbx11k-simplified", "/kaggle/input/tbx11k"]
    for base_dir in tbx_dirs:
        if os.path.exists(base_dir):
            for root, _, files in os.walk(base_dir):
                for file in files:
                    if 'metadata' in file.lower() and file.endswith('.csv'):
                        return os.path.join(root, file)
    return None

def check_tbx11k_availability() -> Dict[str, Union[int, str, None]]:
    """Verify TBX11K dataset accessibility and count samples.
    
    Returns:
        Dict[str, Union[int, str, None]]: A dict containing counts for 'tb' and 'healthy', 
        and the 'metadata_path'.
    """
    metadata_path = find_tbx11k_metadata()
    if metadata_path and os.path.exists(metadata_path):
        df = pd.read_csv(metadata_path)
        tb_count = len(df[df['image_type'] == 'tb'])
        healthy_count = len(df[df['image_type'] == 'healthy'])
        return {'tb': tb_count, 'healthy': healthy_count, 'metadata_path': metadata_path}
    return {'tb': 0, 'healthy': 0, 'metadata_path': None}

def process_tbx11k_images(metadata_path: str, target_class: str, image_type_filter: str, dest_dir: str, limit: int) -> int:
    """Filter and copy TBX11K images according to conditions and limits.
    
    Args:
        metadata_path (str): The valid filepath to the dataset labels CSV.
        target_class (str): The final unified class name to map into (e.g. 'tuberculosis').
        image_type_filter (str): The internal dataframe filter string matching `image_type` (e.g. 'tb').
        dest_dir (str): The folder destination where validated images should be standardized to.
        limit (int): Cap on the maximum amount of files to pull to preserve balance.
        
    Returns:
        int: Total number of valid images successfully transferred.
    """
    if not metadata_path or not os.path.exists(metadata_path):
        return 0
    tbx_df = pd.read_csv(metadata_path)
    filtered_df = tbx_df[tbx_df['image_type'] == image_type_filter]
    
    metadata_dir = os.path.dirname(metadata_path)
    possible_img_dirs = [
        os.path.join(metadata_dir, "images"),
        os.path.join(os.path.dirname(metadata_dir), "images"),
        metadata_dir
    ]
    img_dir = None
    for dir_path in possible_img_dirs:
        if os.path.exists(dir_path):
            test_img = filtered_df.iloc[0]['fname'] if len(filtered_df) > 0 else None
            # Handle potential subdirectories in fname
            if test_img and os.path.exists(os.path.join(dir_path, test_img)):
                img_dir = dir_path
                break
    if not img_dir:
        return 0
    
    # Shuffle consistently
    filtered_df = filtered_df.sample(frac=1, random_state=42).reset_index(drop=True)
    count = 0
    skipped = 0
    
    os.makedirs(dest_dir, exist_ok=True)
    
    with tqdm(total=min(len(filtered_df), limit), desc=f"TBX11K {image_type_filter}", unit="files") as pbar:
        for idx, row in filtered_df.iterrows():
            if count >= limit: break
            img_name = row['fname']
            img_path = Path(os.path.join(img_dir, img_name))
            
            if img_path.exists():
                is_valid, _ = validate_image(img_path)
                if not is_valid or check_duplicate(img_path):
                    skipped += 1
                else:
                    new_filename = standardize_filename("TBX11K", img_path.name, target_class)
                    dest_path = os.path.join(dest_dir, new_filename)
                    if not os.path.exists(dest_path):
                        try:
                            shutil.copy2(img_path, dest_path)
                            img_hash = get_image_hash(Path(dest_path))
                            if img_hash: 
                                image_hashes.add(img_hash)
                            count += 1
                            collection_stats[target_class].setdefault('TBX11K', 0)
                            collection_stats[target_class]['TBX11K'] += 1
                        except Exception: 
                            pass
            pbar.update(1)
    return count

def copy_images_enhanced(source_dir: str, dest_dir: str, file_pattern: str = "*", limit: Optional[int] = None, desc: str = "Copying", source_name: str = "Unknown", class_name: str = "unknown") -> int:
    """Safely copy massive directory subsets using deduplication and hash verification.
    
    Args:
        source_dir (str): Relative or absolute extraction folder.
        dest_dir (str): Unified class collection directory.
        file_pattern (str, optional): Wildcard extension matches. Defaults to "*".
        limit (Optional[int], optional): Maximum cap. Defaults to None.
        desc (str, optional): Tqdm progress bar string. Defaults to "Copying".
        source_name (str, optional): Logging provenance metadata tag. Defaults to "Unknown".
        class_name (str, optional): Classification assignment tag. Defaults to "unknown".
        
    Returns:
        int: The number of valid images correctly transferred.
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    if not source_path.exists(): 
        return 0
        
    count = 0
    skipped = 0
    image_extensions = ['.jpg', '.jpeg', '.png', '.dcm', '.dicom']
    all_files = list(source_path.glob(file_pattern))
    image_files = [f for f in all_files if f.suffix.lower() in image_extensions]
    
    if limit: 
        image_files = image_files[:limit]
        
    os.makedirs(dest_dir, exist_ok=True)
    
    with tqdm(total=len(image_files), desc=desc, unit="files") as pbar:
        for img_file in image_files:
            is_valid, _ = validate_image(img_file)
            if not is_valid or check_duplicate(img_file):
                skipped += 1
            else:
                new_filename = standardize_filename(source_name, img_file.name, class_name)
                dest_file = dest_path / new_filename
                if not dest_file.exists():
                    try:
                        shutil.copy2(img_file, dest_file)
                        img_hash = get_image_hash(dest_file)
                        if img_hash: 
                            image_hashes.add(img_hash)
                        collection_stats[class_name].setdefault(source_name, 0)
                        collection_stats[class_name][source_name] += 1
                        count += 1
                    except Exception: 
                        quality_stats["Copy error"] = quality_stats.get("Copy error",0) + 1
            pbar.update(1)
            
    return count
