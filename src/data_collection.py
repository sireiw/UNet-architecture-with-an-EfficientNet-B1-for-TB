import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from .utils import validate_image, check_duplicate, standardize_filename, get_image_hash, DEBUG, quality_stats
from .config import Config

collection_stats = {class_name: {} for class_name in Config.CLASS_NAMES}
image_hashes = set()

def find_tbx11k_metadata():
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

def check_tbx11k_availability():
    metadata_path = find_tbx11k_metadata()
    if metadata_path and os.path.exists(metadata_path):
        df = pd.read_csv(metadata_path)
        tb_count = len(df[df['image_type'] == 'tb'])
        healthy_count = len(df[df['image_type'] == 'healthy'])
        return {'tb': tb_count, 'healthy': healthy_count, 'metadata_path': metadata_path}
    return {'tb': 0, 'healthy': 0, 'metadata_path': None}

def process_tbx11k_images(metadata_path, target_class, image_type_filter, dest_dir, limit):
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
            if test_img and os.path.exists(os.path.join(dir_path, test_img)):
                img_dir = dir_path
                break
    if not img_dir:
        return 0
    
    filtered_df = filtered_df.sample(frac=1, random_state=42).reset_index(drop=True)
    count = 0
    skipped = 0
    
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
                            img_hash = get_image_hash(dest_path)
                            if img_hash: image_hashes.add(img_hash)
                            count += 1
                            collection_stats[target_class].setdefault('TBX11K', 0)
                            collection_stats[target_class]['TBX11K'] += 1
                        except Exception as e: pass
            pbar.update(1)
    return count

def copy_images_enhanced(source_dir, dest_dir, file_pattern="*", limit=None, desc="Copying", source_name="Unknown", class_name="unknown"):
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    if not source_path.exists(): return 0
    count = 0
    skipped = 0
    image_extensions = ['.jpg', '.jpeg', '.png', '.dcm', '.dicom']
    all_files = list(source_path.glob(file_pattern))
    image_files = [f for f in all_files if f.suffix.lower() in image_extensions]
    if limit: image_files = image_files[:limit]
    
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
                        if img_hash: image_hashes.add(img_hash)
                        collection_stats[class_name].setdefault(source_name, 0)
                        collection_stats[class_name][source_name] += 1
                        count += 1
                    except Exception: quality_stats["Copy error"] = quality_stats.get("Copy error",0) + 1
            pbar.update(1)
    return count

