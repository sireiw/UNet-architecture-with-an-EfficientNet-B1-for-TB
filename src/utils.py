import hashlib
from pathlib import Path
from datetime import datetime
import pydicom
from PIL import Image
import numpy as np
import cv2
from typing import Tuple, Optional, Set, Dict

from .config import Config

DEBUG: bool = False
quality_stats: Dict[str, int] = {}
image_hashes: Set[str] = set()
MIN_IMAGE_SIZE: Tuple[int, int] = (224, 224)
MAX_IMAGE_SIZE: Tuple[int, int] = (4096, 4096)
CHECK_DUPLICATES: bool = True

def get_image_hash(image_path: Path) -> Optional[str]:
    """Generate MD5 hash of image content for duplicate detection.
    
    Args:
        image_path (str): The absolute or relative path to the image file.
        
    Returns:
        Optional[str]: The MD5 hexadecimal string if successful, else None.
    """
    try:
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        if DEBUG:
            print(f"Error hashing {image_path}: {e}")
        return None

def validate_image(image_path: Path) -> Tuple[bool, str]:
    """Validate image quality and dimension properties.
    
    Args:
        image_path (str): Path indicating a '.dcm' dicom or standard PIL image.
        
    Returns:
        Tuple[bool, str]: A boolean indicating validity and a descriptive status string.
    """
    global quality_stats
    try:
        if image_path.suffix.lower() in ['.dcm', '.dicom']:
            dicom = pydicom.dcmread(str(image_path))
            width = dicom.Columns
            height = dicom.Rows
        else:
            img = Image.open(image_path)
            width, height = img.size
            img.verify()
            
        if width < MIN_IMAGE_SIZE[0] or height < MIN_IMAGE_SIZE[1]:
            quality_stats["Too small"] = quality_stats.get("Too small", 0) + 1
            return False, "Image too small"
        if width > MAX_IMAGE_SIZE[0] or height > MAX_IMAGE_SIZE[1]:
            quality_stats["Too large"] = quality_stats.get("Too large", 0) + 1
            return False, "Image too large"
            
        return True, "Valid"
        
    except Exception as e:
        quality_stats["Corrupted"] = quality_stats.get("Corrupted", 0) + 1
        return False, f"Error: {str(e)}"

def check_duplicate(image_path: Path) -> bool:
    """Check if image is a duplicate based on file hash.
    
    Args:
        image_path (str): Path to the image file to check.
        
    Returns:
        bool: True if the file content hash has been seen previously, False otherwise.
    """
    if not CHECK_DUPLICATES:
        return False
        
    img_hash = get_image_hash(image_path)
    if img_hash and img_hash in image_hashes:
        quality_stats["Duplicate"] = quality_stats.get("Duplicate", 0) + 1
        return True
    return False

def standardize_filename(source_name: str, original_filename: str, class_name: str) -> str:
    """Create standardized filename including original source metadata.
    
    Args:
        source_name (str): The originating source collection (e.g. 'NIH', 'TBX11K').
        original_filename (str): The initial filename string.
        class_name (str): The clinical condition classified (e.g. 'tuberculosis').
        
    Returns:
        str: A reformatted filename combining source, base name, and a unique timestamp.
    """
    base_name = Path(original_filename).stem
    extension = Path(original_filename).suffix
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
    return f"{source_name}_{base_name}_{timestamp}{extension}"

def read_image_safe(image_path: Path, grayscale: bool = False) -> Optional[np.ndarray]:
    """Safely construct a numpy array from an image with redundancy fallbacks.
    
    Attempts to read via Pillow, falling back to OpenCV if corrupted explicitly.
    
    Args:
        image_path (str): The path to the image asset.
        grayscale (bool, optional): Whether to cast the image to 1-channel grayscale. Defaults to False.
        
    Returns:
        Optional[np.ndarray]: The constructed image array if successful, None otherwise.
    """
    try:
        if grayscale:
            img = Image.open(image_path).convert('L')
            img_array = np.array(img)
        else:
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img)
        
        if img_array.size == 0:
            raise ValueError("Empty image")
        
        return img_array
        
    except Exception as pil_error:
        try:
            if grayscale:
                img_array = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            else:
                img_array = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if img_array is not None:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            
            if img_array is None or img_array.size == 0:
                raise ValueError("OpenCV returned None or empty image")
                
            return img_array
            
        except Exception:
            return None
