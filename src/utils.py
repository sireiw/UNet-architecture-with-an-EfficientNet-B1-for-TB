import hashlib
from pathlib import Path
from datetime import datetime
import pydicom
from PIL import Image
import numpy as np
import cv2
from .config import Config

DEBUG = False
quality_stats = {}
image_hashes = set()
MIN_IMAGE_SIZE = (224, 224)
MAX_IMAGE_SIZE = (4096, 4096)
CHECK_DUPLICATES = True

def get_image_hash(image_path):
    """Generate MD5 hash of image content for duplicate detection"""
    try:
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        if DEBUG:
            print(f"Error hashing {image_path}: {e}")
        return None

def validate_image(image_path):
    """Validate image quality and properties"""
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

def check_duplicate(image_path):
    """Check if image is a duplicate"""
    if not CHECK_DUPLICATES:
        return False
        
    img_hash = get_image_hash(image_path)
    if img_hash and img_hash in image_hashes:
        quality_stats["Duplicate"] = quality_stats.get("Duplicate", 0) + 1
        return True
    return False

def standardize_filename(source_name, original_filename, class_name):
    """Create standardized filename including source information"""
    base_name = Path(original_filename).stem
    extension = Path(original_filename).suffix
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
    return f"{source_name}_{base_name}_{timestamp}{extension}"

def read_image_safe(image_path, grayscale=False):
    """Read image safely with better error handling"""
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
