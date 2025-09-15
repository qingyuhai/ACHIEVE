"""
Utility Functions
=================

This module provides utility functions for the ACHIEVE framework.
"""

import numpy as np
import cv2
from typing import List, Tuple, Union
import os
from pathlib import Path


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Load an image from file path.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Loaded image as numpy array
        
    Raises:
        ValueError: If image cannot be loaded
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    return image


def save_image(image: np.ndarray, output_path: Union[str, Path]) -> bool:
    """
    Save an image to file.
    
    Args:
        image: Image array to save
        output_path: Output file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        return cv2.imwrite(str(output_path), image)
    except Exception:
        return False


def find_image_files(directory: Union[str, Path], 
                    extensions: List[str] = None) -> List[Path]:
    """
    Find all image files in a directory.
    
    Args:
        directory: Directory to search
        extensions: List of file extensions to include
        
    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    directory = Path(directory)
    image_files = []
    
    for ext in extensions:
        image_files.extend(directory.glob(f"*{ext}"))
        image_files.extend(directory.glob(f"*{ext.upper()}"))
    
    return sorted(image_files)


def create_image_grid(images: List[np.ndarray], 
                     grid_size: Tuple[int, int] = None,
                     padding: int = 10) -> np.ndarray:
    """
    Create a grid layout of images.
    
    Args:
        images: List of images to arrange
        grid_size: (rows, cols) for grid layout
        padding: Padding between images
        
    Returns:
        Grid image
    """
    if not images:
        return np.array([])
    
    # Determine grid size if not provided
    if grid_size is None:
        n_images = len(images)
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
        grid_size = (rows, cols)
    
    rows, cols = grid_size
    
    # Get dimensions of first image
    h, w = images[0].shape[:2]
    channels = len(images[0].shape)
    
    # Create grid image
    grid_h = rows * h + (rows - 1) * padding
    grid_w = cols * w + (cols - 1) * padding
    
    if channels == 3:
        grid_image = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    else:
        grid_image = np.zeros((grid_h, grid_w), dtype=np.uint8)
    
    # Place images in grid
    for i, image in enumerate(images):
        if i >= rows * cols:
            break
        
        row = i // cols
        col = i % cols
        
        start_y = row * (h + padding)
        end_y = start_y + h
        start_x = col * (w + padding)
        end_x = start_x + w
        
        # Resize image if needed
        if image.shape[:2] != (h, w):
            image = cv2.resize(image, (w, h))
        
        grid_image[start_y:end_y, start_x:end_x] = image
    
    return grid_image


def calculate_image_metrics(image: np.ndarray) -> dict:
    """
    Calculate basic image quality metrics.
    
    Args:
        image: Input image
        
    Returns:
        Dictionary of metrics
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Calculate metrics
    metrics = {
        'mean': np.mean(gray),
        'std': np.std(gray),
        'min': np.min(gray),
        'max': np.max(gray),
        'range': np.max(gray) - np.min(gray)
    }
    
    # Calculate contrast using RMS contrast
    metrics['rms_contrast'] = np.sqrt(np.mean((gray - metrics['mean']) ** 2))
    
    # Calculate sharpness using Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    metrics['sharpness'] = laplacian.var()
    
    # Calculate entropy
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    hist = hist + 1e-7  # Add small value to avoid log(0)
    hist_norm = hist / np.sum(hist)
    metrics['entropy'] = -np.sum(hist_norm * np.log2(hist_norm))
    
    return metrics


def normalize_image(image: np.ndarray, 
                   target_range: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """
    Normalize image to target range.
    
    Args:
        image: Input image
        target_range: Target (min, max) range
        
    Returns:
        Normalized image
    """
    image_float = image.astype(np.float32)
    
    # Get current range
    img_min = np.min(image_float)
    img_max = np.max(image_float)
    img_range = img_max - img_min
    
    if img_range == 0:
        return np.full_like(image_float, target_range[0])
    
    # Normalize to target range
    target_min, target_max = target_range
    target_range_size = target_max - target_min
    
    normalized = (image_float - img_min) / img_range * target_range_size + target_min
    
    return normalized


def resize_with_aspect_ratio(image: np.ndarray, 
                           target_size: Tuple[int, int],
                           maintain_aspect: bool = True) -> np.ndarray:
    """
    Resize image with optional aspect ratio maintenance.
    
    Args:
        image: Input image
        target_size: Target (width, height)
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized image
    """
    target_w, target_h = target_size
    
    if not maintain_aspect:
        return cv2.resize(image, (target_w, target_h))
    
    h, w = image.shape[:2]
    
    # Calculate scaling factors
    scale_w = target_w / w
    scale_h = target_h / h
    scale = min(scale_w, scale_h)
    
    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h))
    
    # Create output image with target size
    if len(image.shape) == 3:
        output = np.zeros((target_h, target_w, 3), dtype=image.dtype)
    else:
        output = np.zeros((target_h, target_w), dtype=image.dtype)
    
    # Center the resized image
    start_y = (target_h - new_h) // 2
    start_x = (target_w - new_w) // 2
    output[start_y:start_y + new_h, start_x:start_x + new_w] = resized
    
    return output


def create_histogram(image: np.ndarray, 
                    bins: int = 256,
                    show_channels: bool = True) -> np.ndarray:
    """
    Create histogram visualization of image.
    
    Args:
        image: Input image
        bins: Number of histogram bins
        show_channels: Whether to show individual channels
        
    Returns:
        Histogram image
    """
    # Create histogram plot
    hist_image = np.ones((400, 512, 3), dtype=np.uint8) * 255
    
    if len(image.shape) == 3 and show_channels:
        # Color image - show individual channels
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR
        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
            hist = cv2.normalize(hist, hist, 0, 400, cv2.NORM_MINMAX)
            
            for j in range(bins):
                cv2.line(hist_image,
                        (j * 2, 400),
                        (j * 2, 400 - int(hist[j])),
                        color, 2)
    else:
        # Grayscale or single channel
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        hist = cv2.calcHist([gray], [0], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist, 0, 400, cv2.NORM_MINMAX)
        
        for i in range(bins):
            cv2.line(hist_image,
                    (i * 2, 400),
                    (i * 2, 400 - int(hist[i])),
                    (0, 0, 0), 2)
    
    return hist_image


def blend_images(image1: np.ndarray, 
                image2: np.ndarray,
                alpha: float = 0.5) -> np.ndarray:
    """
    Blend two images together.
    
    Args:
        image1: First image
        image2: Second image
        alpha: Blending factor (0-1)
        
    Returns:
        Blended image
    """
    # Ensure images have same size
    if image1.shape != image2.shape:
        h, w = image1.shape[:2]
        image2 = cv2.resize(image2, (w, h))
    
    # Blend images
    blended = cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0)
    
    return blended


def add_text_overlay(image: np.ndarray, 
                    text: str,
                    position: Tuple[int, int] = (10, 30),
                    font_scale: float = 1.0,
                    color: Tuple[int, int, int] = (255, 255, 255),
                    thickness: int = 2) -> np.ndarray:
    """
    Add text overlay to image.
    
    Args:
        image: Input image
        text: Text to add
        position: Text position (x, y)
        font_scale: Font size scale
        color: Text color (B, G, R)
        thickness: Text thickness
        
    Returns:
        Image with text overlay
    """
    result = image.copy()
    
    # Add text
    cv2.putText(result, text, position, 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                color, thickness)
    
    return result