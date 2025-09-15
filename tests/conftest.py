"""
Test Configuration and Setup
============================

This module provides test configuration and setup utilities.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path


@pytest.fixture
def sample_bright_image():
    """Create a sample bright image for testing."""
    # Create a bright image (mean brightness ~0.8)
    image = np.random.randint(180, 255, (256, 256, 3), dtype=np.uint8)
    return image


@pytest.fixture  
def sample_dark_image():
    """Create a sample dark image for testing."""
    # Create a dark image (mean brightness ~0.2) 
    image = np.random.randint(10, 60, (256, 256, 3), dtype=np.uint8)
    return image


@pytest.fixture
def sample_uneven_image():
    """Create a sample image with uneven illumination."""
    # Create gradient illumination
    x = np.linspace(0, 1, 256)
    y = np.linspace(0, 1, 256)
    X, Y = np.meshgrid(x, y)
    
    # Create brightness gradient
    brightness = (X + Y) / 2  # 0 to 1 gradient
    brightness = (brightness * 200 + 50).astype(np.uint8)
    
    # Create 3-channel image
    image = np.stack([brightness] * 3, axis=-1)
    
    return image


@pytest.fixture
def sample_grayscale_image():
    """Create a sample grayscale image for testing."""
    image = np.random.randint(50, 200, (256, 256), dtype=np.uint8)
    return image


@pytest.fixture
def sample_high_contrast_image():
    """Create a sample high contrast image."""
    # Create checkerboard pattern for high contrast
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    
    # Create checkerboard
    for i in range(0, 256, 32):
        for j in range(0, 256, 32):
            if (i // 32 + j // 32) % 2 == 0:
                image[i:i+32, j:j+32] = 255
            else:
                image[i:i+32, j:j+32] = 0
    
    return image


def create_test_image_with_properties(brightness_level="medium", 
                                    uniformity="uniform",
                                    contrast="normal",
                                    size=(256, 256)):
    """
    Create test image with specific properties.
    
    Args:
        brightness_level: "dark", "medium", or "bright"
        uniformity: "uniform" or "uneven"  
        contrast: "low", "normal", or "high"
        size: Image dimensions
        
    Returns:
        Test image with specified properties
    """
    h, w = size
    
    # Base brightness levels
    brightness_ranges = {
        "dark": (10, 70),
        "medium": (80, 180),
        "bright": (200, 255)
    }
    
    min_val, max_val = brightness_ranges[brightness_level]
    
    if uniformity == "uniform":
        # Create uniform image
        base_value = (min_val + max_val) // 2
        image = np.full((h, w), base_value, dtype=np.uint8)
        
        # Add slight random noise
        noise = np.random.randint(-10, 11, (h, w))
        image = np.clip(image + noise, min_val, max_val).astype(np.uint8)
        
    else:  # uneven
        # Create gradient
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        X, Y = np.meshgrid(x, y)
        
        # Create illumination gradient
        gradient = (X + Y) / 2
        image = (gradient * (max_val - min_val) + min_val).astype(np.uint8)
    
    # Adjust contrast
    if contrast == "low":
        # Reduce contrast by compressing range
        mid = (min_val + max_val) // 2
        image = ((image - mid) * 0.5 + mid).astype(np.uint8)
    elif contrast == "high":
        # Increase contrast
        mid = (min_val + max_val) // 2
        image = np.clip((image - mid) * 1.5 + mid, 0, 255).astype(np.uint8)
    
    # Convert to 3-channel if needed
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    
    return image