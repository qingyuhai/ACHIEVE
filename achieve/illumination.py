"""
Illumination Analysis Module
===========================

This module provides comprehensive illumination analysis capabilities
for detecting lighting conditions and uniformity in images.
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class IlluminationMetrics:
    """Container for illumination analysis results."""
    mean_brightness: float
    brightness_std: float
    brightness_variance: float
    illumination_uniformity: float
    low_light_ratio: float
    high_contrast_areas: float
    dominant_brightness_zone: str
    recommended_mode: str


class IlluminationAnalyzer:
    """
    Analyzes illumination conditions in images to determine optimal processing mode.
    
    This class provides methods to assess lighting conditions, detect uneven
    illumination, and recommend appropriate processing modes.
    """
    
    def __init__(self, 
                 low_light_threshold: float = 0.3,
                 high_contrast_threshold: float = 50,
                 uniformity_threshold: float = 0.7):
        """
        Initialize the IlluminationAnalyzer.
        
        Args:
            low_light_threshold: Threshold for detecting low light conditions (0-1)
            high_contrast_threshold: Threshold for high contrast detection
            uniformity_threshold: Threshold for illumination uniformity (0-1)
        """
        self.low_light_threshold = low_light_threshold
        self.high_contrast_threshold = high_contrast_threshold
        self.uniformity_threshold = uniformity_threshold
    
    def analyze_illumination(self, image: np.ndarray) -> IlluminationMetrics:
        """
        Perform comprehensive illumination analysis on an image.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            IlluminationMetrics containing analysis results
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Normalize to 0-1 range
        normalized = gray.astype(np.float32) / 255.0
        
        # Calculate basic brightness metrics
        mean_brightness = np.mean(normalized)
        brightness_std = np.std(normalized)
        brightness_variance = np.var(normalized)
        
        # Calculate illumination uniformity
        illumination_uniformity = self._calculate_uniformity(normalized)
        
        # Calculate low light ratio
        low_light_ratio = self._calculate_low_light_ratio(normalized)
        
        # Detect high contrast areas
        high_contrast_areas = self._detect_high_contrast_areas(gray)
        
        # Determine dominant brightness zone
        dominant_zone = self._get_dominant_brightness_zone(normalized)
        
        # Recommend processing mode
        recommended_mode = self._recommend_mode(
            mean_brightness, illumination_uniformity, low_light_ratio
        )
        
        return IlluminationMetrics(
            mean_brightness=mean_brightness,
            brightness_std=brightness_std,
            brightness_variance=brightness_variance,
            illumination_uniformity=illumination_uniformity,
            low_light_ratio=low_light_ratio,
            high_contrast_areas=high_contrast_areas,
            dominant_brightness_zone=dominant_zone,
            recommended_mode=recommended_mode
        )
    
    def _calculate_uniformity(self, normalized_image: np.ndarray) -> float:
        """Calculate illumination uniformity using coefficient of variation."""
        if np.mean(normalized_image) == 0:
            return 0.0
        cv = np.std(normalized_image) / np.mean(normalized_image)
        # Convert to uniformity score (higher = more uniform)
        uniformity = 1.0 / (1.0 + cv)
        return min(uniformity, 1.0)
    
    def _calculate_low_light_ratio(self, normalized_image: np.ndarray) -> float:
        """Calculate the ratio of pixels in low light conditions."""
        low_light_pixels = np.sum(normalized_image < self.low_light_threshold)
        total_pixels = normalized_image.size
        return low_light_pixels / total_pixels
    
    def _detect_high_contrast_areas(self, gray_image: np.ndarray) -> float:
        """Detect areas with high local contrast using gradient magnitude."""
        # Calculate gradients
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Calculate ratio of high contrast areas
        high_contrast_pixels = np.sum(gradient_magnitude > self.high_contrast_threshold)
        total_pixels = gradient_magnitude.size
        return high_contrast_pixels / total_pixels
    
    def _get_dominant_brightness_zone(self, normalized_image: np.ndarray) -> str:
        """Determine the dominant brightness zone of the image."""
        mean_brightness = np.mean(normalized_image)
        
        if mean_brightness < 0.3:
            return "dark"
        elif mean_brightness < 0.7:
            return "medium"
        else:
            return "bright"
    
    def _recommend_mode(self, 
                       mean_brightness: float,
                       uniformity: float,
                       low_light_ratio: float) -> str:
        """Recommend processing mode based on illumination analysis."""
        # Low light mode conditions
        if (mean_brightness < self.low_light_threshold or 
            low_light_ratio > 0.6 or 
            uniformity < self.uniformity_threshold):
            return "low_light"
        else:
            return "standard"
    
    def get_illumination_map(self, image: np.ndarray, 
                           window_size: int = 64) -> np.ndarray:
        """
        Generate an illumination map showing local brightness variations.
        
        Args:
            image: Input image
            window_size: Size of local analysis window
            
        Returns:
            Illumination map showing local brightness levels
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Create illumination map using local mean filtering
        kernel = np.ones((window_size, window_size), np.float32) / (window_size * window_size)
        illumination_map = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        
        # Normalize to 0-255 range
        illumination_map = np.clip(illumination_map, 0, 255).astype(np.uint8)
        
        return illumination_map
    
    def detect_shadow_regions(self, image: np.ndarray) -> np.ndarray:
        """
        Detect shadow regions in the image.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            Binary mask where shadows are marked as white (255)
        """
        if len(image.shape) == 3:
            # Convert to HSV for better shadow detection
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # Use saturation and value channels for shadow detection
            shadow_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 80))
        else:
            # For grayscale images, use simple thresholding
            shadow_mask = cv2.threshold(image, 60, 255, cv2.THRESH_BINARY_INV)[1]
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
        
        return shadow_mask