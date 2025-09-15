"""
Dual-Mode Processing Pipeline
============================

This module implements the core dual-mode processing pipeline that
automatically adapts processing strategies based on illumination conditions.
"""

import numpy as np
import cv2
from typing import Dict, Optional, Tuple, Union
from enum import Enum

from .illumination import IlluminationAnalyzer, IlluminationMetrics
from .enhancement import ImageEnhancer


class ProcessingMode(Enum):
    """Enumeration of available processing modes."""
    LOW_LIGHT = "low_light"
    STANDARD = "standard"
    AUTO = "auto"


class ProcessingConfig:
    """Configuration parameters for dual-mode processing."""
    
    def __init__(self,
                 low_light_gamma: float = 0.5,
                 low_light_contrast: float = 1.2,
                 standard_contrast: float = 1.1,
                 enable_clahe: bool = True,
                 enable_retinex: bool = False,
                 enable_denoising: bool = True,
                 enable_sharpening: bool = True):
        """
        Initialize processing configuration.
        
        Args:
            low_light_gamma: Gamma correction for low-light mode
            low_light_contrast: Contrast enhancement for low-light mode
            standard_contrast: Contrast enhancement for standard mode
            enable_clahe: Whether to apply CLAHE enhancement
            enable_retinex: Whether to apply Retinex enhancement
            enable_denoising: Whether to apply denoising
            enable_sharpening: Whether to apply sharpening
        """
        self.low_light_gamma = low_light_gamma
        self.low_light_contrast = low_light_contrast
        self.standard_contrast = standard_contrast
        self.enable_clahe = enable_clahe
        self.enable_retinex = enable_retinex
        self.enable_denoising = enable_denoising
        self.enable_sharpening = enable_sharpening


class DualModeProcessor:
    """
    Dual-mode image processor that adapts processing strategies
    based on illumination conditions.
    """
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """
        Initialize the dual-mode processor.
        
        Args:
            config: Processing configuration parameters
        """
        self.config = config or ProcessingConfig()
        self.illumination_analyzer = IlluminationAnalyzer()
        self.enhancer = ImageEnhancer()
        self._last_metrics = None
        self._mode_history = []
    
    def process_image(self, image: np.ndarray,
                     mode: ProcessingMode = ProcessingMode.AUTO,
                     return_metrics: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, IlluminationMetrics]]:
        """
        Process an image using dual-mode pipeline.
        
        Args:
            image: Input image
            mode: Processing mode to use
            return_metrics: Whether to return illumination metrics
            
        Returns:
            Processed image, optionally with illumination metrics
        """
        # Analyze illumination conditions
        metrics = self.illumination_analyzer.analyze_illumination(image)
        self._last_metrics = metrics
        
        # Determine processing mode
        if mode == ProcessingMode.AUTO:
            selected_mode = ProcessingMode(metrics.recommended_mode)
        else:
            selected_mode = mode
        
        # Track mode history for adaptive behavior
        self._mode_history.append(selected_mode.value)
        if len(self._mode_history) > 10:
            self._mode_history.pop(0)
        
        # Apply appropriate processing pipeline
        if selected_mode == ProcessingMode.LOW_LIGHT:
            processed = self._process_low_light_mode(image, metrics)
        else:
            processed = self._process_standard_mode(image, metrics)
        
        if return_metrics:
            return processed, metrics
        else:
            return processed
    
    def _process_low_light_mode(self, image: np.ndarray,
                               metrics: IlluminationMetrics) -> np.ndarray:
        """
        Process image using low-light optimized pipeline.
        
        Args:
            image: Input image
            metrics: Illumination analysis results
            
        Returns:
            Processed image
        """
        processed = image.copy()
        
        # Stage 1: Illumination correction if uneven
        if metrics.illumination_uniformity < 0.6:
            illum_map = self.illumination_analyzer.get_illumination_map(image)
            processed = self.enhancer.correct_uneven_illumination(processed, illum_map)
        
        # Stage 2: Retinex enhancement for illumination invariance
        if self.config.enable_retinex:
            processed = self.enhancer.retinex_enhancement(processed)
        
        # Stage 3: Brightness enhancement
        processed = self.enhancer.gamma_correction(
            processed, self.config.low_light_gamma)
        
        # Stage 4: Contrast enhancement
        processed = self.enhancer.adjust_contrast(
            processed, self.config.low_light_contrast)
        
        # Stage 5: Adaptive histogram equalization
        if self.config.enable_clahe:
            processed = self.enhancer.apply_clahe(processed, clip_limit=3.0)
        
        # Stage 6: Noise reduction (important after enhancement)
        if self.config.enable_denoising:
            processed = self.enhancer.denoise(processed, method="bilateral")
        
        # Stage 7: Mild sharpening to recover details
        if self.config.enable_sharpening and metrics.mean_brightness > 0.1:
            processed = self.enhancer.sharpen_image(processed, strength=0.3)
        
        return processed
    
    def _process_standard_mode(self, image: np.ndarray,
                              metrics: IlluminationMetrics) -> np.ndarray:
        """
        Process image using standard optimized pipeline.
        
        Args:
            image: Input image
            metrics: Illumination analysis results
            
        Returns:
            Processed image
        """
        processed = image.copy()
        
        # Stage 1: Minor illumination correction if needed
        if metrics.illumination_uniformity < 0.8:
            illum_map = self.illumination_analyzer.get_illumination_map(
                image, window_size=32)
            processed = self.enhancer.correct_uneven_illumination(processed, illum_map)
        
        # Stage 2: Mild contrast enhancement
        processed = self.enhancer.adjust_contrast(
            processed, self.config.standard_contrast)
        
        # Stage 3: Selective CLAHE for high contrast areas
        if self.config.enable_clahe and metrics.high_contrast_areas > 0.1:
            processed = self.enhancer.apply_clahe(processed, clip_limit=1.5)
        
        # Stage 4: Sharpening for detail enhancement
        if self.config.enable_sharpening:
            processed = self.enhancer.sharpen_image(processed, strength=0.5)
        
        # Stage 5: Light denoising if needed
        if self.config.enable_denoising and metrics.brightness_std > 30:
            processed = self.enhancer.denoise(processed, method="bilateral")
        
        return processed
    
    def get_processing_info(self) -> Dict:
        """
        Get information about the last processing operation.
        
        Returns:
            Dictionary containing processing information
        """
        if self._last_metrics is None:
            return {}
        
        return {
            "last_metrics": self._last_metrics,
            "mode_history": self._mode_history.copy(),
            "dominant_mode": self._get_dominant_mode(),
            "processing_stability": self._calculate_stability()
        }
    
    def _get_dominant_mode(self) -> str:
        """Get the most frequently used processing mode."""
        if not self._mode_history:
            return "unknown"
        
        mode_counts = {}
        for mode in self._mode_history:
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        
        return max(mode_counts, key=mode_counts.get)
    
    def _calculate_stability(self) -> float:
        """Calculate processing mode stability (0-1, higher is more stable)."""
        if len(self._mode_history) < 2:
            return 1.0
        
        changes = sum(1 for i in range(1, len(self._mode_history))
                     if self._mode_history[i] != self._mode_history[i-1])
        
        stability = 1.0 - (changes / (len(self._mode_history) - 1))
        return stability
    
    def batch_process(self, images: list,
                     mode: ProcessingMode = ProcessingMode.AUTO,
                     show_progress: bool = False) -> list:
        """
        Process a batch of images.
        
        Args:
            images: List of input images
            mode: Processing mode to use
            show_progress: Whether to show processing progress
            
        Returns:
            List of processed images
        """
        processed_images = []
        
        for i, image in enumerate(images):
            if show_progress:
                print(f"Processing image {i+1}/{len(images)}")
            
            processed = self.process_image(image, mode)
            processed_images.append(processed)
        
        return processed_images
    
    def adaptive_threshold_update(self, feedback_score: float):
        """
        Update processing thresholds based on user feedback.
        
        Args:
            feedback_score: Quality feedback score (0-1, higher is better)
        """
        if self._last_metrics is None:
            return
        
        # Simple adaptive mechanism - adjust thresholds based on feedback
        if feedback_score < 0.5:  # Poor quality
            # Make processing more aggressive
            if self._last_metrics.recommended_mode == "low_light":
                self.config.low_light_gamma = max(0.3, self.config.low_light_gamma - 0.05)
                self.config.low_light_contrast = min(1.5, self.config.low_light_contrast + 0.1)
        elif feedback_score > 0.8:  # Good quality
            # Make processing less aggressive to avoid over-enhancement
            if self._last_metrics.recommended_mode == "low_light":
                self.config.low_light_gamma = min(0.7, self.config.low_light_gamma + 0.02)
                self.config.low_light_contrast = max(1.0, self.config.low_light_contrast - 0.05)
    
    def reset_adaptive_parameters(self):
        """Reset adaptive parameters to default values."""
        self.config = ProcessingConfig()
        self._mode_history.clear()
        self._last_metrics = None