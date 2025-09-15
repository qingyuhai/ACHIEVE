"""
ACHIEVE Framework - Main Interface
=================================

This module provides the main interface for the ACHIEVE dual-mode
illumination-aware framework.
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Union
import time
from pathlib import Path

from .illumination import IlluminationAnalyzer, IlluminationMetrics
from .enhancement import ImageEnhancer  
from .processing import DualModeProcessor, ProcessingMode, ProcessingConfig


class ACHIEVEFramework:
    """
    Main interface for the ACHIEVE dual-mode illumination-aware framework.
    
    This class provides a high-level interface for processing images with
    automatic adaptation to illumination conditions.
    """
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """
        Initialize the ACHIEVE framework.
        
        Args:
            config: Processing configuration parameters
        """
        self.config = config or ProcessingConfig()
        self.processor = DualModeProcessor(self.config)
        self.illumination_analyzer = IlluminationAnalyzer()
        self.enhancer = ImageEnhancer()
        
        # Statistics tracking
        self.processing_stats = {
            "total_images": 0,
            "low_light_count": 0,
            "standard_count": 0,
            "average_processing_time": 0.0,
            "total_processing_time": 0.0
        }
    
    def process_single_image(self, image: Union[np.ndarray, str, Path],
                           mode: ProcessingMode = ProcessingMode.AUTO,
                           save_path: Optional[str] = None,
                           return_analysis: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        Process a single image through the ACHIEVE framework.
        
        Args:
            image: Input image (array, file path, or Path object)
            mode: Processing mode to use
            save_path: Optional path to save the processed image
            return_analysis: Whether to return detailed analysis
            
        Returns:
            Processed image, optionally with analysis results
        """
        start_time = time.time()
        
        # Load image if path provided
        if isinstance(image, (str, Path)):
            img_array = cv2.imread(str(image))
            if img_array is None:
                raise ValueError(f"Could not load image from {image}")
        else:
            img_array = image.copy()
        
        # Process the image
        processed_img, metrics = self.processor.process_image(
            img_array, mode, return_metrics=True)
        
        # Update statistics
        processing_time = time.time() - start_time
        self._update_stats(metrics, processing_time)
        
        # Save if requested
        if save_path:
            cv2.imwrite(save_path, processed_img)
        
        if return_analysis:
            analysis = self._create_analysis_report(metrics, processing_time)
            return processed_img, analysis
        else:
            return processed_img
    
    def process_batch(self, image_paths: List[Union[str, Path]],
                     output_dir: Union[str, Path],
                     mode: ProcessingMode = ProcessingMode.AUTO,
                     preserve_structure: bool = True,
                     show_progress: bool = True) -> Dict:
        """
        Process a batch of images.
        
        Args:
            image_paths: List of input image paths
            output_dir: Output directory for processed images
            mode: Processing mode to use
            preserve_structure: Whether to preserve directory structure
            show_progress: Whether to show progress
            
        Returns:
            Dictionary with batch processing results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            "total_images": len(image_paths),
            "successful": 0,
            "failed": 0,
            "processing_times": [],
            "failed_images": []
        }
        
        for i, img_path in enumerate(image_paths):
            if show_progress:
                print(f"Processing {i+1}/{len(image_paths)}: {Path(img_path).name}")
            
            try:
                start_time = time.time()
                
                # Process image
                processed_img = self.process_single_image(img_path, mode)
                
                # Determine output path
                img_path = Path(img_path)
                if preserve_structure and img_path.is_absolute():
                    # Try to preserve relative structure
                    output_path = output_dir / img_path.name
                else:
                    output_path = output_dir / img_path.name
                
                # Save processed image
                cv2.imwrite(str(output_path), processed_img)
                
                processing_time = time.time() - start_time
                results["processing_times"].append(processing_time)
                results["successful"] += 1
                
            except Exception as e:
                results["failed"] += 1
                results["failed_images"].append({"path": str(img_path), "error": str(e)})
                if show_progress:
                    print(f"Failed to process {img_path}: {e}")
        
        # Calculate statistics
        if results["processing_times"]:
            results["average_time"] = np.mean(results["processing_times"])
            results["total_time"] = np.sum(results["processing_times"])
        
        return results
    
    def analyze_illumination_conditions(self, image: Union[np.ndarray, str, Path]) -> IlluminationMetrics:
        """
        Analyze illumination conditions in an image.
        
        Args:
            image: Input image (array or path)
            
        Returns:
            Illumination analysis results
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            img_array = cv2.imread(str(image))
            if img_array is None:
                raise ValueError(f"Could not load image from {image}")
        else:
            img_array = image.copy()
        
        return self.illumination_analyzer.analyze_illumination(img_array)
    
    def create_illumination_visualization(self, image: Union[np.ndarray, str, Path],
                                       save_path: Optional[str] = None) -> np.ndarray:
        """
        Create a visualization of illumination conditions.
        
        Args:
            image: Input image (array or path)
            save_path: Optional path to save visualization
            
        Returns:
            Illumination visualization image
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            img_array = cv2.imread(str(image))
            if img_array is None:
                raise ValueError(f"Could not load image from {image}")
        else:
            img_array = image.copy()
        
        # Get illumination map and metrics
        illum_map = self.illumination_analyzer.get_illumination_map(img_array)
        metrics = self.illumination_analyzer.analyze_illumination(img_array)
        shadow_mask = self.illumination_analyzer.detect_shadow_regions(img_array)
        
        # Create visualization
        h, w = img_array.shape[:2]
        viz_img = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        
        # Original image (top-left)
        if len(img_array.shape) == 3:
            viz_img[:h, :w] = img_array
        else:
            viz_img[:h, :w] = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        
        # Illumination map (top-right)
        illum_colored = cv2.applyColorMap(illum_map, cv2.COLORMAP_JET)
        viz_img[:h, w:] = illum_colored
        
        # Shadow regions (bottom-left)
        shadow_colored = cv2.cvtColor(shadow_mask, cv2.COLOR_GRAY2BGR)
        viz_img[h:, :w] = shadow_colored
        
        # Processing recommendation (bottom-right)
        info_img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Add text information
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        color = (255, 255, 255)
        thickness = 2
        
        text_lines = [
            f"Mode: {metrics.recommended_mode}",
            f"Brightness: {metrics.mean_brightness:.3f}",
            f"Uniformity: {metrics.illumination_uniformity:.3f}",
            f"Low-light ratio: {metrics.low_light_ratio:.3f}",
            f"Contrast areas: {metrics.high_contrast_areas:.3f}",
            f"Brightness zone: {metrics.dominant_brightness_zone}"
        ]
        
        for i, text in enumerate(text_lines):
            y = 30 + i * 30
            cv2.putText(info_img, text, (10, y), font, font_scale, color, thickness)
        
        viz_img[h:, w:] = info_img
        
        # Add labels
        cv2.putText(viz_img, "Original", (10, 25), font, 0.8, (0, 255, 0), 2)
        cv2.putText(viz_img, "Illumination Map", (w + 10, 25), font, 0.8, (0, 255, 0), 2)
        cv2.putText(viz_img, "Shadow Regions", (10, h + 25), font, 0.8, (0, 255, 0), 2)
        cv2.putText(viz_img, "Analysis", (w + 10, h + 25), font, 0.8, (0, 255, 0), 2)
        
        # Save if requested
        if save_path:
            cv2.imwrite(save_path, viz_img)
        
        return viz_img
    
    def compare_processing_modes(self, image: Union[np.ndarray, str, Path],
                               save_path: Optional[str] = None) -> np.ndarray:
        """
        Create a comparison of different processing modes.
        
        Args:
            image: Input image (array or path)
            save_path: Optional path to save comparison
            
        Returns:
            Comparison visualization image
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            img_array = cv2.imread(str(image))
            if img_array is None:
                raise ValueError(f"Could not load image from {image}")
        else:
            img_array = image.copy()
        
        # Process with different modes
        original = img_array.copy()
        low_light = self.processor.process_image(img_array, ProcessingMode.LOW_LIGHT)
        standard = self.processor.process_image(img_array, ProcessingMode.STANDARD)
        auto = self.processor.process_image(img_array, ProcessingMode.AUTO)
        
        # Create comparison grid
        h, w = img_array.shape[:2]
        comparison = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        
        # Ensure all images are color
        if len(original.shape) == 2:
            original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        if len(low_light.shape) == 2:
            low_light = cv2.cvtColor(low_light, cv2.COLOR_GRAY2BGR)
        if len(standard.shape) == 2:
            standard = cv2.cvtColor(standard, cv2.COLOR_GRAY2BGR)
        if len(auto.shape) == 2:
            auto = cv2.cvtColor(auto, cv2.COLOR_GRAY2BGR)
        
        # Arrange images
        comparison[:h, :w] = original
        comparison[:h, w:] = low_light
        comparison[h:, :w] = standard
        comparison[h:, w:] = auto
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        color = (0, 255, 0)
        thickness = 2
        
        cv2.putText(comparison, "Original", (10, 30), font, font_scale, color, thickness)
        cv2.putText(comparison, "Low-Light Mode", (w + 10, 30), font, font_scale, color, thickness)
        cv2.putText(comparison, "Standard Mode", (10, h + 30), font, font_scale, color, thickness)
        cv2.putText(comparison, "Auto Mode", (w + 10, h + 30), font, font_scale, color, thickness)
        
        # Save if requested
        if save_path:
            cv2.imwrite(save_path, comparison)
        
        return comparison
    
    def get_framework_stats(self) -> Dict:
        """Get framework processing statistics."""
        stats = self.processing_stats.copy()
        stats.update(self.processor.get_processing_info())
        return stats
    
    def _update_stats(self, metrics: IlluminationMetrics, processing_time: float):
        """Update processing statistics."""
        self.processing_stats["total_images"] += 1
        self.processing_stats["total_processing_time"] += processing_time
        
        if metrics.recommended_mode == "low_light":
            self.processing_stats["low_light_count"] += 1
        else:
            self.processing_stats["standard_count"] += 1
        
        # Update average processing time
        total_time = self.processing_stats["total_processing_time"]
        total_images = self.processing_stats["total_images"]
        self.processing_stats["average_processing_time"] = total_time / total_images
    
    def _create_analysis_report(self, metrics: IlluminationMetrics, 
                              processing_time: float) -> Dict:
        """Create detailed analysis report."""
        return {
            "illumination_metrics": metrics,
            "processing_time": processing_time,
            "framework_stats": self.get_framework_stats(),
            "recommended_settings": self._get_recommended_settings(metrics)
        }
    
    def _get_recommended_settings(self, metrics: IlluminationMetrics) -> Dict:
        """Get recommended settings based on image analysis."""
        recommendations = {}
        
        if metrics.recommended_mode == "low_light":
            recommendations.update({
                "gamma_correction": "Strong (0.4-0.6)" if metrics.mean_brightness < 0.2 else "Moderate (0.5-0.7)",
                "contrast_enhancement": "High (1.2-1.4)" if metrics.brightness_std < 20 else "Moderate (1.1-1.2)",
                "noise_reduction": "Essential" if metrics.mean_brightness < 0.15 else "Recommended",
                "clahe": "Aggressive (clip_limit=3.0)" if metrics.illumination_uniformity < 0.5 else "Moderate (clip_limit=2.0)"
            })
        else:
            recommendations.update({
                "contrast_enhancement": "Light (1.05-1.15)",
                "sharpening": "Recommended",
                "noise_reduction": "Optional" if metrics.brightness_std < 25 else "Light",
                "clahe": "Light (clip_limit=1.5)" if metrics.high_contrast_areas > 0.1 else "Skip"
            })
        
        return recommendations