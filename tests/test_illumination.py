"""
Test Illumination Analysis Module
=================================

Tests for the illumination analysis functionality.
"""

import pytest
import numpy as np
import cv2

from achieve.illumination import IlluminationAnalyzer, IlluminationMetrics
from tests.conftest import create_test_image_with_properties


class TestIlluminationAnalyzer:
    """Test cases for IlluminationAnalyzer class."""
    
    def test_init(self):
        """Test analyzer initialization."""
        analyzer = IlluminationAnalyzer()
        assert analyzer.low_light_threshold == 0.3
        assert analyzer.high_contrast_threshold == 50
        assert analyzer.uniformity_threshold == 0.7
        
        # Test with custom parameters
        analyzer = IlluminationAnalyzer(
            low_light_threshold=0.4,
            high_contrast_threshold=60,
            uniformity_threshold=0.8
        )
        assert analyzer.low_light_threshold == 0.4
        assert analyzer.high_contrast_threshold == 60
        assert analyzer.uniformity_threshold == 0.8
    
    def test_analyze_bright_image(self, sample_bright_image):
        """Test analysis of bright image."""
        analyzer = IlluminationAnalyzer()
        metrics = analyzer.analyze_illumination(sample_bright_image)
        
        assert isinstance(metrics, IlluminationMetrics)
        assert metrics.mean_brightness > 0.6  # Should be bright
        assert metrics.dominant_brightness_zone == "bright"
        assert metrics.recommended_mode == "standard"
    
    def test_analyze_dark_image(self, sample_dark_image):
        """Test analysis of dark image."""
        analyzer = IlluminationAnalyzer()
        metrics = analyzer.analyze_illumination(sample_dark_image)
        
        assert isinstance(metrics, IlluminationMetrics)
        assert metrics.mean_brightness < 0.4  # Should be dark
        assert metrics.dominant_brightness_zone == "dark"
        assert metrics.recommended_mode == "low_light"
    
    def test_analyze_grayscale_image(self, sample_grayscale_image):
        """Test analysis of grayscale image."""
        analyzer = IlluminationAnalyzer()
        metrics = analyzer.analyze_illumination(sample_grayscale_image)
        
        assert isinstance(metrics, IlluminationMetrics)
        assert 0 <= metrics.mean_brightness <= 1
        assert metrics.brightness_std >= 0
        assert 0 <= metrics.illumination_uniformity <= 1
    
    def test_analyze_uneven_image(self, sample_uneven_image):
        """Test analysis of unevenly illuminated image."""
        analyzer = IlluminationAnalyzer()
        metrics = analyzer.analyze_illumination(sample_uneven_image)
        
        # Uneven image should have lower uniformity
        assert metrics.illumination_uniformity < 0.8
        assert metrics.brightness_variance > 0
    
    def test_uniformity_calculation(self):
        """Test illumination uniformity calculation."""
        analyzer = IlluminationAnalyzer()
        
        # Uniform image should have high uniformity
        uniform_image = create_test_image_with_properties(
            brightness_level="medium", uniformity="uniform")
        metrics_uniform = analyzer.analyze_illumination(uniform_image)
        
        # Uneven image should have lower uniformity
        uneven_image = create_test_image_with_properties(
            brightness_level="medium", uniformity="uneven")
        metrics_uneven = analyzer.analyze_illumination(uneven_image)
        
        assert metrics_uniform.illumination_uniformity > metrics_uneven.illumination_uniformity
    
    def test_low_light_ratio(self):
        """Test low light ratio calculation."""
        analyzer = IlluminationAnalyzer()
        
        # Dark image should have high low-light ratio
        dark_image = create_test_image_with_properties(brightness_level="dark")
        metrics_dark = analyzer.analyze_illumination(dark_image)
        
        # Bright image should have low low-light ratio
        bright_image = create_test_image_with_properties(brightness_level="bright")
        metrics_bright = analyzer.analyze_illumination(bright_image)
        
        assert metrics_dark.low_light_ratio > metrics_bright.low_light_ratio
    
    def test_mode_recommendation(self):
        """Test processing mode recommendation."""
        analyzer = IlluminationAnalyzer()
        
        # Dark image should recommend low_light mode
        dark_image = create_test_image_with_properties(brightness_level="dark")
        metrics_dark = analyzer.analyze_illumination(dark_image)
        assert metrics_dark.recommended_mode == "low_light"
        
        # Bright uniform image should recommend standard mode
        bright_image = create_test_image_with_properties(brightness_level="bright")
        metrics_bright = analyzer.analyze_illumination(bright_image)
        assert metrics_bright.recommended_mode == "standard"
    
    def test_get_illumination_map(self, sample_uneven_image):
        """Test illumination map generation."""
        analyzer = IlluminationAnalyzer()
        illum_map = analyzer.get_illumination_map(sample_uneven_image)
        
        assert illum_map.shape[:2] == sample_uneven_image.shape[:2]
        assert illum_map.dtype == np.uint8
        assert 0 <= np.min(illum_map) and np.max(illum_map) <= 255
    
    def test_detect_shadow_regions(self, sample_dark_image):
        """Test shadow region detection."""
        analyzer = IlluminationAnalyzer()
        shadow_mask = analyzer.detect_shadow_regions(sample_dark_image)
        
        assert shadow_mask.shape[:2] == sample_dark_image.shape[:2]
        assert shadow_mask.dtype == np.uint8
        assert set(np.unique(shadow_mask)).issubset({0, 255})
    
    def test_high_contrast_detection(self, sample_high_contrast_image):
        """Test high contrast area detection."""
        analyzer = IlluminationAnalyzer()
        metrics = analyzer.analyze_illumination(sample_high_contrast_image)
        
        # High contrast image should have significant high contrast areas
        assert metrics.high_contrast_areas > 0.1
    
    def test_edge_cases(self):
        """Test edge cases."""
        analyzer = IlluminationAnalyzer()
        
        # All black image
        black_image = np.zeros((100, 100, 3), dtype=np.uint8)
        metrics_black = analyzer.analyze_illumination(black_image)
        assert metrics_black.mean_brightness == 0
        assert metrics_black.recommended_mode == "low_light"
        
        # All white image
        white_image = np.full((100, 100, 3), 255, dtype=np.uint8)
        metrics_white = analyzer.analyze_illumination(white_image)
        assert metrics_white.mean_brightness == 1.0
        assert metrics_white.dominant_brightness_zone == "bright"
        
    def test_different_window_sizes(self, sample_uneven_image):
        """Test illumination map with different window sizes."""
        analyzer = IlluminationAnalyzer()
        
        # Test different window sizes
        for window_size in [32, 64, 128]:
            illum_map = analyzer.get_illumination_map(sample_uneven_image, window_size)
            assert illum_map.shape[:2] == sample_uneven_image.shape[:2]