"""
Test Image Enhancement Module
============================

Tests for the image enhancement functionality.
"""

import pytest
import numpy as np
import cv2

from achieve.enhancement import ImageEnhancer


class TestImageEnhancer:
    """Test cases for ImageEnhancer class."""
    
    def test_init(self):
        """Test enhancer initialization."""
        enhancer = ImageEnhancer()
        assert enhancer is not None
    
    def test_gamma_correction(self, sample_dark_image):
        """Test gamma correction."""
        enhancer = ImageEnhancer()
        
        # Test brightening (gamma < 1)
        brightened = enhancer.gamma_correction(sample_dark_image, 0.5)
        assert brightened.shape == sample_dark_image.shape
        assert np.mean(brightened) > np.mean(sample_dark_image)
        
        # Test darkening (gamma > 1)
        darkened = enhancer.gamma_correction(sample_dark_image, 1.5)
        assert darkened.shape == sample_dark_image.shape
        assert np.mean(darkened) < np.mean(sample_dark_image)
    
    def test_adjust_contrast(self, sample_bright_image):
        """Test contrast adjustment."""
        enhancer = ImageEnhancer()
        
        # Test contrast increase
        enhanced = enhancer.adjust_contrast(sample_bright_image, 1.2)
        assert enhanced.shape == sample_bright_image.shape
        assert enhanced.dtype == np.uint8
        
        # Test contrast decrease
        reduced = enhancer.adjust_contrast(sample_bright_image, 0.8)
        assert reduced.shape == sample_bright_image.shape
    
    def test_apply_clahe_color(self, sample_dark_image):
        """Test CLAHE on color image."""
        enhancer = ImageEnhancer()
        enhanced = enhancer.apply_clahe(sample_dark_image)
        
        assert enhanced.shape == sample_dark_image.shape
        assert enhanced.dtype == np.uint8
        # CLAHE should generally increase local contrast
        assert np.std(enhanced) >= np.std(sample_dark_image) * 0.8
    
    def test_apply_clahe_grayscale(self, sample_grayscale_image):
        """Test CLAHE on grayscale image."""
        enhancer = ImageEnhancer()
        enhanced = enhancer.apply_clahe(sample_grayscale_image)
        
        assert enhanced.shape == sample_grayscale_image.shape
        assert enhanced.dtype == np.uint8
    
    def test_sharpen_image_color(self, sample_bright_image):
        """Test image sharpening on color image."""
        enhancer = ImageEnhancer()
        sharpened = enhancer.sharpen_image(sample_bright_image, strength=0.5)
        
        assert sharpened.shape == sample_bright_image.shape
        assert sharpened.dtype == np.uint8
    
    def test_sharpen_image_grayscale(self, sample_grayscale_image):
        """Test image sharpening on grayscale image."""
        enhancer = ImageEnhancer()
        sharpened = enhancer.sharpen_image(sample_grayscale_image, strength=0.5)
        
        assert sharpened.shape == sample_grayscale_image.shape
        assert sharpened.dtype == np.uint8
    
    def test_denoise_methods(self, sample_dark_image):
        """Test different denoising methods."""
        enhancer = ImageEnhancer()
        
        # Test bilateral filtering
        denoised_bilateral = enhancer.denoise(sample_dark_image, method="bilateral")
        assert denoised_bilateral.shape == sample_dark_image.shape
        
        # Test Gaussian filtering
        denoised_gaussian = enhancer.denoise(sample_dark_image, method="gaussian")
        assert denoised_gaussian.shape == sample_dark_image.shape
        
        # Test median filtering  
        denoised_median = enhancer.denoise(sample_dark_image, method="median")
        assert denoised_median.shape == sample_dark_image.shape
        
        # Test invalid method
        denoised_invalid = enhancer.denoise(sample_dark_image, method="invalid")
        assert np.array_equal(denoised_invalid, sample_dark_image)
    
    def test_enhance_low_light(self, sample_dark_image):
        """Test low light enhancement."""
        enhancer = ImageEnhancer()
        enhanced = enhancer.enhance_low_light(sample_dark_image)
        
        assert enhanced.shape == sample_dark_image.shape
        assert enhanced.dtype == np.uint8
        # Enhanced image should generally be brighter
        assert np.mean(enhanced) > np.mean(sample_dark_image)
    
    def test_enhance_standard(self, sample_bright_image):
        """Test standard enhancement."""
        enhancer = ImageEnhancer()
        enhanced = enhancer.enhance_standard(sample_bright_image)
        
        assert enhanced.shape == sample_bright_image.shape
        assert enhanced.dtype == np.uint8
    
    def test_retinex_enhancement_color(self, sample_uneven_image):
        """Test Retinex enhancement on color image."""
        enhancer = ImageEnhancer()
        enhanced = enhancer.retinex_enhancement(sample_uneven_image)
        
        assert enhanced.shape == sample_uneven_image.shape
        assert enhanced.dtype == np.uint8
    
    def test_retinex_enhancement_grayscale(self, sample_grayscale_image):
        """Test Retinex enhancement on grayscale image."""
        enhancer = ImageEnhancer()
        enhanced = enhancer.retinex_enhancement(sample_grayscale_image)
        
        assert enhanced.shape == sample_grayscale_image.shape
        assert enhanced.dtype == np.uint8
    
    def test_correct_uneven_illumination(self, sample_uneven_image):
        """Test uneven illumination correction."""
        enhancer = ImageEnhancer()
        
        # Create a simple illumination map
        h, w = sample_uneven_image.shape[:2]
        illum_map = np.full((h, w), 128, dtype=np.uint8)
        
        corrected = enhancer.correct_uneven_illumination(sample_uneven_image, illum_map)
        
        assert corrected.shape == sample_uneven_image.shape
        assert corrected.dtype == np.uint8
    
    def test_enhancement_parameters(self, sample_dark_image):
        """Test enhancement with different parameters."""
        enhancer = ImageEnhancer()
        
        # Test different gamma values
        for gamma in [0.3, 0.5, 0.7, 1.0, 1.5]:
            corrected = enhancer.gamma_correction(sample_dark_image, gamma)
            assert corrected.shape == sample_dark_image.shape
            assert corrected.dtype == np.uint8
        
        # Test different contrast values
        for alpha in [0.5, 1.0, 1.5, 2.0]:
            adjusted = enhancer.adjust_contrast(sample_dark_image, alpha)
            assert adjusted.shape == sample_dark_image.shape
            assert adjusted.dtype == np.uint8
    
    def test_edge_cases(self):
        """Test edge cases."""
        enhancer = ImageEnhancer()
        
        # Test with all-zero image
        zero_image = np.zeros((100, 100, 3), dtype=np.uint8)
        enhanced = enhancer.enhance_low_light(zero_image)
        assert enhanced.shape == zero_image.shape
        
        # Test with all-white image
        white_image = np.full((100, 100, 3), 255, dtype=np.uint8)
        enhanced = enhancer.enhance_standard(white_image)
        assert enhanced.shape == white_image.shape
        
        # Test with single pixel image
        tiny_image = np.array([[[128, 128, 128]]], dtype=np.uint8)
        enhanced = enhancer.gamma_correction(tiny_image, 0.5)
        assert enhanced.shape == tiny_image.shape
    
    def test_chained_enhancements(self, sample_dark_image):
        """Test chaining multiple enhancements."""
        enhancer = ImageEnhancer()
        
        # Chain multiple operations
        enhanced = sample_dark_image.copy()
        enhanced = enhancer.gamma_correction(enhanced, 0.5)
        enhanced = enhancer.adjust_contrast(enhanced, 1.2)
        enhanced = enhancer.apply_clahe(enhanced)
        enhanced = enhancer.denoise(enhanced)
        
        assert enhanced.shape == sample_dark_image.shape
        assert enhanced.dtype == np.uint8