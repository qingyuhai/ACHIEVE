"""
Test Dual-Mode Processing Module
===============================

Tests for the dual-mode processing functionality.
"""

import pytest
import numpy as np
import cv2

from achieve.processing import DualModeProcessor, ProcessingMode, ProcessingConfig
from achieve.illumination import IlluminationMetrics


class TestProcessingConfig:
    """Test cases for ProcessingConfig class."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = ProcessingConfig()
        
        assert config.low_light_gamma == 0.5
        assert config.low_light_contrast == 1.2
        assert config.standard_contrast == 1.1
        assert config.enable_clahe == True
        assert config.enable_retinex == False
        assert config.enable_denoising == True
        assert config.enable_sharpening == True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ProcessingConfig(
            low_light_gamma=0.4,
            low_light_contrast=1.3,
            standard_contrast=1.2,
            enable_clahe=False,
            enable_retinex=True
        )
        
        assert config.low_light_gamma == 0.4
        assert config.low_light_contrast == 1.3
        assert config.standard_contrast == 1.2
        assert config.enable_clahe == False
        assert config.enable_retinex == True


class TestDualModeProcessor:
    """Test cases for DualModeProcessor class."""
    
    def test_init_default(self):
        """Test processor initialization with default config."""
        processor = DualModeProcessor()
        
        assert processor.config is not None
        assert processor.illumination_analyzer is not None
        assert processor.enhancer is not None
        assert processor._last_metrics is None
        assert len(processor._mode_history) == 0
    
    def test_init_custom_config(self):
        """Test processor initialization with custom config."""
        config = ProcessingConfig(low_light_gamma=0.4)
        processor = DualModeProcessor(config)
        
        assert processor.config.low_light_gamma == 0.4
    
    def test_process_image_auto_mode(self, sample_dark_image, sample_bright_image):
        """Test image processing in auto mode."""
        processor = DualModeProcessor()
        
        # Test with dark image (should use low_light mode)
        processed_dark = processor.process_image(sample_dark_image, ProcessingMode.AUTO)
        assert processed_dark.shape == sample_dark_image.shape
        assert processed_dark.dtype == np.uint8
        
        # Test with bright image (should use standard mode)
        processed_bright = processor.process_image(sample_bright_image, ProcessingMode.AUTO)
        assert processed_bright.shape == sample_bright_image.shape
        assert processed_bright.dtype == np.uint8
    
    def test_process_image_low_light_mode(self, sample_dark_image):
        """Test image processing in low light mode."""
        processor = DualModeProcessor()
        processed = processor.process_image(sample_dark_image, ProcessingMode.LOW_LIGHT)
        
        assert processed.shape == sample_dark_image.shape
        assert processed.dtype == np.uint8
        # Processed image should generally be brighter
        assert np.mean(processed) >= np.mean(sample_dark_image)
    
    def test_process_image_standard_mode(self, sample_bright_image):
        """Test image processing in standard mode."""
        processor = DualModeProcessor()
        processed = processor.process_image(sample_bright_image, ProcessingMode.STANDARD)
        
        assert processed.shape == sample_bright_image.shape
        assert processed.dtype == np.uint8
    
    def test_process_image_with_metrics(self, sample_dark_image):
        """Test image processing with metrics return."""
        processor = DualModeProcessor()
        processed, metrics = processor.process_image(
            sample_dark_image, ProcessingMode.AUTO, return_metrics=True)
        
        assert processed.shape == sample_dark_image.shape
        assert isinstance(metrics, IlluminationMetrics)
        assert processor._last_metrics is not None
    
    def test_mode_history_tracking(self, sample_dark_image, sample_bright_image):
        """Test mode history tracking."""
        processor = DualModeProcessor()
        
        # Process multiple images
        processor.process_image(sample_dark_image, ProcessingMode.AUTO)
        processor.process_image(sample_bright_image, ProcessingMode.AUTO)
        processor.process_image(sample_dark_image, ProcessingMode.LOW_LIGHT)
        
        assert len(processor._mode_history) == 3
        assert "low_light" in processor._mode_history
        assert "standard" in processor._mode_history
    
    def test_batch_process(self, sample_dark_image, sample_bright_image):
        """Test batch processing."""
        processor = DualModeProcessor()
        images = [sample_dark_image, sample_bright_image, sample_dark_image]
        
        processed_images = processor.batch_process(images, ProcessingMode.AUTO)
        
        assert len(processed_images) == 3
        for processed in processed_images:
            assert processed.dtype == np.uint8
    
    def test_get_processing_info(self, sample_dark_image):
        """Test getting processing information."""
        processor = DualModeProcessor()
        
        # Initially should be empty
        info = processor.get_processing_info()
        assert info == {}
        
        # After processing should have information
        processor.process_image(sample_dark_image, ProcessingMode.AUTO)
        info = processor.get_processing_info()
        
        assert "last_metrics" in info
        assert "mode_history" in info
        assert "dominant_mode" in info
        assert "processing_stability" in info
    
    def test_adaptive_threshold_update(self, sample_dark_image):
        """Test adaptive threshold updates."""
        processor = DualModeProcessor()
        
        # Process image to get initial metrics
        processor.process_image(sample_dark_image, ProcessingMode.AUTO)
        initial_gamma = processor.config.low_light_gamma
        
        # Provide negative feedback
        processor.adaptive_threshold_update(0.3)
        assert processor.config.low_light_gamma != initial_gamma
        
        # Reset and provide positive feedback
        processor.reset_adaptive_parameters()
        processor.process_image(sample_dark_image, ProcessingMode.AUTO)
        processor.adaptive_threshold_update(0.9)
    
    def test_reset_adaptive_parameters(self, sample_dark_image):
        """Test resetting adaptive parameters."""
        processor = DualModeProcessor()
        
        # Process some images and modify parameters
        processor.process_image(sample_dark_image, ProcessingMode.AUTO)
        processor.adaptive_threshold_update(0.3)
        
        # Reset parameters
        processor.reset_adaptive_parameters()
        
        assert len(processor._mode_history) == 0
        assert processor._last_metrics is None
        assert processor.config.low_light_gamma == ProcessingConfig().low_light_gamma
    
    def test_processing_stability(self, sample_dark_image, sample_bright_image):
        """Test processing stability calculation."""
        processor = DualModeProcessor()
        
        # Process same type of images (should be stable)
        for _ in range(5):
            processor.process_image(sample_dark_image, ProcessingMode.AUTO)
        
        info = processor.get_processing_info()
        assert info["processing_stability"] > 0.8  # High stability
        
        # Mix different types (should be less stable)
        processor = DualModeProcessor()
        for _ in range(3):
            processor.process_image(sample_dark_image, ProcessingMode.AUTO)
            processor.process_image(sample_bright_image, ProcessingMode.AUTO)
        
        info = processor.get_processing_info()
        assert info["processing_stability"] < 0.8  # Lower stability
    
    def test_different_config_options(self, sample_dark_image):
        """Test processing with different configuration options."""
        # Test with CLAHE disabled
        config = ProcessingConfig(enable_clahe=False)
        processor = DualModeProcessor(config)
        processed = processor.process_image(sample_dark_image, ProcessingMode.LOW_LIGHT)
        assert processed.shape == sample_dark_image.shape
        
        # Test with Retinex enabled
        config = ProcessingConfig(enable_retinex=True)
        processor = DualModeProcessor(config)
        processed = processor.process_image(sample_dark_image, ProcessingMode.LOW_LIGHT)
        assert processed.shape == sample_dark_image.shape
        
        # Test with denoising disabled
        config = ProcessingConfig(enable_denoising=False)
        processor = DualModeProcessor(config)
        processed = processor.process_image(sample_dark_image, ProcessingMode.LOW_LIGHT)
        assert processed.shape == sample_dark_image.shape
    
    def test_edge_cases(self):
        """Test edge cases."""
        processor = DualModeProcessor()
        
        # Test with minimal size image
        tiny_image = np.full((10, 10, 3), 50, dtype=np.uint8)
        processed = processor.process_image(tiny_image, ProcessingMode.AUTO)
        assert processed.shape == tiny_image.shape
        
        # Test with single channel image
        gray_image = np.full((100, 100), 100, dtype=np.uint8)
        processed = processor.process_image(gray_image, ProcessingMode.AUTO)
        assert processed.shape == gray_image.shape