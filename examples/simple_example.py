#!/usr/bin/env python3
"""
Simple ACHIEVE Framework Usage Example
=====================================

This is a basic example showing how to use the ACHIEVE framework
to process images with different illumination conditions.
"""

import sys
import numpy as np
import cv2
from pathlib import Path

# Add parent directory to path for importing achieve
sys.path.insert(0, str(Path(__file__).parent.parent))

from achieve import ACHIEVEFramework, ProcessingMode


def main():
    """Simple usage example."""
    print("Simple ACHIEVE Framework Example")
    print("================================")
    
    # Initialize the framework
    framework = ACHIEVEFramework()
    
    # Create a sample dark image for demonstration
    print("Creating sample dark image...")
    dark_image = np.random.randint(10, 60, (300, 300, 3), dtype=np.uint8)
    
    # Process the image using auto mode (recommended)
    print("Processing image with AUTO mode...")
    processed_image = framework.process_single_image(
        dark_image, 
        mode=ProcessingMode.AUTO
    )
    
    # Save results
    cv2.imwrite("sample_dark_original.jpg", dark_image)
    cv2.imwrite("sample_dark_processed.jpg", processed_image)
    
    print("Images saved:")
    print("  - sample_dark_original.jpg (original)")
    print("  - sample_dark_processed.jpg (processed)")
    
    # Analyze illumination conditions
    print("\nAnalyzing illumination conditions...")
    metrics = framework.analyze_illumination_conditions(dark_image)
    
    print(f"Analysis Results:")
    print(f"  Mean brightness: {metrics.mean_brightness:.3f}")
    print(f"  Illumination uniformity: {metrics.illumination_uniformity:.3f}")
    print(f"  Recommended mode: {metrics.recommended_mode}")
    print(f"  Dominant brightness zone: {metrics.dominant_brightness_zone}")
    
    # Create comparison of different modes
    print("\nCreating mode comparison...")
    comparison = framework.compare_processing_modes(dark_image)
    cv2.imwrite("mode_comparison.jpg", comparison)
    print("Mode comparison saved as: mode_comparison.jpg")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()