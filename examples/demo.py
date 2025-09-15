#!/usr/bin/env python3
"""
ACHIEVE Framework Demo
=====================

This script demonstrates the capabilities of the ACHIEVE dual-mode 
illumination-aware framework.
"""

import sys
import numpy as np
import cv2
from pathlib import Path

# Add parent directory to path for importing achieve
sys.path.insert(0, str(Path(__file__).parent.parent))

from achieve import ACHIEVEFramework, ProcessingMode, ProcessingConfig


def create_demo_images():
    """Create demo images with different illumination conditions."""
    print("Creating demo images...")
    
    # Create output directory
    demo_dir = Path("demo_images")
    demo_dir.mkdir(exist_ok=True)
    
    # 1. Bright uniform image
    bright_image = np.random.randint(180, 255, (512, 512, 3), dtype=np.uint8)
    cv2.imwrite(str(demo_dir / "bright_uniform.jpg"), bright_image)
    
    # 2. Dark uniform image
    dark_image = np.random.randint(10, 60, (512, 512, 3), dtype=np.uint8)
    cv2.imwrite(str(demo_dir / "dark_uniform.jpg"), dark_image)
    
    # 3. Gradient illumination (uneven lighting)
    x = np.linspace(0, 1, 512)
    y = np.linspace(0, 1, 512)
    X, Y = np.meshgrid(x, y)
    gradient = (X + Y) / 2
    gradient_image = (gradient * 200 + 55).astype(np.uint8)
    gradient_image = np.stack([gradient_image] * 3, axis=-1)
    cv2.imwrite(str(demo_dir / "gradient_illumination.jpg"), gradient_image)
    
    # 4. Mixed lighting (bright center, dark edges)
    h, w = 512, 512
    center_x, center_y = w // 2, h // 2
    y_coords, x_coords = np.ogrid[:h, :w]
    distance = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    max_distance = np.sqrt(center_x**2 + center_y**2)
    brightness = 255 - (distance / max_distance * 200)
    brightness = np.clip(brightness, 30, 255).astype(np.uint8)
    mixed_image = np.stack([brightness] * 3, axis=-1)
    cv2.imwrite(str(demo_dir / "mixed_lighting.jpg"), mixed_image)
    
    # 5. High contrast image
    contrast_image = np.zeros((512, 512, 3), dtype=np.uint8)
    for i in range(0, 512, 64):
        for j in range(0, 512, 64):
            if (i // 64 + j // 64) % 2 == 0:
                contrast_image[i:i+64, j:j+64] = 255
            else:
                contrast_image[i:i+64, j:j+64] = 20
    cv2.imwrite(str(demo_dir / "high_contrast.jpg"), contrast_image)
    
    print(f"Demo images created in {demo_dir}")
    return demo_dir


def demo_single_image_processing(framework, image_path):
    """Demo single image processing with analysis."""
    print(f"\n{'='*60}")
    print(f"Processing: {image_path.name}")
    print(f"{'='*60}")
    
    # Process with auto mode and get analysis
    processed, analysis = framework.process_single_image(
        image_path, 
        mode=ProcessingMode.AUTO,
        return_analysis=True
    )
    
    # Print analysis results
    metrics = analysis["illumination_metrics"]
    print(f"Illumination Analysis:")
    print(f"  Mean brightness: {metrics.mean_brightness:.3f}")
    print(f"  Illumination uniformity: {metrics.illumination_uniformity:.3f}")
    print(f"  Low light ratio: {metrics.low_light_ratio:.3f}")
    print(f"  High contrast areas: {metrics.high_contrast_areas:.3f}")
    print(f"  Dominant brightness zone: {metrics.dominant_brightness_zone}")
    print(f"  Recommended mode: {metrics.recommended_mode}")
    print(f"  Processing time: {analysis['processing_time']:.3f} seconds")
    
    # Save processed result
    output_path = image_path.parent / "processed" / f"processed_{image_path.name}"
    output_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(output_path), processed)
    print(f"Processed image saved to: {output_path}")
    
    return processed, analysis


def demo_mode_comparison(framework, image_path):
    """Demo comparison of different processing modes."""
    print(f"\nCreating mode comparison for: {image_path.name}")
    
    # Create comparison visualization
    comparison = framework.compare_processing_modes(image_path)
    
    # Save comparison
    comparison_path = image_path.parent / "comparisons" / f"comparison_{image_path.name}"
    comparison_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(comparison_path), comparison)
    print(f"Mode comparison saved to: {comparison_path}")
    
    return comparison


def demo_illumination_visualization(framework, image_path):
    """Demo illumination analysis visualization."""
    print(f"Creating illumination visualization for: {image_path.name}")
    
    # Create illumination visualization
    viz = framework.create_illumination_visualization(image_path)
    
    # Save visualization
    viz_path = image_path.parent / "visualizations" / f"illumination_{image_path.name}"
    viz_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(viz_path), viz)
    print(f"Illumination visualization saved to: {viz_path}")
    
    return viz


def demo_batch_processing(framework, demo_dir):
    """Demo batch processing capabilities."""
    print(f"\n{'='*60}")
    print("BATCH PROCESSING DEMO")
    print(f"{'='*60}")
    
    # Get all demo images
    image_paths = list(demo_dir.glob("*.jpg"))
    
    # Process batch
    output_dir = demo_dir / "batch_processed"
    results = framework.process_batch(
        image_paths, 
        output_dir,
        mode=ProcessingMode.AUTO,
        show_progress=True
    )
    
    print(f"\nBatch Processing Results:")
    print(f"  Total images: {results['total_images']}")
    print(f"  Successful: {results['successful']}")
    print(f"  Failed: {results['failed']}")
    if results.get('average_time'):
        print(f"  Average processing time: {results['average_time']:.3f} seconds")
        print(f"  Total processing time: {results['total_time']:.3f} seconds")
    
    return results


def demo_custom_configuration():
    """Demo custom configuration options."""
    print(f"\n{'='*60}")
    print("CUSTOM CONFIGURATION DEMO")
    print(f"{'='*60}")
    
    # Create custom configuration for aggressive low-light processing
    aggressive_config = ProcessingConfig(
        low_light_gamma=0.4,  # More aggressive brightening
        low_light_contrast=1.4,  # Higher contrast
        enable_retinex=True,  # Enable Retinex enhancement
        enable_clahe=True,
        enable_denoising=True,
        enable_sharpening=True
    )
    
    # Create framework with custom config
    custom_framework = ACHIEVEFramework(aggressive_config)
    
    print("Created custom framework with aggressive low-light processing:")
    print(f"  Low-light gamma: {aggressive_config.low_light_gamma}")
    print(f"  Low-light contrast: {aggressive_config.low_light_contrast}")
    print(f"  Retinex enabled: {aggressive_config.enable_retinex}")
    
    return custom_framework


def demo_adaptive_processing(framework, demo_dir):
    """Demo adaptive processing capabilities."""
    print(f"\n{'='*60}")
    print("ADAPTIVE PROCESSING DEMO")
    print(f"{'='*60}")
    
    # Process several images to build mode history
    image_paths = list(demo_dir.glob("*.jpg"))[:3]
    
    for image_path in image_paths:
        framework.process_single_image(image_path, ProcessingMode.AUTO)
    
    # Get processing info
    info = framework.get_framework_stats()
    
    print("Framework Statistics:")
    print(f"  Total images processed: {info['total_images']}")
    print(f"  Low-light mode count: {info['low_light_count']}")
    print(f"  Standard mode count: {info['standard_count']}")
    print(f"  Average processing time: {info['average_processing_time']:.3f} seconds")
    
    if 'dominant_mode' in info:
        print(f"  Dominant processing mode: {info['dominant_mode']}")
        print(f"  Processing stability: {info['processing_stability']:.3f}")
    
    # Demo adaptive feedback
    print("\nDemonstrating adaptive feedback...")
    framework.processor.adaptive_threshold_update(0.3)  # Negative feedback
    print("Applied negative feedback - parameters adjusted")
    
    return info


def main():
    """Main demo function."""
    print("ACHIEVE Framework Demonstration")
    print("===============================")
    
    try:
        # Create demo images
        demo_dir = create_demo_images()
        
        # Initialize framework
        print("\nInitializing ACHIEVE framework...")
        framework = ACHIEVEFramework()
        
        # Demo single image processing for each demo image
        image_paths = list(demo_dir.glob("*.jpg"))
        
        for image_path in image_paths:
            # Process image with analysis
            processed, analysis = demo_single_image_processing(framework, image_path)
            
            # Create mode comparison
            comparison = demo_mode_comparison(framework, image_path)
            
            # Create illumination visualization
            viz = demo_illumination_visualization(framework, image_path)
        
        # Demo batch processing
        batch_results = demo_batch_processing(framework, demo_dir)
        
        # Demo custom configuration
        custom_framework = demo_custom_configuration()
        
        # Demo adaptive processing
        adaptive_info = demo_adaptive_processing(framework, demo_dir)
        
        print(f"\n{'='*60}")
        print("DEMO COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"Results saved in: {demo_dir}")
        print("Check the following directories:")
        print("  - processed/: Individual processed images")
        print("  - comparisons/: Mode comparison visualizations")  
        print("  - visualizations/: Illumination analysis visualizations")
        print("  - batch_processed/: Batch processing results")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())