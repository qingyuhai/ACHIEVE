# ACHIEVE: Dual-Mode Illumination-Aware Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)

## Overview

ACHIEVE (Adaptive CHoice Illumination Enhancement VErsatile) is a sophisticated dual-mode illumination-aware framework designed to process images in unevenly illuminated environments. The framework automatically adapts its processing strategy based on illumination conditions, providing optimal enhancement for both low-light and standard lighting scenarios.

### Key Features

- **Dual-Mode Processing**: Automatic switching between low-light and standard processing modes
- **Illumination Analysis**: Comprehensive analysis of lighting conditions and uniformity
- **Adaptive Enhancement**: Dynamic parameter adjustment based on image characteristics
- **Multiple Enhancement Techniques**: Gamma correction, CLAHE, Retinex, denoising, and sharpening
- **Batch Processing**: Efficient processing of multiple images
- **Visualization Tools**: Illumination maps, shadow detection, and mode comparisons
- **Customizable Configuration**: Flexible parameters for different use cases

## Installation

### From Source

```bash
git clone https://github.com/qingyuhai/ACHIEVE-.git
cd ACHIEVE-
pip install -r requirements.txt
pip install -e .
```

### Dependencies

- OpenCV Python (>=4.5.0)
- NumPy (>=1.19.0)
- SciPy (>=1.6.0)
- scikit-image (>=0.18.0)
- matplotlib (>=3.3.0)

## Quick Start

```python
from achieve import ACHIEVEFramework, ProcessingMode
import cv2

# Initialize the framework
framework = ACHIEVEFramework()

# Load and process an image
image = cv2.imread('your_image.jpg')
processed = framework.process_single_image(image, mode=ProcessingMode.AUTO)

# Save the result
cv2.imwrite('processed_image.jpg', processed)
```

## Architecture

The ACHIEVE framework consists of four main components:

### 1. Illumination Analyzer (`illumination.py`)
- Analyzes lighting conditions and uniformity
- Calculates metrics like brightness, contrast, and low-light ratios
- Recommends optimal processing mode
- Generates illumination maps and detects shadow regions

### 2. Image Enhancer (`enhancement.py`)
- Implements various enhancement techniques:
  - Gamma correction for brightness adjustment
  - Contrast Limited Adaptive Histogram Equalization (CLAHE)
  - Multi-Scale Retinex for illumination invariance
  - Bilateral filtering for noise reduction
  - Unsharp masking for sharpening

### 3. Dual-Mode Processor (`processing.py`)
- Coordinates the processing pipeline
- Switches between low-light and standard modes
- Tracks processing history for adaptive behavior
- Supports batch processing and parameter adaptation

### 4. Main Framework (`achieve_framework.py`)
- High-level interface for all functionality
- Provides visualization and analysis tools
- Handles file I/O and batch operations
- Maintains processing statistics

## Processing Modes

### Auto Mode (Recommended)
Automatically selects the optimal processing mode based on illumination analysis:
- **Low-light mode**: For dark or unevenly illuminated images
- **Standard mode**: For well-lit images with uniform illumination

### Low-Light Mode
Optimized for challenging lighting conditions:
1. Illumination correction using estimated illumination maps
2. Optional Retinex enhancement for illumination invariance
3. Gamma correction for brightness enhancement
4. Contrast enhancement
5. Adaptive histogram equalization (CLAHE)
6. Bilateral filtering for noise reduction
7. Mild sharpening for detail recovery

### Standard Mode
Optimized for normal lighting conditions:
1. Minor illumination correction if needed
2. Mild contrast enhancement
3. Selective CLAHE for high contrast areas
4. Detail-preserving sharpening
5. Light denoising if required

## Usage Examples

### Basic Usage

```python
from achieve import ACHIEVEFramework, ProcessingMode

# Initialize framework
framework = ACHIEVEFramework()

# Process single image with analysis
processed, analysis = framework.process_single_image(
    'input.jpg',
    mode=ProcessingMode.AUTO,
    return_analysis=True
)

print(f"Recommended mode: {analysis['illumination_metrics'].recommended_mode}")
print(f"Mean brightness: {analysis['illumination_metrics'].mean_brightness:.3f}")
```

### Batch Processing

```python
from pathlib import Path

# Process all images in a directory
image_paths = list(Path("input_folder").glob("*.jpg"))
results = framework.process_batch(
    image_paths,
    "output_folder",
    mode=ProcessingMode.AUTO,
    show_progress=True
)

print(f"Processed {results['successful']} images successfully")
```

### Custom Configuration

```python
from achieve import ProcessingConfig

# Create custom configuration
config = ProcessingConfig(
    low_light_gamma=0.4,        # More aggressive brightening
    low_light_contrast=1.4,     # Higher contrast
    enable_retinex=True,        # Enable Retinex enhancement
    enable_clahe=True
)

# Initialize with custom config
framework = ACHIEVEFramework(config)
```

### Illumination Analysis

```python
# Analyze illumination conditions
metrics = framework.analyze_illumination_conditions('image.jpg')

print(f"Illumination uniformity: {metrics.illumination_uniformity:.3f}")
print(f"Low light ratio: {metrics.low_light_ratio:.3f}")
print(f"Dominant brightness zone: {metrics.dominant_brightness_zone}")

# Create illumination visualization
viz = framework.create_illumination_visualization('image.jpg', 'analysis.jpg')
```

### Mode Comparison

```python
# Compare different processing modes
comparison = framework.compare_processing_modes('image.jpg', 'comparison.jpg')
```

## Configuration Options

The `ProcessingConfig` class allows customization of processing parameters:

```python
config = ProcessingConfig(
    low_light_gamma=0.5,        # Gamma correction for low-light (0.3-0.7)
    low_light_contrast=1.2,     # Contrast enhancement for low-light
    standard_contrast=1.1,      # Contrast enhancement for standard
    enable_clahe=True,          # Enable CLAHE enhancement
    enable_retinex=False,       # Enable Retinex enhancement
    enable_denoising=True,      # Enable noise reduction
    enable_sharpening=True      # Enable sharpening
)
```

## Running Examples

The framework includes comprehensive examples:

```bash
# Run the comprehensive demo
cd examples
python demo.py

# Run the simple example
python simple_example.py
```

## Testing

Run the test suite to verify installation:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=achieve --cov-report=html
```

## Performance Characteristics

- **Processing Speed**: ~0.1-0.5 seconds per image (512x512) on modern hardware
- **Memory Usage**: Scales linearly with image size
- **Scalability**: Supports batch processing with progress tracking
- **Adaptive Behavior**: Learns from processing history for stability

## Use Cases

- **Security Surveillance**: Enhance low-light surveillance footage
- **Medical Imaging**: Improve visibility of medical images with uneven illumination
- **Autonomous Vehicles**: Process images from cameras in varying lighting conditions
- **Photography**: Automatic enhancement of photos taken in challenging lighting
- **Remote Sensing**: Process satellite/aerial imagery with varying illumination
- **Industrial Inspection**: Enhance images for quality control under different lighting

## Technical Details

### Illumination Metrics

The framework calculates several key metrics:
- **Mean Brightness**: Overall brightness level (0-1)
- **Illumination Uniformity**: Measure of lighting consistency (0-1)
- **Low-Light Ratio**: Proportion of pixels in low-light conditions
- **High Contrast Areas**: Areas with significant local contrast
- **Brightness Variance**: Statistical measure of brightness distribution

### Enhancement Algorithms

1. **Gamma Correction**: Power-law transformation for brightness adjustment
2. **CLAHE**: Contrast-limited adaptive histogram equalization
3. **Retinex**: Multi-scale retinex for illumination-invariant processing
4. **Bilateral Filtering**: Edge-preserving noise reduction
5. **Unsharp Masking**: Detail enhancement through high-pass filtering

## Contributing

We welcome contributions to the ACHIEVE framework! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request with a clear description

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use ACHIEVE in your research, please cite:

```bibtex
@software{achieve_framework,
  title={ACHIEVE: Dual-Mode Illumination-Aware Framework for Unevenly Illuminated Environments},
  author={ACHIEVE Team},
  year={2024},
  url={https://github.com/qingyuhai/ACHIEVE-}
}
```

## Support

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Check the examples directory for usage patterns
- Review the test suite for implementation details

## Roadmap

Future enhancements planned:
- [ ] GPU acceleration support
- [ ] Real-time video processing
- [ ] Additional enhancement algorithms
- [ ] Web-based interface
- [ ] Integration with popular ML frameworks
- [ ] Advanced adaptive learning mechanisms