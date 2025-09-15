# ACHIEVE Framework - Quick Reference Guide

## Installation
```bash
pip install -r requirements.txt
pip install -e .
```

## Basic Usage

### Single Image Processing
```python
from achieve import ACHIEVEFramework, ProcessingMode

# Initialize framework
framework = ACHIEVEFramework()

# Process image (automatic mode selection)
processed = framework.process_single_image('input.jpg')

# Process with specific mode
processed = framework.process_single_image('input.jpg', ProcessingMode.LOW_LIGHT)

# Get detailed analysis
processed, analysis = framework.process_single_image(
    'input.jpg', 
    return_analysis=True
)
```

### Illumination Analysis
```python
# Analyze lighting conditions
metrics = framework.analyze_illumination_conditions('image.jpg')
print(f"Brightness: {metrics.mean_brightness:.3f}")
print(f"Uniformity: {metrics.illumination_uniformity:.3f}")  
print(f"Recommended mode: {metrics.recommended_mode}")
```

### Batch Processing
```python
from pathlib import Path

# Process all images in directory
image_paths = list(Path("input").glob("*.jpg"))
results = framework.process_batch(
    image_paths,
    "output_directory", 
    mode=ProcessingMode.AUTO
)
```

### Custom Configuration
```python
from achieve import ProcessingConfig

# Create custom settings
config = ProcessingConfig(
    low_light_gamma=0.4,      # Brightness adjustment
    low_light_contrast=1.3,   # Contrast enhancement
    enable_retinex=True,      # Illumination invariant processing
    enable_clahe=True,        # Adaptive histogram equalization
    enable_denoising=True,    # Noise reduction
    enable_sharpening=True    # Detail enhancement
)

framework = ACHIEVEFramework(config)
```

### Visualizations
```python
# Create illumination analysis visualization
viz = framework.create_illumination_visualization('image.jpg')

# Compare processing modes
comparison = framework.compare_processing_modes('image.jpg')
```

## Processing Modes

- **`ProcessingMode.AUTO`**: Automatic mode selection (recommended)
- **`ProcessingMode.LOW_LIGHT`**: Optimized for dark/uneven illumination  
- **`ProcessingMode.STANDARD`**: Optimized for normal lighting conditions

## Key Classes

### `ACHIEVEFramework`
Main interface for the framework
- `process_single_image()`: Process individual images
- `process_batch()`: Process multiple images  
- `analyze_illumination_conditions()`: Analyze lighting
- `create_illumination_visualization()`: Generate analysis plots
- `compare_processing_modes()`: Compare different modes

### `IlluminationAnalyzer` 
Analyze lighting conditions
- `analyze_illumination()`: Comprehensive analysis
- `get_illumination_map()`: Generate illumination map
- `detect_shadow_regions()`: Find shadow areas

### `ImageEnhancer`
Image enhancement techniques
- `enhance_low_light()`: Low-light optimization
- `enhance_standard()`: Standard enhancement
- `gamma_correction()`: Brightness adjustment
- `apply_clahe()`: Adaptive histogram equalization
- `sharpen_image()`: Detail enhancement
- `denoise()`: Noise reduction

### `DualModeProcessor`
Processing pipeline coordinator
- `process_image()`: Main processing function
- `batch_process()`: Process multiple images
- `get_processing_info()`: Get statistics

## Common Parameters

### Enhancement Settings
- `gamma`: 0.3-0.7 for low-light, 1.0 for standard (lower = brighter)
- `alpha`: 1.0-1.5 for contrast (higher = more contrast)
- `clip_limit`: 1.0-3.0 for CLAHE (higher = more aggressive)

### Thresholds
- `low_light_threshold`: 0.3 (brightness threshold for low-light detection)
- `uniformity_threshold`: 0.7 (uniformity threshold for mode selection)
- `high_contrast_threshold`: 50 (gradient threshold for contrast detection)

## Examples

See the `examples/` directory:
- `simple_example.py`: Basic usage
- `demo.py`: Comprehensive demonstration

## Testing

```bash
# Basic functionality test
python -c "from achieve import *; print('Import successful!')"

# Run examples
cd examples
python simple_example.py
```

## Tips

1. **Use AUTO mode** for best results - it automatically selects optimal processing
2. **Dark images** benefit from low-light mode with gamma correction
3. **Uneven lighting** is handled automatically through illumination correction
4. **Batch processing** is more efficient for multiple images
5. **Custom configs** allow fine-tuning for specific use cases
6. **Visualizations** help understand framework decisions

## Performance

- **Speed**: ~0.1-0.5s per 512x512 image
- **Memory**: Scales linearly with image size  
- **Batch**: Processes multiple images efficiently
- **Adaptive**: Learns from processing history

For more details, see the main README.md file.