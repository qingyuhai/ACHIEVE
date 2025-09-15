"""
ACHIEVE: Dual-Mode Illumination-Aware Framework
===============================================

A comprehensive framework for handling unevenly illuminated environments
with adaptive dual-mode processing capabilities.

Modules:
    - illumination: Illumination analysis and detection
    - enhancement: Image enhancement techniques
    - processing: Dual-mode processing pipeline
    - utils: Utility functions and helpers
"""

__version__ = "1.0.0"
__author__ = "ACHIEVE Team"

from .illumination import IlluminationAnalyzer
from .enhancement import ImageEnhancer
from .processing import DualModeProcessor, ProcessingMode, ProcessingConfig
from .achieve_framework import ACHIEVEFramework

__all__ = [
    "IlluminationAnalyzer",
    "ImageEnhancer", 
    "DualModeProcessor",
    "ProcessingMode",
    "ProcessingConfig",
    "ACHIEVEFramework"
]