"""
Image Enhancement Module
========================

This module provides various image enhancement techniques optimized
for different illumination conditions and processing modes.
"""

import numpy as np
import cv2
from typing import Optional, Tuple, Union
from scipy import ndimage
from skimage import exposure, filters


class ImageEnhancer:
    """
    Provides image enhancement techniques for different illumination conditions.
    
    This class implements various enhancement algorithms optimized for
    low-light and standard lighting conditions.
    """
    
    def __init__(self):
        """Initialize the ImageEnhancer."""
        pass
    
    def enhance_low_light(self, image: np.ndarray, 
                         gamma: float = 0.5,
                         alpha: float = 1.2) -> np.ndarray:
        """
        Enhance images in low-light conditions.
        
        Args:
            image: Input image
            gamma: Gamma correction parameter (< 1 brightens)
            alpha: Contrast adjustment parameter
            
        Returns:
            Enhanced image
        """
        enhanced = image.copy()
        
        # Apply gamma correction for brightness enhancement
        enhanced = self.gamma_correction(enhanced, gamma)
        
        # Apply contrast enhancement
        enhanced = self.adjust_contrast(enhanced, alpha)
        
        # Apply adaptive histogram equalization
        enhanced = self.apply_clahe(enhanced)
        
        # Reduce noise that may be amplified
        enhanced = self.denoise(enhanced)
        
        return enhanced
    
    def enhance_standard(self, image: np.ndarray,
                        enhance_contrast: bool = True,
                        sharpen: bool = True) -> np.ndarray:
        """
        Enhance images in standard lighting conditions.
        
        Args:
            image: Input image
            enhance_contrast: Whether to apply contrast enhancement
            sharpen: Whether to apply sharpening
            
        Returns:
            Enhanced image
        """
        enhanced = image.copy()
        
        if enhance_contrast:
            # Mild contrast enhancement
            enhanced = self.adjust_contrast(enhanced, 1.1)
        
        if sharpen:
            # Apply mild sharpening
            enhanced = self.sharpen_image(enhanced)
        
        return enhanced
    
    def gamma_correction(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """
        Apply gamma correction to adjust image brightness.
        
        Args:
            image: Input image
            gamma: Gamma parameter (< 1 brightens, > 1 darkens)
            
        Returns:
            Gamma-corrected image
        """
        # Build lookup table for gamma correction
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype(np.uint8)
        
        # Apply gamma correction using lookup table
        return cv2.LUT(image, table)
    
    def adjust_contrast(self, image: np.ndarray, alpha: float) -> np.ndarray:
        """
        Adjust image contrast.
        
        Args:
            image: Input image
            alpha: Contrast multiplier (> 1 increases contrast)
            
        Returns:
            Contrast-adjusted image
        """
        # Convert to float for processing
        float_img = image.astype(np.float32)
        
        # Apply contrast adjustment
        adjusted = alpha * float_img
        
        # Clip values and convert back to uint8
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        
        return adjusted
    
    def apply_clahe(self, image: np.ndarray, 
                   clip_limit: float = 2.0,
                   tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
        
        Args:
            image: Input image
            clip_limit: Clipping limit for contrast limiting
            tile_grid_size: Size of grid for histogram equalization
            
        Returns:
            CLAHE-enhanced image
        """
        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=clip_limit, 
                               tileGridSize=tile_grid_size)
        
        if len(image.shape) == 3:
            # Apply CLAHE to each channel in LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            # Apply CLAHE directly for grayscale
            enhanced = clahe.apply(image)
        
        return enhanced
    
    def sharpen_image(self, image: np.ndarray, 
                     strength: float = 0.5) -> np.ndarray:
        """
        Apply image sharpening using unsharp masking.
        
        Args:
            image: Input image
            strength: Sharpening strength (0-1)
            
        Returns:
            Sharpened image
        """
        if len(image.shape) == 3:
            # Apply sharpening to each channel
            enhanced = np.zeros_like(image)
            for i in range(3):
                enhanced[:, :, i] = self._apply_unsharp_mask(
                    image[:, :, i], strength)
        else:
            enhanced = self._apply_unsharp_mask(image, strength)
        
        return enhanced
    
    def _apply_unsharp_mask(self, channel: np.ndarray, 
                           strength: float) -> np.ndarray:
        """Apply unsharp masking to a single channel."""
        # Create Gaussian blur
        blurred = cv2.GaussianBlur(channel, (0, 0), 1.5)
        
        # Create unsharp mask
        mask = channel.astype(np.float32) - blurred.astype(np.float32)
        
        # Apply mask
        sharpened = channel.astype(np.float32) + strength * mask
        
        # Clip and convert back
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        return sharpened
    
    def denoise(self, image: np.ndarray, method: str = "bilateral") -> np.ndarray:
        """
        Apply noise reduction to the image.
        
        Args:
            image: Input image
            method: Denoising method ("bilateral", "gaussian", "median")
            
        Returns:
            Denoised image
        """
        if method == "bilateral":
            if len(image.shape) == 3:
                denoised = cv2.bilateralFilter(image, 9, 75, 75)
            else:
                denoised = cv2.bilateralFilter(image, 9, 75, 75)
        elif method == "gaussian":
            denoised = cv2.GaussianBlur(image, (5, 5), 0)
        elif method == "median":
            denoised = cv2.medianBlur(image, 5)
        else:
            denoised = image.copy()
        
        return denoised
    
    def retinex_enhancement(self, image: np.ndarray, 
                           sigma_list: list = [15, 80, 250]) -> np.ndarray:
        """
        Apply Multi-Scale Retinex (MSR) enhancement for illumination invariance.
        
        Args:
            image: Input image
            sigma_list: List of sigma values for different scales
            
        Returns:
            Retinex-enhanced image
        """
        # Convert to float and add small epsilon to avoid log(0)
        float_img = image.astype(np.float32) + 1.0
        
        if len(image.shape) == 3:
            # Apply to each channel
            enhanced = np.zeros_like(float_img)
            for i in range(3):
                enhanced[:, :, i] = self._apply_msr_channel(
                    float_img[:, :, i], sigma_list)
        else:
            enhanced = self._apply_msr_channel(float_img, sigma_list)
        
        # Normalize and convert back to uint8
        enhanced = self._normalize_retinex_output(enhanced)
        
        return enhanced
    
    def _apply_msr_channel(self, channel: np.ndarray, 
                          sigma_list: list) -> np.ndarray:
        """Apply Multi-Scale Retinex to a single channel."""
        retinex_sum = np.zeros_like(channel)
        
        for sigma in sigma_list:
            # Create Gaussian kernel
            kernel_size = int(6 * sigma + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(channel, (kernel_size, kernel_size), sigma)
            
            # Calculate retinex for this scale
            retinex = np.log(channel) - np.log(blurred + 1e-6)
            retinex_sum += retinex
        
        # Average across scales
        msr = retinex_sum / len(sigma_list)
        
        return msr
    
    def _normalize_retinex_output(self, retinex_img: np.ndarray) -> np.ndarray:
        """Normalize Retinex output to 0-255 range."""
        # Simple normalization
        min_val = np.min(retinex_img)
        max_val = np.max(retinex_img)
        
        if max_val > min_val:
            normalized = 255 * (retinex_img - min_val) / (max_val - min_val)
        else:
            normalized = retinex_img
        
        return np.clip(normalized, 0, 255).astype(np.uint8)
    
    def correct_uneven_illumination(self, image: np.ndarray,
                                   illumination_map: np.ndarray) -> np.ndarray:
        """
        Correct uneven illumination using an illumination map.
        
        Args:
            image: Input image
            illumination_map: Estimated illumination map
            
        Returns:
            Illumination-corrected image
        """
        # Convert to float for processing
        float_img = image.astype(np.float32)
        float_illum = illumination_map.astype(np.float32) + 1e-6
        
        if len(image.shape) == 3:
            # Apply correction to each channel
            corrected = np.zeros_like(float_img)
            for i in range(3):
                corrected[:, :, i] = (float_img[:, :, i] / float_illum) * 128
        else:
            corrected = (float_img / float_illum) * 128
        
        # Normalize and convert back
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        
        return corrected