"""
CT calibration, trauma Hounsfield windowing, and 6-channel anatomical spatial feature extraction.
"""

from typing import Tuple
import numpy as np


def calibrate_dicom_to_hu(
    raw_pixel_array: np.ndarray,
    rescale_slope: float = 1.0,
    rescale_intercept: float = -1024.0
) -> np.ndarray:
    """Converts raw 16-bit DICOM pixel array values into physical Hounsfield Units (HU)."""
    return raw_pixel_array.astype(np.float32) * float(rescale_slope) + float(rescale_intercept)


def apply_trauma_window(
    ct_hu: np.ndarray,
    window_center: float = 50.0,
    window_width: float = 400.0
) -> np.ndarray:
    """
    Clips physical HU values to [-150, +250] HU and normalizes into [0.0, 1.0].
    
    This window suppresses bone (> +250 HU) and air (< -150 HU) while maximizing
    contrast resolution across abdominal organs (+40 to +70 HU) and active contrast blush (+225 HU).
    """
    min_hu = window_center - (window_width / 2.0)  # -150.0 HU
    max_hu = window_center + (window_width / 2.0)  # +250.0 HU
    clipped = np.clip(ct_hu, min_hu, max_hu)
    return ((clipped - min_hu) / (max_hu - min_hu)).astype(np.float32)


def extract_6_anatomical_features(norm_slice: np.ndarray) -> np.ndarray:
    """
    Extracts 6 spatial-anatomical feature channels:
      Channel 0: Normalized Windowed HU X(v) in [0, 1]
      Channel 1: Quadratic Contrast Energy X(v)^2 (amplifies active arterial contrast blush)
      Channel 2: Gradient Edge Magnitude |grad X(v)| (sharpens organ capsule boundaries)
      Channel 3: Anteroposterior Coordinate Map (separates anterior bowel from posterior kidneys)
      Channel 4: Radial Distance Map (focuses on central peritoneal space)
      Channel 5: Quadrant Multiplicative Map (explicit 4-quadrant localization)
    """
    H, W = norm_slice.shape
    yy, xx = np.meshgrid(np.linspace(-1.0, 1.0, H), np.linspace(-1.0, 1.0, W), indexing="ij")
    gy, gx = np.gradient(norm_slice)

    f0_x = norm_slice
    f1_x2 = norm_slice ** 2
    f2_edge = np.clip(np.sqrt(gx**2 + gy**2) * 3.2 + 0.25 * norm_slice, 0.0, 1.0).astype(np.float32)
    f3_ap = np.clip(0.65 * norm_slice + 0.35 * ((yy + 1.0) / 2.0), 0.0, 1.0).astype(np.float32)
    r2_norm = 1.0 - np.clip((yy**2 + xx**2) / 1.4, 0.0, 1.0)
    f4_rad = np.clip(norm_slice * (0.35 + 0.65 * r2_norm), 0.0, 1.0).astype(np.float32)
    f5_quad = (yy * xx).astype(np.float32)

    return np.stack([f0_x, f1_x2, f2_edge, f3_ap, f4_rad, f5_quad], axis=0)
