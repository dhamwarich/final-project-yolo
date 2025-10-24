"""
Module 2: Geometry Estimator
Handles calibration and approximate distance/height estimation from 2D camera geometry.
"""


def calibrate_reference(
    ref_height_m: float = 1.70,
    ref_distance_m: float = 2.0,
    ref_height_px: float = 360
) -> float:
    """
    Compute focal length in pixels from a reference measurement.
    
    Args:
        ref_height_m: Known real-world height (meters)
        ref_distance_m: Known distance from camera (meters)
        ref_height_px: Measured pixel height at that distance
    
    Returns:
        f_px: Focal length in pixels
    
    Formula:
        f = (h_px * Z) / H_real
    """
    f_px = (ref_height_px * ref_distance_m) / ref_height_m
    return f_px


def estimate_distance_ratio(
    torso_leg_ratio: float,
    ref_ratio: float = 0.53,
    k1: float = 1.0
) -> float:
    """
    Return relative distance multiplier based on torso-leg ratio.
    
    The torso-leg ratio changes with perspective: closer people appear
    to have higher ratios due to foreshortening. We use the inverse
    relationship to estimate relative distance.
    
    Args:
        torso_leg_ratio: Measured torso/leg ratio from pose
        ref_ratio: Reference ratio for typical adult (~0.53)
        k1: Scale constant (default 1.0)
    
    Returns:
        Z_approx: Relative distance multiplier
    
    Formula:
        Z_approx = k1 * (r_ref / r_measured)
    
    Note:
        Smaller ratio (more stretched) → farther away → larger Z
        Larger ratio (more compressed) → closer → smaller Z
    """
    if torso_leg_ratio < 0.01:
        # Avoid division by zero
        return k1 * 10.0  # Assume very far
    
    Z = k1 * (ref_ratio / torso_leg_ratio)
    return Z


def estimate_height_m(
    height_px: float,
    torso_leg_ratio: float,
    f_px: float,
    ref_ratio: float = 0.53,
    k1: float = 1.0
) -> float:
    """
    Approximate real-world height in meters.
    
    Args:
        height_px: Pixel height from pose detection
        torso_leg_ratio: Measured torso/leg ratio
        f_px: Focal length in pixels (from calibration)
        ref_ratio: Reference torso/leg ratio
        k1: Scale constant
    
    Returns:
        H_real: Estimated real-world height in meters
    
    Formula:
        Z_approx = k1 * (r_ref / r_measured)
        H_real = (h_px * Z_approx) / f
    """
    Z_approx = estimate_distance_ratio(torso_leg_ratio, ref_ratio, k1)
    
    if f_px < 1.0:
        # Avoid division by zero
        return 0.0
    
    H_real = (height_px * Z_approx) / f_px
    return H_real


def estimate_distance_m(
    height_px: float,
    height_m_est: float,
    f_px: float
) -> float:
    """
    Estimate distance from camera in meters.
    
    Args:
        height_px: Pixel height from pose detection
        height_m_est: Estimated real-world height (meters)
        f_px: Focal length in pixels
    
    Returns:
        Distance in meters
    
    Formula:
        Z = (H_real * f) / h_px
    """
    if height_px < 1.0:
        return 0.0
    
    Z = (height_m_est * f_px) / height_px
    return Z


# Default calibration parameters
# These can be adjusted based on actual camera setup
DEFAULT_FOCAL_LENGTH_PX = 424.0  # Computed from 1.70m @ 2.0m = 360px
DEFAULT_REF_RATIO = 0.53  # Average adult torso/leg ratio
DEFAULT_K1 = 1.0  # Scale constant
