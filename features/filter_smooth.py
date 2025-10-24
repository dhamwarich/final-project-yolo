"""
Module: Filter & Smooth
Outlier filtering and temporal smoothing for pose features.
"""


def is_valid_frame(features: dict) -> bool:
    """
    Rejects frames with impossible or implausible values.
    
    Args:
        features: Dictionary of computed features
    
    Returns:
        True if frame is valid, False otherwise
    
    Thresholds:
        - height_px > 100 (minimum realistic height)
        - shoulder_hip < 3 (max reasonable shoulder/hip ratio)
        - torso_leg < 3 (max reasonable torso/leg ratio)
        - All values must be positive and finite
    """
    height_px = features.get('height_px', 0)
    shoulder_hip = features.get('shoulder_hip', 0)
    torso_leg = features.get('torso_leg', 0)
    
    # Check for invalid numeric values
    if not all([
        isinstance(height_px, (int, float)),
        isinstance(shoulder_hip, (int, float)),
        isinstance(torso_leg, (int, float))
    ]):
        return False
    
    # Check for NaN or infinity
    import math
    if any([
        math.isnan(height_px) or math.isinf(height_px),
        math.isnan(shoulder_hip) or math.isinf(shoulder_hip),
        math.isnan(torso_leg) or math.isinf(torso_leg)
    ]):
        return False
    
    # Apply thresholds
    if height_px < 100:  # Too short
        return False
    
    if shoulder_hip < 0.5 or shoulder_hip > 3.0:  # Unrealistic ratio
        return False
    
    if torso_leg < 0.2 or torso_leg > 3.0:  # Unrealistic ratio
        return False
    
    return True


def smooth_features(curr: dict, prev: dict, alpha: float = 0.2) -> dict:
    """
    Apply exponential moving average (EMA) to numeric fields.
    
    Formula: smoothed = alpha * current + (1 - alpha) * previous
    
    Args:
        curr: Current frame features
        prev: Previous frame features (or empty dict on first frame)
        alpha: Smoothing factor (0-1). Lower = more smoothing
               0.2 means 20% current, 80% previous
    
    Returns:
        Smoothed features dictionary
    """
    # If no previous features, return current
    if not prev:
        return curr.copy()
    
    smoothed = {}
    
    # Smooth numeric fields
    numeric_fields = ['height_px', 'shoulder_hip', 'torso_leg']
    
    for field in numeric_fields:
        if field in curr and field in prev:
            # Apply EMA
            smoothed[field] = alpha * curr[field] + (1 - alpha) * prev[field]
        else:
            # Use current if no previous
            smoothed[field] = curr[field]
    
    # For bbox_center, smooth the coordinates
    if 'bbox_center' in curr and 'bbox_center' in prev:
        curr_x, curr_y = curr['bbox_center']
        prev_x, prev_y = prev['bbox_center']
        smoothed['bbox_center'] = (
            alpha * curr_x + (1 - alpha) * prev_x,
            alpha * curr_y + (1 - alpha) * prev_y
        )
    else:
        smoothed['bbox_center'] = curr['bbox_center']
    
    return smoothed


def smooth_appearance_features(curr: dict, prev: dict, alpha: float = 0.3) -> dict:
    """
    Apply EMA smoothing to appearance features (color, texture).
    
    Args:
        curr: Current appearance features
        prev: Previous appearance features (or empty dict on first)
        alpha: Smoothing factor (0-1)
    
    Returns:
        Smoothed appearance features
    """
    # If no previous features, return current
    if not prev:
        return curr.copy()
    
    smoothed = {}
    
    # Smooth appearance fields
    appearance_fields = ['top_h', 'bot_h', 'texture']
    
    for field in appearance_fields:
        if field in curr and field in prev:
            # Apply EMA
            smoothed[field] = alpha * curr[field] + (1 - alpha) * prev[field]
        elif field in curr:
            # Use current if no previous
            smoothed[field] = curr[field]
        else:
            smoothed[field] = 0.0
    
    return smoothed
