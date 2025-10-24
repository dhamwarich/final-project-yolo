"""
Module 2: Ratio Calculator
Compute geometric descriptors from pose keypoints.
"""

import math


def estimate_height_px(kps: dict) -> float:
    """
    Return pixel height = y(ankle_avg) - y(nose).
    
    Args:
        kps: Dictionary of keypoints
    
    Returns:
        Estimated height in pixels
    """
    nose_y = kps['nose'][1]
    ankle_l_y = kps['ankle_l'][1]
    ankle_r_y = kps['ankle_r'][1]
    
    # Average ankle y-coordinate
    ankle_avg_y = (ankle_l_y + ankle_r_y) / 2
    
    # Height is the vertical distance
    height = abs(ankle_avg_y - nose_y)
    
    return float(height)


def calc_shoulder_hip_ratio(kps: dict) -> float:
    """
    Return width(shoulder) / width(hip).
    
    Args:
        kps: Dictionary of keypoints
    
    Returns:
        Shoulder to hip width ratio
    """
    # Calculate shoulder width
    shoulder_l = kps['shoulder_l']
    shoulder_r = kps['shoulder_r']
    shoulder_width = math.sqrt(
        (shoulder_r[0] - shoulder_l[0])**2 + 
        (shoulder_r[1] - shoulder_l[1])**2
    )
    
    # Calculate hip width
    hip_l = kps['hip_l']
    hip_r = kps['hip_r']
    hip_width = math.sqrt(
        (hip_r[0] - hip_l[0])**2 + 
        (hip_r[1] - hip_l[1])**2
    )
    
    # Avoid division by zero
    if hip_width < 1e-6:
        return 0.0
    
    ratio = shoulder_width / hip_width
    return float(ratio)


def calc_torso_leg_ratio(kps: dict) -> float:
    """
    Return vertical distance(shoulder→hip) / (hip→ankle).
    
    Args:
        kps: Dictionary of keypoints
    
    Returns:
        Torso to leg length ratio
    """
    # Average shoulder y-coordinate
    shoulder_avg_y = (kps['shoulder_l'][1] + kps['shoulder_r'][1]) / 2
    
    # Average hip y-coordinate
    hip_avg_y = (kps['hip_l'][1] + kps['hip_r'][1]) / 2
    
    # Average ankle y-coordinate
    ankle_avg_y = (kps['ankle_l'][1] + kps['ankle_r'][1]) / 2
    
    # Torso length (shoulder to hip)
    torso_length = abs(hip_avg_y - shoulder_avg_y)
    
    # Leg length (hip to ankle)
    leg_length = abs(ankle_avg_y - hip_avg_y)
    
    # Avoid division by zero
    if leg_length < 1e-6:
        return 0.0
    
    ratio = torso_length / leg_length
    return float(ratio)


def compile_features(kps: dict, bbox: tuple) -> dict:
    """
    Combine all metrics into one feature vector.
    
    Args:
        kps: Dictionary of keypoints
        bbox: Bounding box (x1, y1, x2, y2)
    
    Returns:
        Dictionary containing all computed features:
        {
            'height_px': float,
            'shoulder_hip': float,
            'torso_leg': float,
            'bbox_center': (x, y)
        }
    """
    height_px = estimate_height_px(kps)
    shoulder_hip = calc_shoulder_hip_ratio(kps)
    torso_leg = calc_torso_leg_ratio(kps)
    
    # Calculate bbox center
    x1, y1, x2, y2 = bbox
    bbox_center = ((x1 + x2) / 2, (y1 + y2) / 2)
    
    features = {
        'height_px': height_px,
        'shoulder_hip': shoulder_hip,
        'torso_leg': torso_leg,
        'bbox_center': bbox_center
    }
    
    return features
