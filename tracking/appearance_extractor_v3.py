"""
Module: Appearance Extractor V3 (LAB Color-Region)
Extract separate shirt and pants histograms in LAB color space.
"""

import cv2
import numpy as np
from typing import Dict, Tuple


def extract_lab_histograms(
    frame: np.ndarray,
    bbox: Tuple[float, float, float, float],
    bins: int = 16
) -> Dict:
    """
    Extract LAB-based color histograms for shirt (upper) and pants (lower).
    
    Args:
        frame: Full video frame (BGR)
        bbox: Bounding box (x1, y1, x2, y2)
        bins: Number of histogram bins (default 16)
    
    Returns:
        Dictionary with:
        - shirt_hist: Upper region LAB histogram (16 bins on A channel)
        - pants_hist: Lower region LAB histogram (16 bins on A channel)
        - aspect: Bbox aspect ratio
        - sat_mean: Mean saturation (for validation)
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Ensure valid bbox
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    
    if x2 <= x1 or y2 <= y1:
        # Invalid bbox - return empty histograms
        return {
            "shirt_hist": np.zeros(bins),
            "pants_hist": np.zeros(bins),
            "aspect": 1.0,
            "sat_mean": 0.0
        }
    
    # Extract person region
    person = frame[y1:y2, x1:x2]
    
    if person.size == 0:
        return {
            "shirt_hist": np.zeros(bins),
            "pants_hist": np.zeros(bins),
            "aspect": 1.0,
            "sat_mean": 0.0
        }
    
    h, w = person.shape[:2]
    
    # Compute bbox aspect ratio (height/width)
    aspect = h / max(w, 1.0)
    
    # Split into shirt (60% - upper body) and pants (40% - lower body)
    split_point = int(0.6 * h)
    shirt_region = person[:split_point, :]
    pants_region = person[split_point:, :]
    
    # Convert to LAB color space
    # LAB is more perceptually uniform than HSV
    # L = lightness, A = green-red, B = blue-yellow
    lab = cv2.cvtColor(person, cv2.COLOR_BGR2LAB)
    
    # Use A channel (green-red) for color discrimination
    # A channel ranges from 0-255, centered at 128
    a_channel = lab[:, :, 1]
    
    # Extract A channel for shirt and pants regions
    shirt_a = a_channel[:split_point, :] if split_point > 0 else np.array([])
    pants_a = a_channel[split_point:, :] if h - split_point > 0 else np.array([])
    
    # Compute normalized histograms on A channel
    if shirt_a.size > 0:
        shirt_hist, _ = np.histogram(shirt_a.flatten(), bins=bins, range=(0, 255))
        shirt_hist = shirt_hist.astype(np.float32)
        shirt_hist = shirt_hist / (shirt_hist.sum() + 1e-6)  # Normalize
    else:
        shirt_hist = np.zeros(bins, dtype=np.float32)
    
    if pants_a.size > 0:
        pants_hist, _ = np.histogram(pants_a.flatten(), bins=bins, range=(0, 255))
        pants_hist = pants_hist.astype(np.float32)
        pants_hist = pants_hist / (pants_hist.sum() + 1e-6)  # Normalize
    else:
        pants_hist = np.zeros(bins, dtype=np.float32)
    
    # Compute mean saturation (HSV) for validation
    hsv = cv2.cvtColor(person, cv2.COLOR_BGR2HSV)
    sat_mean = np.mean(hsv[:, :, 1]) / 255.0  # Normalize to 0-1
    
    return {
        "shirt_hist": shirt_hist,
        "pants_hist": pants_hist,
        "aspect": float(aspect),
        "sat_mean": float(sat_mean)
    }


def compute_bhattacharyya_distance(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """
    Compute Bhattacharyya distance between two histograms.
    
    Args:
        hist1: First histogram (normalized)
        hist2: Second histogram (normalized)
    
    Returns:
        Bhattacharyya distance (0 = identical, 1 = completely different)
    """
    # Ensure histograms are normalized
    hist1 = hist1 / (hist1.sum() + 1e-6)
    hist2 = hist2 / (hist2.sum() + 1e-6)
    
    # Bhattacharyya coefficient
    bc = np.sum(np.sqrt(hist1 * hist2))
    
    # Bhattacharyya distance
    if bc > 0:
        bd = -np.log(bc)
        # Normalize to 0-1 range
        bd = min(bd / 5.0, 1.0)
    else:
        bd = 1.0
    
    return float(bd)


def compute_shirt_pants_distance(
    color1: Dict,
    color2: Dict
) -> float:
    """
    Compute weighted distance between shirt and pants descriptors.
    
    Args:
        color1: First color dict (shirt_hist, pants_hist)
        color2: Second color dict (shirt_hist, pants_hist)
    
    Returns:
        Combined color distance (0-1 range)
    """
    shirt_hist1 = color1.get('shirt_hist', np.zeros(16))
    pants_hist1 = color1.get('pants_hist', np.zeros(16))
    shirt_hist2 = color2.get('shirt_hist', np.zeros(16))
    pants_hist2 = color2.get('pants_hist', np.zeros(16))
    
    # Compute Bhattacharyya distance for shirt and pants
    shirt_dist = compute_bhattacharyya_distance(shirt_hist1, shirt_hist2)
    pants_dist = compute_bhattacharyya_distance(pants_hist1, pants_hist2)
    
    # Weight shirt more (60/40 split, matching region split)
    # But we want the cost function to handle weighting, so return separately
    # For backward compatibility, return weighted average
    color_dist = 0.7 * shirt_dist + 0.3 * pants_dist
    
    return float(color_dist)


def compute_iou(
    bbox1: Tuple[float, float, float, float],
    bbox2: Tuple[float, float, float, float]
) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Compute intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Compute union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    iou = intersection / union
    return float(iou)


def get_dominant_color_lab(hist: np.ndarray, bins: int = 16) -> Tuple[int, int, int]:
    """
    Get dominant color from LAB histogram for visualization.
    
    Args:
        hist: LAB A-channel histogram (16 bins)
        bins: Number of bins
    
    Returns:
        BGR color tuple for visualization
    """
    if hist.sum() == 0:
        return (128, 128, 128)  # Gray
    
    # Find bin with max value
    max_bin = np.argmax(hist)
    
    # Convert bin to A channel value (0-255)
    a_value = int((max_bin / bins) * 255)
    
    # Create LAB color with L=128 (medium), A=dominant, B=128 (neutral)
    lab_color = np.uint8([[[128, a_value, 128]]])
    
    # Convert to BGR for visualization
    bgr_color = cv2.cvtColor(lab_color, cv2.COLOR_LAB2BGR)[0][0]
    
    return tuple(map(int, bgr_color))
