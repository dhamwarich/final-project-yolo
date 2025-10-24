"""
Module: Appearance Extractor V3 Normalized (HSV with Saturation Masking)
Extract shirt and pants histograms using HSV with saturation masking to avoid gray bias.
"""

import cv2
import numpy as np
from typing import Dict, Tuple


def extract_color_regions_hsv(
    frame: np.ndarray,
    bbox: Tuple[float, float, float, float],
    bins: int = 16,
    sat_threshold: int = 30
) -> Dict:
    """
    Extract HSV-based color histograms for shirt (upper) and pants (lower) with saturation masking.
    
    Args:
        frame: Full video frame (BGR)
        bbox: Bounding box (x1, y1, x2, y2)
        bins: Number of histogram bins (default 16)
        sat_threshold: Minimum saturation to include (default 30)
    
    Returns:
        Dictionary with:
        - shirt_hist: Upper region HSV histogram (16 bins on Hue)
        - pants_hist: Lower region HSV histogram (16 bins on Hue)
        - shirt_color_mean: Mean BGR color of shirt region
        - pants_color_mean: Mean BGR color of pants region
        - aspect: Bbox aspect ratio
        - sat_mean: Mean saturation
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
            "shirt_hist": np.zeros(bins, dtype=np.float32),
            "pants_hist": np.zeros(bins, dtype=np.float32),
            "shirt_color_mean": np.array([128, 128, 128], dtype=np.uint8),
            "pants_color_mean": np.array([128, 128, 128], dtype=np.uint8),
            "aspect": 1.0,
            "sat_mean": 0.0
        }
    
    # Extract person region
    person = frame[y1:y2, x1:x2]
    
    if person.size == 0:
        return {
            "shirt_hist": np.zeros(bins, dtype=np.float32),
            "pants_hist": np.zeros(bins, dtype=np.float32),
            "shirt_color_mean": np.array([128, 128, 128], dtype=np.uint8),
            "pants_color_mean": np.array([128, 128, 128], dtype=np.uint8),
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
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(person, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    
    # Create saturation mask (exclude low-saturation/gray pixels)
    sat_mask = S > sat_threshold
    
    # Extract shirt region with mask
    shirt_hue = H[:split_point, :]
    shirt_sat_mask = sat_mask[:split_point, :]
    
    # Extract pants region with mask
    pants_hue = H[split_point:, :]
    pants_sat_mask = sat_mask[split_point:, :]
    
    # Compute masked histograms using cv2.calcHist
    if shirt_sat_mask.sum() > 0:
        shirt_hist = cv2.calcHist(
            [shirt_hue], 
            [0], 
            shirt_sat_mask.astype(np.uint8) * 255, 
            [bins], 
            [0, 180]
        )
        shirt_hist = shirt_hist.flatten().astype(np.float32)
        shirt_hist = shirt_hist / (shirt_hist.sum() + 1e-6)  # Normalize
    else:
        shirt_hist = np.zeros(bins, dtype=np.float32)
    
    if pants_sat_mask.sum() > 0:
        pants_hist = cv2.calcHist(
            [pants_hue], 
            [0], 
            pants_sat_mask.astype(np.uint8) * 255, 
            [bins], 
            [0, 180]
        )
        pants_hist = pants_hist.flatten().astype(np.float32)
        pants_hist = pants_hist / (pants_hist.sum() + 1e-6)  # Normalize
    else:
        pants_hist = np.zeros(bins, dtype=np.float32)
    
    # Compute mean colors for visualization (using masked regions)
    shirt_color_mean = cv2.mean(shirt_region, mask=shirt_sat_mask.astype(np.uint8) * 255)[:3]
    shirt_color_mean = np.array(shirt_color_mean, dtype=np.uint8)
    
    pants_color_mean = cv2.mean(pants_region, mask=pants_sat_mask.astype(np.uint8) * 255)[:3]
    pants_color_mean = np.array(pants_color_mean, dtype=np.uint8)
    
    # Compute mean saturation (for validation)
    sat_mean = np.mean(S) / 255.0  # Normalize to 0-1
    
    return {
        "shirt_hist": shirt_hist,
        "pants_hist": pants_hist,
        "shirt_color_mean": shirt_color_mean,
        "pants_color_mean": pants_color_mean,
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
) -> Tuple[float, float]:
    """
    Compute separate distances for shirt and pants descriptors.
    
    Args:
        color1: First color dict (shirt_hist, pants_hist)
        color2: Second color dict (shirt_hist, pants_hist)
    
    Returns:
        Tuple of (shirt_distance, pants_distance)
    """
    shirt_hist1 = color1.get('shirt_hist', np.zeros(16))
    pants_hist1 = color1.get('pants_hist', np.zeros(16))
    shirt_hist2 = color2.get('shirt_hist', np.zeros(16))
    pants_hist2 = color2.get('pants_hist', np.zeros(16))
    
    # Compute Bhattacharyya distance for shirt and pants
    shirt_dist = compute_bhattacharyya_distance(shirt_hist1, shirt_hist2)
    pants_dist = compute_bhattacharyya_distance(pants_hist1, pants_hist2)
    
    return float(shirt_dist), float(pants_dist)


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


def get_dominant_hue(hist: np.ndarray, bins: int = 16) -> int:
    """
    Get dominant hue from histogram for visualization.
    
    Args:
        hist: Hue histogram (16 bins)
        bins: Number of bins
    
    Returns:
        Dominant hue value (0-180)
    """
    if hist.sum() == 0:
        return 0
    
    # Find bin with max value
    max_bin = np.argmax(hist)
    
    # Convert bin to hue value (0-180)
    hue = int((max_bin / bins) * 180)
    
    return hue


def hue_to_bgr(hue: int) -> Tuple[int, int, int]:
    """
    Convert hue value to BGR color for visualization.
    
    Args:
        hue: Hue value (0-180)
    
    Returns:
        BGR color tuple
    """
    # Create HSV color with full saturation and value
    hsv_color = np.uint8([[[hue, 200, 200]]])
    
    # Convert to BGR
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
    
    return tuple(map(int, bgr_color))
