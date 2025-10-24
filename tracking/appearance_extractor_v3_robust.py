"""
Module: Appearance Extractor V3 Robust (Expanded Regions + SV Gating)
Extract colors from larger regions with saturation/brightness filtering.
Improves shirt color stability under pose and lighting variation.
"""

import cv2
import numpy as np
from typing import Dict, Tuple


def extract_color_robust(
    frame: np.ndarray,
    bbox: Tuple[float, float, float, float],
    bins: int = 16,
    sat_threshold: int = 60,
    val_min: int = 40,
    val_max: int = 220
) -> Dict:
    """
    Extract HSV colors from expanded regions with SV gating.
    
    Sampling zones (expanded from center patches):
    - Shirt: 20-60% height, 25-75% width (40% × 50% = larger, robust sampling)
    - Pants: 65-90% height, 25-75% width (25% × 50%)
    
    SV Gating:
    - S > 60: Skip low-saturation (gray/white) pixels
    - V ∈ [40, 220]: Skip very dark shadows and bright highlights
    
    Args:
        frame: Full video frame (BGR)
        bbox: Bounding box (x1, y1, x2, y2)
        bins: Number of histogram bins (default 16)
        sat_threshold: Minimum saturation (default 60, stricter than before)
        val_min: Minimum value/brightness (default 40)
        val_max: Maximum value/brightness (default 220)
    
    Returns:
        Dictionary with HSV means, histograms, confidence, and stability info
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Ensure valid bbox
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    
    if x2 <= x1 or y2 <= y1:
        # Invalid bbox
        return {
            "shirt_hist": np.zeros(bins, dtype=np.float32),
            "pants_hist": np.zeros(bins, dtype=np.float32),
            "shirt_HSV": (0.0, 0.0, 0.0),
            "pants_HSV": (0.0, 0.0, 0.0),
            "shirt_color_bgr": np.array([128, 128, 128], dtype=np.uint8),
            "pants_color_bgr": np.array([128, 128, 128], dtype=np.uint8),
            "shirt_confidence": 0.0,
            "pants_confidence": 0.0,
            "aspect": 1.0,
            "sat_mean": 0.0
        }
    
    # Extract person region
    w = x2 - x1
    h = y2 - y1
    person = frame[y1:y2, x1:x2]
    
    if person.size == 0:
        return {
            "shirt_hist": np.zeros(bins, dtype=np.float32),
            "pants_hist": np.zeros(bins, dtype=np.float32),
            "shirt_HSV": (0.0, 0.0, 0.0),
            "pants_HSV": (0.0, 0.0, 0.0),
            "shirt_color_bgr": np.array([128, 128, 128], dtype=np.uint8),
            "pants_color_bgr": np.array([128, 128, 128], dtype=np.uint8),
            "shirt_confidence": 0.0,
            "pants_confidence": 0.0,
            "aspect": 1.0,
            "sat_mean": 0.0
        }
    
    aspect = h / max(w, 1.0)
    
    # Define expanded sampling zones (larger than center patches)
    # Shirt: 20-60% height (40% of body), 25-75% width (50%)
    shirt_y1 = int(0.20 * h)
    shirt_y2 = int(0.60 * h)
    # Pants: 65-90% height (25% of body), 25-75% width (50%)
    pants_y1 = int(0.65 * h)
    pants_y2 = int(0.90 * h)
    shirt_x1 = int(0.25 * w)
    shirt_x2 = int(0.75 * w)
    
    # Ensure valid patch coordinates
    shirt_y1 = max(0, min(shirt_y1, h))
    shirt_y2 = max(0, min(shirt_y2, h))
    pants_y1 = max(0, min(pants_y1, h))
    pants_y2 = max(0, min(pants_y2, h))
    shirt_x1 = max(0, min(shirt_x1, w))
    shirt_x2 = max(0, min(shirt_x2, w))
    
    # Extract regions
    shirt_region = person[shirt_y1:shirt_y2, shirt_x1:shirt_x2]
    pants_region = person[pants_y1:pants_y2, shirt_x1:shirt_x2]
    
    # Helper function to extract HSV mean with SV gating
    def mean_hsv_with_confidence(region):
        if region.size == 0:
            return (0.0, 0.0, 0.0), np.array([128, 128, 128], dtype=np.uint8), 0.0
        
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)
        
        # Saturation and value gating
        S_mask = S > sat_threshold
        V_mask = (V > val_min) & (V < val_max)
        mask = S_mask & V_mask
        
        if mask.sum() == 0:
            # No valid pixels, return gray with zero confidence
            return (0.0, 0.0, 0.0), np.array([128, 128, 128], dtype=np.uint8), 0.0
        
        # Compute mean HSV with mask
        mean_vals = cv2.mean(hsv, mask=mask.astype(np.uint8) * 255)
        h_mean, s_mean, v_mean = mean_vals[:3]
        
        # Compute mean BGR for visualization
        bgr_mean = cv2.mean(region, mask=mask.astype(np.uint8) * 255)
        bgr = np.array(bgr_mean[:3], dtype=np.uint8)
        
        # Confidence based on saturation (0-1 scale)
        confidence = np.clip(s_mean / 128.0, 0.0, 1.0)
        
        return (float(h_mean), float(s_mean), float(v_mean)), bgr, float(confidence)
    
    # Extract shirt and pants HSV means with confidence
    shirt_HSV, shirt_bgr, shirt_conf = mean_hsv_with_confidence(shirt_region)
    pants_HSV, pants_bgr, pants_conf = mean_hsv_with_confidence(pants_region)
    
    # Compute histograms for matching
    hsv_person = cv2.cvtColor(person, cv2.COLOR_BGR2HSV)
    H_person, S_person, V_person = cv2.split(hsv_person)
    
    # Shirt histogram from expanded region
    if shirt_region.size > 0:
        hsv_shirt = cv2.cvtColor(shirt_region, cv2.COLOR_BGR2HSV)
        H_shirt, S_shirt, V_shirt = cv2.split(hsv_shirt)
        
        S_mask_shirt = S_shirt > sat_threshold
        V_mask_shirt = (V_shirt > val_min) & (V_shirt < val_max)
        shirt_mask = S_mask_shirt & V_mask_shirt
        
        if shirt_mask.sum() > 0:
            shirt_hist = cv2.calcHist(
                [H_shirt],
                [0],
                shirt_mask.astype(np.uint8) * 255,
                [bins],
                [0, 180]
            )
            shirt_hist = shirt_hist.flatten().astype(np.float32)
            shirt_hist = shirt_hist / (shirt_hist.sum() + 1e-6)
        else:
            shirt_hist = np.zeros(bins, dtype=np.float32)
    else:
        shirt_hist = np.zeros(bins, dtype=np.float32)
    
    # Pants histogram from expanded region
    if pants_region.size > 0:
        hsv_pants = cv2.cvtColor(pants_region, cv2.COLOR_BGR2HSV)
        H_pants, S_pants, V_pants = cv2.split(hsv_pants)
        
        S_mask_pants = S_pants > sat_threshold
        V_mask_pants = (V_pants > val_min) & (V_pants < val_max)
        pants_mask = S_mask_pants & V_mask_pants
        
        if pants_mask.sum() > 0:
            pants_hist = cv2.calcHist(
                [H_pants],
                [0],
                pants_mask.astype(np.uint8) * 255,
                [bins],
                [0, 180]
            )
            pants_hist = pants_hist.flatten().astype(np.float32)
            pants_hist = pants_hist / (pants_hist.sum() + 1e-6)
        else:
            pants_hist = np.zeros(bins, dtype=np.float32)
    else:
        pants_hist = np.zeros(bins, dtype=np.float32)
    
    # Overall saturation mean
    sat_mean = np.mean(S_person) / 255.0
    
    return {
        "shirt_hist": shirt_hist,
        "pants_hist": pants_hist,
        "shirt_HSV": shirt_HSV,
        "pants_HSV": pants_HSV,
        "shirt_color_bgr": shirt_bgr,
        "pants_color_bgr": pants_bgr,
        "shirt_confidence": shirt_conf,
        "pants_confidence": pants_conf,
        "aspect": float(aspect),
        "sat_mean": float(sat_mean)
    }


def compute_bhattacharyya_distance(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """Compute Bhattacharyya distance between two histograms."""
    hist1 = hist1 / (hist1.sum() + 1e-6)
    hist2 = hist2 / (hist2.sum() + 1e-6)
    
    bc = np.sum(np.sqrt(hist1 * hist2))
    
    if bc > 0:
        bd = -np.log(bc)
        bd = min(bd / 5.0, 1.0)
    else:
        bd = 1.0
    
    return float(bd)


def compute_shirt_pants_distance(
    color1: Dict,
    color2: Dict
) -> Tuple[float, float]:
    """Compute separate distances for shirt and pants."""
    shirt_hist1 = color1.get('shirt_hist', np.zeros(16))
    pants_hist1 = color1.get('pants_hist', np.zeros(16))
    shirt_hist2 = color2.get('shirt_hist', np.zeros(16))
    pants_hist2 = color2.get('pants_hist', np.zeros(16))
    
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
    
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return float(intersection / union)
