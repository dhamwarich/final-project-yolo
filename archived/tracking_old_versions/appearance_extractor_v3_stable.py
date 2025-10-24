"""
Module: Appearance Extractor V3 Stable (Center Patch Sampling)
Extract colors from small, fixed-position zones for stability.
Reduces flickering by focusing on torso/thigh centers instead of full regions.
"""

import cv2
import numpy as np
from typing import Dict, Tuple


def extract_center_color_stable(
    frame: np.ndarray,
    bbox: Tuple[float, float, float, float],
    bins: int = 16,
    sat_threshold: int = 50
) -> Dict:
    """
    Extract HSV colors from small center patches for stability.
    
    Center sampling zones:
    - Shirt: 25-45% height, 35-65% width (torso center)
    - Pants: 70-90% height, 35-65% width (thigh center)
    
    Args:
        frame: Full video frame (BGR)
        bbox: Bounding box (x1, y1, x2, y2)
        bins: Number of histogram bins (default 16)
        sat_threshold: Minimum saturation (default 50)
    
    Returns:
        Dictionary with HSV means, histograms, and stability info
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
            "aspect": 1.0,
            "sat_mean": 0.0
        }
    
    aspect = h / max(w, 1.0)
    
    # Define center patch zones (30% width × 20% height)
    shirt_y1 = int(0.25 * h)
    shirt_y2 = int(0.45 * h)
    pants_y1 = int(0.70 * h)
    pants_y2 = int(0.90 * h)
    shirt_x1 = int(0.35 * w)
    shirt_x2 = int(0.65 * w)
    
    # Ensure valid patch coordinates
    shirt_y1 = max(0, min(shirt_y1, h))
    shirt_y2 = max(0, min(shirt_y2, h))
    pants_y1 = max(0, min(pants_y1, h))
    pants_y2 = max(0, min(pants_y2, h))
    shirt_x1 = max(0, min(shirt_x1, w))
    shirt_x2 = max(0, min(shirt_x2, w))
    
    # Extract center patches
    shirt_patch = person[shirt_y1:shirt_y2, shirt_x1:shirt_x2]
    pants_patch = person[pants_y1:pants_y2, shirt_x1:shirt_x2]
    
    # Helper function to extract HSV mean with masking
    def mean_hsv(region):
        if region.size == 0:
            return (0.0, 0.0, 0.0), np.array([128, 128, 128], dtype=np.uint8)
        
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)
        
        # High saturation mask
        mask = S > sat_threshold
        
        if mask.sum() == 0:
            # No colorful pixels, return gray
            return (0.0, 0.0, 0.0), np.array([128, 128, 128], dtype=np.uint8)
        
        # Compute mean HSV with mask
        mean_vals = cv2.mean(hsv, mask=mask.astype(np.uint8) * 255)
        h_mean, s_mean, v_mean = mean_vals[:3]
        
        # Compute mean BGR for visualization
        bgr_mean = cv2.mean(region, mask=mask.astype(np.uint8) * 255)
        bgr = np.array(bgr_mean[:3], dtype=np.uint8)
        
        return (float(h_mean), float(s_mean), float(v_mean)), bgr
    
    # Extract shirt and pants HSV means from center patches
    shirt_HSV, shirt_bgr = mean_hsv(shirt_patch)
    pants_HSV, pants_bgr = mean_hsv(pants_patch)
    
    # Compute histograms for matching (use center patches too)
    hsv_person = cv2.cvtColor(person, cv2.COLOR_BGR2HSV)
    H_person, S_person, V_person = cv2.split(hsv_person)
    sat_mask_person = S_person > sat_threshold
    
    # Shirt histogram from center patch
    if shirt_patch.size > 0:
        hsv_shirt = cv2.cvtColor(shirt_patch, cv2.COLOR_BGR2HSV)
        H_shirt, S_shirt, V_shirt = cv2.split(hsv_shirt)
        shirt_sat_mask = S_shirt > sat_threshold
        
        if shirt_sat_mask.sum() > 0:
            shirt_hist = cv2.calcHist(
                [H_shirt],
                [0],
                shirt_sat_mask.astype(np.uint8) * 255,
                [bins],
                [0, 180]
            )
            shirt_hist = shirt_hist.flatten().astype(np.float32)
            shirt_hist = shirt_hist / (shirt_hist.sum() + 1e-6)
        else:
            shirt_hist = np.zeros(bins, dtype=np.float32)
    else:
        shirt_hist = np.zeros(bins, dtype=np.float32)
    
    # Pants histogram from center patch
    if pants_patch.size > 0:
        hsv_pants = cv2.cvtColor(pants_patch, cv2.COLOR_BGR2HSV)
        H_pants, S_pants, V_pants = cv2.split(hsv_pants)
        pants_sat_mask = S_pants > sat_threshold
        
        if pants_sat_mask.sum() > 0:
            pants_hist = cv2.calcHist(
                [H_pants],
                [0],
                pants_sat_mask.astype(np.uint8) * 255,
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
