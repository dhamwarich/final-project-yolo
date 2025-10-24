"""
Module: Appearance Extractor V3 Cropped (Tight Crop + High Saturation Mask)
Extract shirt/pants colors with tighter cropping and S > 50 masking to avoid gray bias.
"""

import cv2
import numpy as np
from typing import Dict, Tuple


def extract_color_regions_cropped(
    frame: np.ndarray,
    bbox: Tuple[float, float, float, float],
    bins: int = 16,
    sat_threshold: int = 50,
    crop_padding: float = 0.1
) -> Dict:
    """
    Extract HSV colors with tight cropping and high saturation masking.
    
    Args:
        frame: Full video frame (BGR)
        bbox: Bounding box (x1, y1, x2, y2)
        bins: Number of histogram bins (default 16)
        sat_threshold: Minimum saturation (default 50 for vivid colors)
        crop_padding: Padding to crop inward (default 0.1 = 10%)
    
    Returns:
        Dictionary with HSV means and histograms
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Apply tight crop (pad inward to avoid background)
    w, h = x2 - x1, y2 - y1
    pad_x = int(w * crop_padding)
    pad_y = int(h * crop_padding)
    
    x1 += pad_x
    x2 -= pad_x
    y1 += pad_y
    y2 -= pad_y
    
    # Ensure valid bbox after padding
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
    
    h_person, w_person = person.shape[:2]
    aspect = h_person / max(w_person, 1.0)
    
    # Split into shirt (60%) and pants (40%)
    split_point = int(0.6 * h_person)
    shirt_region = person[:split_point, :]
    pants_region = person[split_point:, :]
    
    # Helper function to extract HSV mean with masking
    def mean_hsv(region):
        if region.size == 0:
            return (0.0, 0.0, 0.0), np.array([128, 128, 128], dtype=np.uint8)
        
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)
        
        # High saturation mask (S > 50)
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
    
    # Extract shirt and pants HSV means
    shirt_HSV, shirt_bgr = mean_hsv(shirt_region)
    pants_HSV, pants_bgr = mean_hsv(pants_region)
    
    # Compute histograms for matching (same as before)
    hsv = cv2.cvtColor(person, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    sat_mask = S > sat_threshold
    
    shirt_hue = H[:split_point, :]
    shirt_sat_mask = sat_mask[:split_point, :]
    pants_hue = H[split_point:, :]
    pants_sat_mask = sat_mask[split_point:, :]
    
    if shirt_sat_mask.sum() > 0:
        shirt_hist = cv2.calcHist(
            [shirt_hue],
            [0],
            shirt_sat_mask.astype(np.uint8) * 255,
            [bins],
            [0, 180]
        )
        shirt_hist = shirt_hist.flatten().astype(np.float32)
        shirt_hist = shirt_hist / (shirt_hist.sum() + 1e-6)
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
        pants_hist = pants_hist / (pants_hist.sum() + 1e-6)
    else:
        pants_hist = np.zeros(bins, dtype=np.float32)
    
    # Overall saturation mean
    sat_mean = np.mean(S) / 255.0
    
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
