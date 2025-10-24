"""
Module: Appearance Extractor
Extract color and texture features for Re-ID from person bounding box.
"""

import cv2
import numpy as np
from typing import Dict, Tuple


def extract_appearance_features(
    frame: np.ndarray,
    bbox: Tuple[float, float, float, float]
) -> Dict[str, float]:
    """
    Extract dominant clothing colors and texture cues from person bbox.
    
    Args:
        frame: Full video frame (BGR)
        bbox: Bounding box (x1, y1, x2, y2)
    
    Returns:
        Dictionary with:
        - top_h: Top region hue (normalized 0-1)
        - bot_h: Bottom region hue (normalized 0-1)
        - texture: Texture variance (Laplacian)
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Ensure valid bbox
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    
    if x2 <= x1 or y2 <= y1:
        # Invalid bbox
        return {"top_h": 0.0, "bot_h": 0.0, "texture": 0.0}
    
    # Extract person region
    person = frame[y1:y2, x1:x2]
    
    if person.size == 0:
        return {"top_h": 0.0, "bot_h": 0.0, "texture": 0.0}
    
    h, w = person.shape[:2]
    
    # Split into top (torso) and bottom (legs)
    # Top 60% is torso/upper body, bottom 40% is legs
    split_point = int(0.6 * h)
    top = person[:split_point, :]
    bottom = person[split_point:, :]
    
    # Convert to HSV for color analysis
    hsv = cv2.cvtColor(person, cv2.COLOR_BGR2HSV)
    
    # Extract mean hue from top and bottom regions
    if split_point > 0:
        top_hsv = np.mean(hsv[:split_point, :, 0])
    else:
        top_hsv = 0.0
    
    if h - split_point > 0:
        bot_hsv = np.mean(hsv[split_point:, :, 0])
    else:
        bot_hsv = 0.0
    
    # Normalize hue to 0-1 (OpenCV hue is 0-180)
    top_h_norm = top_hsv / 180.0
    bot_h_norm = bot_hsv / 180.0
    
    # Extract texture using Laplacian variance
    gray = cv2.cvtColor(person, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture = laplacian.var() / 1000.0  # Normalize
    
    return {
        "top_h": float(top_h_norm),
        "bot_h": float(bot_h_norm),
        "texture": float(texture)
    }


def compute_color_distance(
    color1: Dict[str, float],
    color2: Dict[str, float]
) -> float:
    """
    Compute distance between two color descriptors.
    
    Args:
        color1: First color dict (top_h, bot_h)
        color2: Second color dict (top_h, bot_h)
    
    Returns:
        Color distance (0-1 range)
    """
    # Hue is circular (0 == 1), so use circular distance
    def circular_dist(h1, h2):
        diff = abs(h1 - h2)
        return min(diff, 1.0 - diff)
    
    top_dist = circular_dist(color1.get('top_h', 0), color2.get('top_h', 0))
    bot_dist = circular_dist(color1.get('bot_h', 0), color2.get('bot_h', 0))
    
    # Weight top more (torso is more visible and stable)
    color_dist = 0.6 * top_dist + 0.4 * bot_dist
    
    return float(color_dist)


def compute_texture_distance(
    tex1: float,
    tex2: float
) -> float:
    """
    Compute distance between texture values.
    
    Args:
        tex1: First texture variance
        tex2: Second texture variance
    
    Returns:
        Normalized texture distance
    """
    # Normalize by max expected texture variance
    max_texture = 10.0
    tex1_norm = min(tex1, max_texture) / max_texture
    tex2_norm = min(tex2, max_texture) / max_texture
    
    return float(abs(tex1_norm - tex2_norm))


def compute_iou(
    bbox1: Tuple[float, float, float, float],
    bbox2: Tuple[float, float, float, float]
) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1: First bbox (x1, y1, x2, y2)
        bbox2: Second bbox (x1, y1, x2, y2)
    
    Returns:
        IoU value (0-1)
    """
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
