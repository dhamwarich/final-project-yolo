"""
Module: Appearance Extractor V2 (Histogram-based)
Extract histogram-based color features and texture for robust Re-ID.
"""

import cv2
import numpy as np
from typing import Dict, Tuple


def extract_histogram_features(
    frame: np.ndarray,
    bbox: Tuple[float, float, float, float],
    bins: int = 16
) -> Dict:
    """
    Extract histogram-based color features from person bbox.
    
    Args:
        frame: Full video frame (BGR)
        bbox: Bounding box (x1, y1, x2, y2)
        bins: Number of histogram bins (default 16)
    
    Returns:
        Dictionary with:
        - top_hist: Top region hue histogram (16 bins, normalized)
        - bot_hist: Bottom region hue histogram (16 bins, normalized)
        - texture: Texture variance
        - aspect: Bbox aspect ratio
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
            "top_hist": np.zeros(bins),
            "bot_hist": np.zeros(bins),
            "texture": 0.0,
            "aspect": 1.0
        }
    
    # Extract person region
    person = frame[y1:y2, x1:x2]
    
    if person.size == 0:
        return {
            "top_hist": np.zeros(bins),
            "bot_hist": np.zeros(bins),
            "texture": 0.0,
            "aspect": 1.0
        }
    
    h, w = person.shape[:2]
    
    # Compute bbox aspect ratio (height/width)
    aspect = h / max(w, 1.0)
    
    # Split into top (60% - torso) and bottom (40% - legs)
    split_point = int(0.6 * h)
    top = person[:split_point, :]
    bottom = person[split_point:, :]
    
    # Convert to HSV for color analysis
    hsv = cv2.cvtColor(person, cv2.COLOR_BGR2HSV)
    
    # Extract hue channel (0-180 in OpenCV)
    hue = hsv[:, :, 0]
    
    # Compute histograms for top and bottom regions
    top_hue = hue[:split_point, :] if split_point > 0 else np.array([])
    bot_hue = hue[split_point:, :] if h - split_point > 0 else np.array([])
    
    # Compute normalized histograms
    if top_hue.size > 0:
        top_hist, _ = np.histogram(top_hue.flatten(), bins=bins, range=(0, 180))
        top_hist = top_hist.astype(np.float32)
        top_hist = top_hist / (top_hist.sum() + 1e-6)  # Normalize
    else:
        top_hist = np.zeros(bins, dtype=np.float32)
    
    if bot_hue.size > 0:
        bot_hist, _ = np.histogram(bot_hue.flatten(), bins=bins, range=(0, 180))
        bot_hist = bot_hist.astype(np.float32)
        bot_hist = bot_hist / (bot_hist.sum() + 1e-6)  # Normalize
    else:
        bot_hist = np.zeros(bins, dtype=np.float32)
    
    # Extract texture using Laplacian variance
    gray = cv2.cvtColor(person, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture = laplacian.var() / 1000.0  # Normalize
    
    return {
        "top_hist": top_hist,
        "bot_hist": bot_hist,
        "texture": float(texture),
        "aspect": float(aspect)
    }


def compute_bhattacharyya_distance(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """
    Compute Bhattacharyya distance between two histograms.
    
    Lower distance = more similar histograms.
    
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
    # BD = -ln(BC), clamped to [0, 1] range
    if bc > 0:
        bd = -np.log(bc)
        # Normalize to 0-1 range (ln(1) = 0, ln(0.01) ≈ 4.6)
        bd = min(bd / 5.0, 1.0)
    else:
        bd = 1.0
    
    return float(bd)


def compute_histogram_color_distance(
    color1: Dict,
    color2: Dict
) -> float:
    """
    Compute distance between two histogram-based color descriptors.
    
    Args:
        color1: First color dict (top_hist, bot_hist)
        color2: Second color dict (top_hist, bot_hist)
    
    Returns:
        Color distance (0-1 range)
    """
    top_hist1 = color1.get('top_hist', np.zeros(16))
    bot_hist1 = color1.get('bot_hist', np.zeros(16))
    top_hist2 = color2.get('top_hist', np.zeros(16))
    bot_hist2 = color2.get('bot_hist', np.zeros(16))
    
    # Compute Bhattacharyya distance for top and bottom
    top_dist = compute_bhattacharyya_distance(top_hist1, top_hist2)
    bot_dist = compute_bhattacharyya_distance(bot_hist1, bot_hist2)
    
    # Weight top more (torso is more visible and stable)
    color_dist = 0.6 * top_dist + 0.4 * bot_dist
    
    return float(color_dist)


def compute_texture_distance(tex1: float, tex2: float) -> float:
    """Compute distance between texture values."""
    max_texture = 10.0
    tex1_norm = min(tex1, max_texture) / max_texture
    tex2_norm = min(tex2, max_texture) / max_texture
    return float(abs(tex1_norm - tex2_norm))


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


def get_dominant_hue(hist: np.ndarray, bins: int = 16) -> float:
    """
    Get dominant hue from histogram for visualization.
    
    Args:
        hist: Hue histogram (16 bins)
        bins: Number of bins
    
    Returns:
        Dominant hue normalized to 0-1
    """
    if hist.sum() == 0:
        return 0.0
    
    # Find bin with max value
    max_bin = np.argmax(hist)
    
    # Convert bin to hue (0-180 → 0-1)
    hue = (max_bin / bins) * 180.0
    hue_norm = hue / 180.0
    
    return float(hue_norm)
