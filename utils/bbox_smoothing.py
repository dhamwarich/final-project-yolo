"""
Module: Bounding Box Smoothing
Reduce bbox jitter using exponential moving average (EMA).
"""

import numpy as np
from typing import Tuple, Optional


class BBoxSmoother:
    """Smooth bounding box coordinates using EMA."""
    
    def __init__(self, beta: float = 0.6):
        """
        Initialize bbox smoother.
        
        Args:
            beta: Smoothing factor (0-1). Higher = more smoothing.
                  0.6 means 60% previous, 40% current (reduces jitter by ~70%)
        """
        self.beta = beta
        self.prev_bbox: Optional[np.ndarray] = None
    
    def smooth(self, bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """
        Apply EMA smoothing to bounding box.
        
        Args:
            bbox: Current bounding box (x1, y1, x2, y2)
        
        Returns:
            Smoothed bounding box
        """
        curr_bbox = np.array(bbox, dtype=np.float32)
        
        if self.prev_bbox is None:
            # First frame, no smoothing
            self.prev_bbox = curr_bbox
            return tuple(curr_bbox)
        
        # Apply EMA: smooth = beta * prev + (1-beta) * curr
        smooth_bbox = self.beta * self.prev_bbox + (1 - self.beta) * curr_bbox
        
        # Update previous
        self.prev_bbox = smooth_bbox
        
        return tuple(smooth_bbox)
    
    def reset(self):
        """Reset smoother state."""
        self.prev_bbox = None


def smooth_bbox_ema(
    curr_bbox: Tuple[float, float, float, float],
    prev_bbox: Optional[Tuple[float, float, float, float]],
    beta: float = 0.6
) -> Tuple[float, float, float, float]:
    """
    Standalone function to smooth bbox with EMA.
    
    Args:
        curr_bbox: Current bounding box (x1, y1, x2, y2)
        prev_bbox: Previous bounding box or None
        beta: Smoothing factor (0-1)
    
    Returns:
        Smoothed bounding box
    """
    if prev_bbox is None:
        return curr_bbox
    
    curr = np.array(curr_bbox, dtype=np.float32)
    prev = np.array(prev_bbox, dtype=np.float32)
    
    smooth = beta * prev + (1 - beta) * curr
    
    return tuple(smooth)


def compute_bbox_jitter(
    bboxes: list
) -> float:
    """
    Compute average bbox jitter (frame-to-frame movement).
    
    Args:
        bboxes: List of bounding boxes
    
    Returns:
        Average jitter in pixels
    """
    if len(bboxes) < 2:
        return 0.0
    
    jitters = []
    for i in range(1, len(bboxes)):
        prev = np.array(bboxes[i-1])
        curr = np.array(bboxes[i])
        jitter = np.linalg.norm(curr - prev)
        jitters.append(jitter)
    
    return float(np.mean(jitters))
