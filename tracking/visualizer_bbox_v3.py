"""
Module: Visualizer BBox V3 (Tuned, Simplified)
Simplified visualization with dominant color patch from histogram.
"""

import cv2
import numpy as np
from typing import Dict, Tuple
from tracking.appearance_extractor_v2 import get_dominant_hue


class BBoxVisualizerV3:
    """Simplified visualizer for tuned tracking without height display."""
    
    def __init__(self):
        """Initialize visualizer with color palette."""
        self.colors = self._generate_colors(50)
        self.color_cache = {}
    
    def _generate_colors(self, n: int) -> list:
        """Generate n visually distinct colors."""
        colors = []
        for i in range(n):
            hue = int(180 * i / n)
            color = cv2.cvtColor(
                np.uint8([[[hue, 255, 255]]]),
                cv2.COLOR_HSV2BGR
            )[0][0]
            colors.append(tuple(map(int, color)))
        return colors
    
    def get_color_for_id(self, track_id: int) -> Tuple[int, int, int]:
        """Get persistent color for a track ID."""
        if track_id not in self.color_cache:
            color_idx = track_id % len(self.colors)
            self.color_cache[track_id] = self.colors[color_idx]
        return self.color_cache[track_id]
    
    def hue_to_bgr(self, hue_norm: float) -> Tuple[int, int, int]:
        """Convert normalized hue (0-1) to BGR color."""
        hue = int(hue_norm * 180)
        color = cv2.cvtColor(
            np.uint8([[[hue, 200, 200]]]),
            cv2.COLOR_HSV2BGR
        )[0][0]
        return tuple(map(int, color))
    
    def draw_track_bbox_tuned(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        track_id: int,
        top_hist: np.ndarray,
        conf: float
    ) -> np.ndarray:
        """
        Draw simplified bounding box with track ID and color patch.
        
        Args:
            frame: Input frame
            bbox: Bounding box (x1, y1, x2, y2)
            track_id: Track ID
            top_hist: Top clothing histogram
            conf: Detection confidence
        
        Returns:
            Annotated frame
        """
        x1, y1, x2, y2 = map(int, bbox)
        color = self.get_color_for_id(track_id)
        
        # Draw bounding box
        thickness = 3
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Get dominant hue from histogram
        dominant_hue = get_dominant_hue(top_hist)
        clothing_color = self.hue_to_bgr(dominant_hue)
        
        # Prepare label
        label = f"ID:{track_id}"
        conf_text = f"C:{conf:.2f}"
        
        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        
        # Get text sizes
        (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        (conf_w, conf_h), _ = cv2.getTextSize(conf_text, font, font_scale - 0.1, font_thickness - 1)
        
        # Color patch size
        patch_size = 20
        patch_spacing = 5
        
        # Draw label background
        label_y = max(y1 - 10, label_h + 10)
        total_width = label_w + patch_size + patch_spacing + 8
        
        cv2.rectangle(
            frame,
            (x1, label_y - label_h - 8),
            (x1 + total_width, label_y + 4),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            frame,
            label,
            (x1 + 4, label_y - 4),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness
        )
        
        # Draw clothing color patch
        patch_x = x1 + label_w + patch_spacing + 4
        cv2.rectangle(
            frame,
            (patch_x, label_y - label_h - 6),
            (patch_x + patch_size, label_y - label_h - 6 + patch_size),
            clothing_color,
            -1
        )
        cv2.rectangle(
            frame,
            (patch_x, label_y - label_h - 6),
            (patch_x + patch_size, label_y - label_h - 6 + patch_size),
            (255, 255, 255),
            1
        )
        
        # Draw confidence
        conf_y = label_y + conf_h + 6
        cv2.rectangle(
            frame,
            (x1, label_y + 4),
            (x1 + conf_w + 8, conf_y + 4),
            (0, 0, 0),
            -1
        )
        
        cv2.putText(
            frame,
            conf_text,
            (x1 + 4, conf_y),
            font,
            font_scale - 0.1,
            color,
            font_thickness - 1
        )
        
        return frame


def draw_tuned_overlay(
    frame: np.ndarray,
    tracks: list,
    bboxes: dict,
    visualizer: BBoxVisualizerV3
) -> np.ndarray:
    """
    Draw tuned tracking visualization.
    
    Args:
        frame: Input frame
        tracks: List of track dictionaries
        bboxes: Dictionary mapping track IDs to bounding boxes
        visualizer: BBoxVisualizerV3 instance
    
    Returns:
        Annotated frame
    """
    annotated = frame.copy()
    
    for track in tracks:
        track_id = track['id']
        
        if track_id not in bboxes:
            continue
        
        bbox = bboxes[track_id]
        top_hist = track.get('top_hist', np.zeros(16))
        
        annotated = visualizer.draw_track_bbox_tuned(
            annotated,
            bbox,
            track_id,
            top_hist,
            track['conf']
        )
    
    return annotated
