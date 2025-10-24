"""
Module: Visualizer BBox V4 (Color-Region Refined)
Visualization with separate shirt and pants color patches.
"""

import cv2
import numpy as np
from typing import Dict, Tuple
from tracking.appearance_extractor_v3 import get_dominant_color_lab


class BBoxVisualizerV4:
    """Visualizer with shirt and pants color patches."""
    
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
    
    def draw_track_bbox_refined(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        track_id: int,
        shirt_hist: np.ndarray,
        pants_hist: np.ndarray,
        conf: float
    ) -> np.ndarray:
        """
        Draw bounding box with track ID and shirt/pants color patches.
        
        Args:
            frame: Input frame
            bbox: Bounding box (x1, y1, x2, y2)
            track_id: Track ID
            shirt_hist: Shirt LAB histogram
            pants_hist: Pants LAB histogram
            conf: Detection confidence
        
        Returns:
            Annotated frame
        """
        x1, y1, x2, y2 = map(int, bbox)
        color = self.get_color_for_id(track_id)
        
        # Draw bounding box
        thickness = 3
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Get dominant colors from LAB histograms
        shirt_color = get_dominant_color_lab(shirt_hist)
        pants_color = get_dominant_color_lab(pants_hist)
        
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
        total_width = label_w + 2 * patch_size + 2 * patch_spacing + 8
        
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
        
        # Draw shirt color patch
        patch_x = x1 + label_w + patch_spacing + 4
        cv2.rectangle(
            frame,
            (patch_x, label_y - label_h - 6),
            (patch_x + patch_size, label_y - label_h - 6 + patch_size),
            shirt_color,
            -1
        )
        cv2.rectangle(
            frame,
            (patch_x, label_y - label_h - 6),
            (patch_x + patch_size, label_y - label_h - 6 + patch_size),
            (255, 255, 255),
            1
        )
        
        # Draw pants color patch
        pants_x = patch_x + patch_size + patch_spacing
        cv2.rectangle(
            frame,
            (pants_x, label_y - label_h - 6),
            (pants_x + patch_size, label_y - label_h - 6 + patch_size),
            pants_color,
            -1
        )
        cv2.rectangle(
            frame,
            (pants_x, label_y - label_h - 6),
            (pants_x + patch_size, label_y - label_h - 6 + patch_size),
            (255, 255, 255),
            1
        )
        
        # Add labels for patches (S=shirt, P=pants)
        label_font_scale = 0.3
        cv2.putText(
            frame,
            "S",
            (patch_x + 5, label_y - label_h + 2),
            font,
            label_font_scale,
            (255, 255, 255),
            1
        )
        cv2.putText(
            frame,
            "P",
            (pants_x + 5, label_y - label_h + 2),
            font,
            label_font_scale,
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


def draw_refined_overlay(
    frame: np.ndarray,
    tracks: list,
    bboxes: dict,
    visualizer: BBoxVisualizerV4
) -> np.ndarray:
    """
    Draw refined tracking visualization with shirt/pants patches.
    
    Args:
        frame: Input frame
        tracks: List of track dictionaries
        bboxes: Dictionary mapping track IDs to bounding boxes
        visualizer: BBoxVisualizerV4 instance
    
    Returns:
        Annotated frame
    """
    annotated = frame.copy()
    
    for track in tracks:
        track_id = track['id']
        
        if track_id not in bboxes:
            continue
        
        bbox = bboxes[track_id]
        shirt_hist = track.get('shirt_hist', np.zeros(16))
        pants_hist = track.get('pants_hist', np.zeros(16))
        
        annotated = visualizer.draw_track_bbox_refined(
            annotated,
            bbox,
            track_id,
            shirt_hist,
            pants_hist,
            track['conf']
        )
    
    return annotated
