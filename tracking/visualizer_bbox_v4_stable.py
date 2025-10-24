"""
Module: Visualizer BBox V4 Stable (Smoothed Color Patches)
Visualization with temporally smoothed colors to eliminate flicker.
"""

import cv2
import numpy as np
from typing import Dict, Tuple
from utils.color_utils import hsv_to_bgr


class BBoxVisualizerV4Stable:
    """Visualizer with smoothed HSV-based color patches."""
    
    def __init__(self):
        self.colors = self._generate_colors(50)
        self.color_cache = {}
    
    def _generate_colors(self, n: int) -> list:
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
        if track_id not in self.color_cache:
            color_idx = track_id % len(self.colors)
            self.color_cache[track_id] = self.colors[color_idx]
        return self.color_cache[track_id]
    
    def draw_track_bbox_stable(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        track_id: int,
        shirt_HSV: Tuple[float, float, float],
        pants_HSV: Tuple[float, float, float],
        conf: float,
        shirt_H_std: float = 0.0,
        pants_H_std: float = 0.0
    ) -> np.ndarray:
        """
        Draw bounding box with smoothed vivid color patches.
        
        Args:
            frame: Input frame
            bbox: Bounding box (x1, y1, x2, y2)
            track_id: Track ID
            shirt_HSV: Smoothed shirt (H, S, V) tuple
            pants_HSV: Smoothed pants (H, S, V) tuple
            conf: Detection confidence
            shirt_H_std: Shirt hue standard deviation
            pants_H_std: Pants hue standard deviation
        
        Returns:
            Annotated frame
        """
        x1, y1, x2, y2 = map(int, bbox)
        color = self.get_color_for_id(track_id)
        
        # Draw bounding box
        thickness = 3
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Convert smoothed HSV to vivid BGR
        shirt_H, shirt_S, shirt_V = shirt_HSV
        pants_H, pants_S, pants_V = pants_HSV
        
        # Use high saturation and value for vivid display
        shirt_bgr = hsv_to_bgr(shirt_H, s=200, v=200)
        pants_bgr = hsv_to_bgr(pants_H, s=200, v=200)
        
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
            shirt_bgr,
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
            pants_bgr,
            -1
        )
        cv2.rectangle(
            frame,
            (pants_x, label_y - label_h - 6),
            (pants_x + patch_size, label_y - label_h - 6 + patch_size),
            (255, 255, 255),
            1
        )
        
        # Add labels for patches
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
        
        # Draw confidence and stability info
        conf_y = label_y + conf_h + 6
        stability_text = f"σ:{shirt_H_std:.1f}°/{pants_H_std:.1f}°"
        
        cv2.rectangle(
            frame,
            (x1, label_y + 4),
            (x1 + max(conf_w, 80) + 8, conf_y + conf_h + 8),
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
        
        # Show stability (hue variance)
        stability_color = (0, 255, 0) if (shirt_H_std <= 5.0 and pants_H_std <= 5.0) else (0, 165, 255)
        cv2.putText(
            frame,
            stability_text,
            (x1 + 4, conf_y + conf_h + 4),
            font,
            0.35,
            stability_color,
            1
        )
        
        return frame


def draw_stable_overlay(
    frame: np.ndarray,
    tracks: list,
    bboxes: dict,
    visualizer: BBoxVisualizerV4Stable
) -> np.ndarray:
    """
    Draw stable tracking visualization with smoothed colors.
    
    Args:
        frame: Input frame
        tracks: List of track dictionaries
        bboxes: Dictionary mapping track IDs to bounding boxes
        visualizer: BBoxVisualizerV4Stable instance
    
    Returns:
        Annotated frame
    """
    annotated = frame.copy()
    
    for track in tracks:
        track_id = track['id']
        
        if track_id not in bboxes:
            continue
        
        bbox = bboxes[track_id]
        
        # Use smoothed HSV for stable visualization
        shirt_HSV = track.get('smooth_shirt_HSV', (0, 0, 0))
        pants_HSV = track.get('smooth_pants_HSV', (0, 0, 0))
        
        # Get hue variance for stability display
        shirt_H_std = track.get('shirt_H_std', 0.0)
        pants_H_std = track.get('pants_H_std', 0.0)
        
        annotated = visualizer.draw_track_bbox_stable(
            annotated,
            bbox,
            track_id,
            shirt_HSV,
            pants_HSV,
            track['conf'],
            shirt_H_std,
            pants_H_std
        )
    
    return annotated
