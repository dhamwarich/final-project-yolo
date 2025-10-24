"""
Module: Visualizer BBox V5 Temporal (Stable Long-Term Colors)
Visualization with 15-frame EMA colors and temporal metrics.
"""

import cv2
import numpy as np
from typing import Dict, Tuple
from utils.color_utils import hsv_to_bgr


class BBoxVisualizerV5Temporal:
    """Visualizer with long-term stable colors and temporal metrics."""
    
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
    
    def draw_track_bbox_temporal(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        track_id: int,
        stable_shirt_HSV: Tuple[float, float, float],
        stable_pants_HSV: Tuple[float, float, float],
        conf: float,
        hits: int,
        misses: int,
        avg_geom_std: float = 0.0,
        color_conf: float = 0.0
    ) -> np.ndarray:
        """
        Draw bounding box with stable long-term colors.
        
        Args:
            frame: Input frame
            bbox: Bounding box (x1, y1, x2, y2)
            track_id: Track ID
            stable_shirt_HSV: Stable shirt (H, S, V) from 15-frame EMA
            stable_pants_HSV: Stable pants (H, S, V) from 15-frame EMA
            conf: Detection confidence
            hits: Number of successful matches
            misses: Number of missed frames
            avg_geom_std: Average geometry variance
            color_conf: Color confidence (average)
        
        Returns:
            Annotated frame
        """
        x1, y1, x2, y2 = map(int, bbox)
        color = self.get_color_for_id(track_id)
        
        # Grey out if confidence too low
        if conf < 0.3:
            color = (128, 128, 128)
        
        # Draw bounding box
        thickness = 3
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Convert stable HSV to vivid BGR for display
        shirt_H, shirt_S, shirt_V = stable_shirt_HSV
        pants_H, pants_S, pants_V = stable_pants_HSV
        
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
        
        # Draw stable shirt color patch
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
        
        # Draw stable pants color patch
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
        
        # Draw temporal metrics
        metrics_y = label_y + 18
        retention_text = f"Frames:{hits}/{hits+misses}"
        coherence_text = f"Geom:{avg_geom_std:.3f}"
        
        # Calculate info box height
        info_height = 50
        
        cv2.rectangle(
            frame,
            (x1, label_y + 4),
            (x1 + max(100, total_width), metrics_y + info_height),
            (0, 0, 0),
            -1
        )
        
        # Show detection confidence
        cv2.putText(
            frame,
            conf_text,
            (x1 + 4, metrics_y),
            font,
            0.4,
            color,
            1
        )
        
        # Show retention (hits/total)
        retention_pct = hits / max(1, hits + misses)
        retention_color = (0, 255, 0) if retention_pct >= 0.85 else (0, 165, 255)
        cv2.putText(
            frame,
            retention_text,
            (x1 + 4, metrics_y + 15),
            font,
            0.35,
            retention_color,
            1
        )
        
        # Show geometry coherence
        coherence_ok = avg_geom_std <= 0.1
        coherence_color = (0, 255, 0) if coherence_ok else (0, 165, 255)
        cv2.putText(
            frame,
            coherence_text,
            (x1 + 4, metrics_y + 30),
            font,
            0.35,
            coherence_color,
            1
        )
        
        return frame


def draw_temporal_overlay(
    frame: np.ndarray,
    tracks: list,
    bboxes: dict,
    visualizer: BBoxVisualizerV5Temporal
) -> np.ndarray:
    """
    Draw temporal tracking visualization with stable colors.
    
    Args:
        frame: Input frame
        tracks: List of track dictionaries
        bboxes: Dictionary mapping track IDs to bounding boxes
        visualizer: BBoxVisualizerV5Temporal instance
    
    Returns:
        Annotated frame
    """
    annotated = frame.copy()
    
    for track in tracks:
        track_id = track['id']
        
        if track_id not in bboxes:
            continue
        
        bbox = bboxes[track_id]
        
        # Use stable long-term colors (15-frame EMA)
        stable_shirt_HSV = track.get('stable_shirt_HSV', (0, 0, 0))
        stable_pants_HSV = track.get('stable_pants_HSV', (0, 0, 0))
        
        # Get temporal metrics
        hits = track.get('hits', 0)
        misses = track.get('misses', 0)
        avg_geom_std = track.get('avg_geom_std', 0.0)
        
        # Average color confidence
        shirt_conf = track.get('shirt_confidence', 0.0)
        pants_conf = track.get('pants_confidence', 0.0)
        avg_color_conf = (shirt_conf + pants_conf) / 2.0
        
        annotated = visualizer.draw_track_bbox_temporal(
            annotated,
            bbox,
            track_id,
            stable_shirt_HSV,
            stable_pants_HSV,
            track['conf'],
            hits,
            misses,
            avg_geom_std,
            avg_color_conf
        )
    
    return annotated
