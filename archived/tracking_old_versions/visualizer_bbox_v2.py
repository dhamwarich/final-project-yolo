"""
Module: Visualizer BBox V2 (Appearance-Enhanced)
Color-coded bounding box visualization with color patches showing appearance features.
"""

import cv2
import numpy as np
from typing import Dict, Tuple


class BBoxVisualizerV2:
    """Visualize tracked persons with color patches showing dominant clothing colors."""
    
    def __init__(self):
        """Initialize visualizer with color palette."""
        # Generate distinct colors for different IDs
        self.colors = self._generate_colors(50)
        self.color_cache = {}
    
    def _generate_colors(self, n: int) -> list:
        """Generate n visually distinct colors."""
        colors = []
        for i in range(n):
            # Use HSV color space for better distinction
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
        """
        Convert normalized hue (0-1) to BGR color.
        
        Args:
            hue_norm: Hue value normalized to 0-1
        
        Returns:
            BGR color tuple
        """
        hue = int(hue_norm * 180)  # Convert to OpenCV hue range (0-180)
        color = cv2.cvtColor(
            np.uint8([[[hue, 200, 200]]]),  # Medium saturation/value for visibility
            cv2.COLOR_HSV2BGR
        )[0][0]
        return tuple(map(int, color))
    
    def draw_track_bbox_with_appearance(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        track_id: int,
        height_m: float,
        top_h: float,
        bot_h: float,
        conf: float
    ) -> np.ndarray:
        """
        Draw bounding box with track ID and appearance color patches.
        
        Args:
            frame: Input frame
            bbox: Bounding box (x1, y1, x2, y2)
            track_id: Track ID
            height_m: Estimated height in meters
            top_h: Top clothing hue (0-1)
            bot_h: Bottom clothing hue (0-1)
            conf: Detection confidence
        
        Returns:
            Annotated frame
        """
        x1, y1, x2, y2 = map(int, bbox)
        color = self.get_color_for_id(track_id)
        
        # Draw bounding box
        thickness = 3
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label text
        label = f"ID:{track_id}  H:{height_m:.2f}m"
        conf_text = f"Conf:{conf:.2f}"
        
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
        
        # Draw label background (above bbox)
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
        
        # Draw color patches (top and bottom clothing)
        patch_x_start = x1 + label_w + patch_spacing + 4
        
        # Top clothing patch
        top_color = self.hue_to_bgr(top_h)
        cv2.rectangle(
            frame,
            (patch_x_start, label_y - label_h - 6),
            (patch_x_start + patch_size, label_y - label_h - 6 + patch_size),
            top_color,
            -1
        )
        cv2.rectangle(
            frame,
            (patch_x_start, label_y - label_h - 6),
            (patch_x_start + patch_size, label_y - label_h - 6 + patch_size),
            (255, 255, 255),
            1
        )
        
        # Bottom clothing patch
        bot_color = self.hue_to_bgr(bot_h)
        cv2.rectangle(
            frame,
            (patch_x_start + patch_size + patch_spacing, label_y - label_h - 6),
            (patch_x_start + 2 * patch_size + patch_spacing, label_y - label_h - 6 + patch_size),
            bot_color,
            -1
        )
        cv2.rectangle(
            frame,
            (patch_x_start + patch_size + patch_spacing, label_y - label_h - 6),
            (patch_x_start + 2 * patch_size + patch_spacing, label_y - label_h - 6 + patch_size),
            (255, 255, 255),
            1
        )
        
        # Draw confidence (below main label)
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
    
    def draw_all_tracks(
        self,
        frame: np.ndarray,
        tracks: list,
        bboxes: dict
    ) -> np.ndarray:
        """
        Draw all tracked persons on frame with appearance.
        
        Args:
            frame: Input frame
            tracks: List of track dictionaries
            bboxes: Dictionary mapping track IDs to bounding boxes
        
        Returns:
            Annotated frame
        """
        for track in tracks:
            track_id = track['id']
            if track_id in bboxes:
                frame = self.draw_track_bbox_with_appearance(
                    frame,
                    bboxes[track_id],
                    track_id,
                    track['height_m'],
                    track.get('top_h', 0.0),
                    track.get('bot_h', 0.0),
                    track['conf']
                )
        return frame


def draw_appearance_overlay(
    frame: np.ndarray,
    tracks: list,
    bboxes: dict,
    visualizer: BBoxVisualizerV2
) -> np.ndarray:
    """
    Main function to draw appearance-enhanced tracking visualization.
    
    Args:
        frame: Input frame
        tracks: List of track dictionaries with IDs and appearance features
        bboxes: Dictionary mapping track IDs to (x1, y1, x2, y2)
        visualizer: BBoxVisualizerV2 instance
    
    Returns:
        Annotated frame with bounding boxes and color patches
    """
    annotated = frame.copy()
    
    for track in tracks:
        track_id = track['id']
        
        if track_id not in bboxes:
            continue
        
        bbox = bboxes[track_id]
        
        annotated = visualizer.draw_track_bbox_with_appearance(
            annotated,
            bbox,
            track_id,
            track['height_m'],
            track.get('top_h', 0.0),
            track.get('bot_h', 0.0),
            track['conf']
        )
    
    return annotated
