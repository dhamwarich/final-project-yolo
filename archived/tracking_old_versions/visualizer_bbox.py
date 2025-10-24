"""
Module 4: Visualizer BBox
Color-coded bounding box visualization with persistent track IDs.
"""

import cv2
import numpy as np
from typing import Dict, Tuple


class BBoxVisualizer:
    """Visualize tracked persons with color-coded bounding boxes and ID labels."""
    
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
    
    def draw_track_bbox(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        track_id: int,
        height_m: float,
        dist_est: float,
        conf: float
    ) -> np.ndarray:
        """
        Draw bounding box with track ID and metrics.
        
        Args:
            frame: Input frame
            bbox: Bounding box (x1, y1, x2, y2)
            track_id: Track ID
            height_m: Estimated height in meters
            dist_est: Estimated distance in meters
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
        label = f"ID:{track_id}  H:{height_m:.2f}m  D:{dist_est:.2f}m"
        conf_text = f"Conf:{conf:.2f}"
        
        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        
        # Get text sizes
        (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        (conf_w, conf_h), _ = cv2.getTextSize(conf_text, font, font_scale - 0.1, font_thickness - 1)
        
        # Draw label background (above bbox)
        label_y = max(y1 - 10, label_h + 10)
        cv2.rectangle(
            frame,
            (x1, label_y - label_h - 8),
            (x1 + label_w + 8, label_y + 4),
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
        Draw all tracked persons on frame.
        
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
                frame = self.draw_track_bbox(
                    frame,
                    bboxes[track_id],
                    track_id,
                    track['height_m'],
                    bboxes[track_id][4] if len(bboxes[track_id]) > 4 else 0.0,  # dist_est
                    track['conf']
                )
        return frame


def draw_tracking_overlay(
    frame: np.ndarray,
    tracks: list,
    bboxes: dict,
    visualizer: BBoxVisualizer
) -> np.ndarray:
    """
    Main function to draw tracking visualization.
    
    Args:
        frame: Input frame
        tracks: List of track dictionaries with IDs and features
        bboxes: Dictionary mapping track IDs to (x1, y1, x2, y2, dist_est)
        visualizer: BBoxVisualizer instance
    
    Returns:
        Annotated frame with bounding boxes and labels
    """
    annotated = frame.copy()
    
    for track in tracks:
        track_id = track['id']
        
        if track_id not in bboxes:
            continue
        
        bbox_data = bboxes[track_id]
        x1, y1, x2, y2 = bbox_data[:4]
        dist_est = bbox_data[4] if len(bbox_data) > 4 else 0.0
        
        annotated = visualizer.draw_track_bbox(
            annotated,
            (x1, y1, x2, y2),
            track_id,
            track['height_m'],
            dist_est,
            track['conf']
        )
    
    return annotated
