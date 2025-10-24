"""
Module 4: Visualize Pose
Render skeleton and feature text on each frame for debugging.
"""

import numpy as np
import cv2


# Define skeleton connections
SKELETON_CONNECTIONS = [
    ('shoulder_l', 'shoulder_r'),
    ('shoulder_l', 'hip_l'),
    ('shoulder_r', 'hip_r'),
    ('hip_l', 'hip_r'),
    ('hip_l', 'knee_l'),
    ('hip_r', 'knee_r'),
    ('knee_l', 'ankle_l'),
    ('knee_r', 'ankle_r'),
    ('nose', 'shoulder_l'),
    ('nose', 'shoulder_r'),
]


def draw_pose(frame: np.ndarray, kps: dict) -> np.ndarray:
    """
    Draw keypoint skeleton.
    
    Args:
        frame: Input frame
        kps: Dictionary of keypoints
    
    Returns:
        Frame with skeleton drawn
    """
    # Create a copy to avoid modifying original
    annotated = frame.copy()
    
    # Draw keypoints
    for key, (x, y) in kps.items():
        if x > 0 and y > 0:  # Only draw if keypoint is detected
            cv2.circle(annotated, (int(x), int(y)), 4, (0, 255, 0), -1)
    
    # Draw skeleton connections
    for point1, point2 in SKELETON_CONNECTIONS:
        if point1 in kps and point2 in kps:
            x1, y1 = kps[point1]
            x2, y2 = kps[point2]
            
            # Only draw if both keypoints are detected
            if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                cv2.line(
                    annotated,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (255, 0, 0),
                    2
                )
    
    return annotated


def overlay_metrics(frame: np.ndarray, features: dict, bbox_center: tuple = None) -> np.ndarray:
    """
    Draw height/ratio text on image near person's bounding box.
    
    Args:
        frame: Input frame
        features: Dictionary of computed features
        bbox_center: Optional (x, y) tuple for text placement. If None, uses top-left
    
    Returns:
        Frame with metrics overlay
    """
    # Create text with metrics
    height = features['height_px']
    sh_ratio = features['shoulder_hip']
    tl_ratio = features['torso_leg']
    
    text = f"H:{height:.0f} SH:{sh_ratio:.2f} TL:{tl_ratio:.2f}"
    
    # Draw text background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    
    # Position text
    if bbox_center is not None:
        # Place above the bounding box center
        x = int(bbox_center[0] - text_width // 2)
        y = int(bbox_center[1] - 40)  # Offset above center
        
        # Ensure text stays within frame bounds
        x = max(5, min(x, frame.shape[1] - text_width - 5))
        y = max(text_height + 5, min(y, frame.shape[0] - 5))
    else:
        # Default to top-left
        x, y = 10, 30
    
    # Draw background rectangle
    cv2.rectangle(
        frame,
        (x - 5, y - text_height - 5),
        (x + text_width + 5, y + baseline + 5),
        (0, 0, 0),
        -1
    )
    
    # Draw text (yellow/cyan tone)
    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        font_scale,
        (0, 255, 255),  # Cyan/yellow tone
        thickness
    )
    
    return frame
