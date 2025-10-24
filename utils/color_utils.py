"""
Module: Color Utilities
Convert HSV to BGR and map hue values to human-readable color names.
"""

import cv2
import numpy as np
from typing import Tuple


def hsv_to_bgr(h: float, s: int = 200, v: int = 200) -> Tuple[int, int, int]:
    """
    Convert HSV values to BGR color for visualization.
    
    Args:
        h: Hue value (0-180 in OpenCV)
        s: Saturation value (0-255, default 200 for vivid colors)
        v: Value/brightness (0-255, default 200 for visibility)
    
    Returns:
        BGR color tuple
    """
    # Clamp values to valid ranges
    h = int(max(0, min(180, h)))
    s = int(max(0, min(255, s)))
    v = int(max(0, min(255, v)))
    
    # Create HSV color
    hsv_color = np.uint8([[[h, s, v]]])
    
    # Convert to BGR
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0, 0]
    
    return tuple(map(int, bgr_color))


def hue_to_name(h: float) -> str:
    """
    Map hue value to human-readable color name.
    
    Hue ranges (OpenCV 0-180 scale):
    - 0-10, 170-180: Red/Pink
    - 10-25: Orange/Brown
    - 25-35: Yellow
    - 35-85: Green
    - 85-125: Blue
    - 125-145: Violet/Purple
    - 145-170: Magenta
    
    Args:
        h: Hue value (0-180)
    
    Returns:
        Color name string
    """
    h = float(h)
    
    if h < 10 or h >= 170:
        return "red/pink"
    elif h < 25:
        return "orange/brown"
    elif h < 35:
        return "yellow"
    elif h < 85:
        return "green"
    elif h < 125:
        return "blue"
    elif h < 145:
        return "violet/purple"
    elif h < 170:
        return "magenta"
    else:
        return "unknown"


def saturation_to_descriptor(s: float) -> str:
    """
    Map saturation value to descriptive term.
    
    Args:
        s: Saturation value (0-255)
    
    Returns:
        Saturation descriptor
    """
    s = float(s)
    
    if s < 50:
        return "gray"
    elif s < 100:
        return "pale"
    elif s < 150:
        return "moderate"
    elif s < 200:
        return "vibrant"
    else:
        return "vivid"


def hsv_to_full_description(h: float, s: float, v: float) -> str:
    """
    Get full color description from HSV values.
    
    Args:
        h: Hue (0-180)
        s: Saturation (0-255)
        v: Value (0-255)
    
    Returns:
        Full color description (e.g., "vivid blue", "pale yellow")
    """
    color_name = hue_to_name(h)
    sat_desc = saturation_to_descriptor(s)
    
    # For very low saturation, just return "gray"
    if s < 50:
        return "gray"
    
    # For low saturation, mention it
    if s < 100:
        return f"{sat_desc} {color_name}"
    
    # For normal-high saturation, just use color name
    return color_name


def get_contrasting_text_color(bgr: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """
    Get contrasting text color (black or white) for a background color.
    
    Args:
        bgr: Background BGR color
    
    Returns:
        Contrasting BGR color for text
    """
    # Calculate luminance
    b, g, r = bgr
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    
    # Return white text for dark backgrounds, black for light
    return (255, 255, 255) if luminance < 128 else (0, 0, 0)
