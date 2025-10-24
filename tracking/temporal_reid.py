"""
Module: Temporal Re-ID (Motion Similarity & Feature Memory)
Focus on temporal coherence using geometry + motion instead of unreliable color.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from collections import deque


class FeatureMemory:
    """
    Feature memory bank for temporal tracking.
    Stores EMA of geometry, motion, and color with age tracking.
    """
    
    def __init__(
        self,
        geom_alpha: float = 0.3,
        motion_alpha: float = 0.3,
        color_alpha: float = 0.1
    ):
        """
        Initialize feature memory.
        
        Args:
            geom_alpha: EMA smoothing for geometry (0.3 = 70% prev, 30% new)
            motion_alpha: EMA smoothing for motion (0.3)
            color_alpha: EMA smoothing for color (0.1 = minimal update)
        """
        self.geom_alpha = geom_alpha
        self.motion_alpha = motion_alpha
        self.color_alpha = color_alpha
        
        # Geometry memory (shoulder_hip, aspect)
        self.shoulder_hip_mem = None
        self.aspect_mem = None
        
        # Motion memory (velocity vector)
        self.velocity_mem = np.array([0.0, 0.0])
        self.prev_center = None
        self.center_history = deque(maxlen=3)  # Last 3 positions
        
        # Color memory (long-term EMA for visualization)
        self.shirt_HSV_mem = None
        self.pants_HSV_mem = None
        self.color_history = deque(maxlen=15)  # Last 15 frames for stable color
        
        # Age tracking
        self.frames_since_update = 0
        self.total_updates = 0
        
    def update_geometry(self, shoulder_hip: float, aspect: float):
        """Update geometry memory with EMA."""
        if self.shoulder_hip_mem is None:
            self.shoulder_hip_mem = shoulder_hip
            self.aspect_mem = aspect
        else:
            alpha = self.geom_alpha
            self.shoulder_hip_mem = alpha * shoulder_hip + (1 - alpha) * self.shoulder_hip_mem
            self.aspect_mem = alpha * aspect + (1 - alpha) * self.aspect_mem
    
    def update_motion(self, center: Tuple[float, float]):
        """Update motion memory with velocity computation."""
        center_arr = np.array(center)
        self.center_history.append(center_arr)
        
        if self.prev_center is not None:
            # Compute velocity (pixels per frame)
            velocity = center_arr - self.prev_center
            
            # Update velocity memory with EMA
            alpha = self.motion_alpha
            self.velocity_mem = alpha * velocity + (1 - alpha) * self.velocity_mem
        
        self.prev_center = center_arr
    
    def update_color(self, shirt_HSV: Tuple[float, float, float], pants_HSV: Tuple[float, float, float]):
        """Update color memory with minimal EMA (for visualization only)."""
        if self.shirt_HSV_mem is None:
            self.shirt_HSV_mem = np.array(shirt_HSV)
            self.pants_HSV_mem = np.array(pants_HSV)
        else:
            alpha = self.color_alpha
            self.shirt_HSV_mem = alpha * np.array(shirt_HSV) + (1 - alpha) * self.shirt_HSV_mem
            self.pants_HSV_mem = alpha * np.array(pants_HSV) + (1 - alpha) * self.pants_HSV_mem
        
        # Add to history for long-term stable color
        self.color_history.append((shirt_HSV, pants_HSV))
    
    def update(self, detection: Dict):
        """Update all features from detection."""
        self.update_geometry(
            detection['shoulder_hip'],
            detection.get('aspect', 1.0)
        )
        self.update_motion(detection['bbox_center'])
        self.update_color(
            detection.get('shirt_HSV', (0, 0, 0)),
            detection.get('pants_HSV', (0, 0, 0))
        )
        
        self.frames_since_update = 0
        self.total_updates += 1
    
    def age(self):
        """Increment age (frames since last update)."""
        self.frames_since_update += 1
    
    def get_stable_color(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Get long-term stable color (average of last 15 frames)."""
        if len(self.color_history) == 0:
            return (0, 0, 0), (0, 0, 0)
        
        # Average over color history
        shirt_H_vals = [c[0][0] for c in self.color_history]
        shirt_S_vals = [c[0][1] for c in self.color_history]
        shirt_V_vals = [c[0][2] for c in self.color_history]
        
        pants_H_vals = [c[1][0] for c in self.color_history]
        pants_S_vals = [c[1][1] for c in self.color_history]
        pants_V_vals = [c[1][2] for c in self.color_history]
        
        stable_shirt = (
            float(np.mean(shirt_H_vals)),
            float(np.mean(shirt_S_vals)),
            float(np.mean(shirt_V_vals))
        )
        
        stable_pants = (
            float(np.mean(pants_H_vals)),
            float(np.mean(pants_S_vals)),
            float(np.mean(pants_V_vals))
        )
        
        return stable_shirt, stable_pants
    
    def get_state(self) -> Dict:
        """Get current memory state."""
        stable_shirt, stable_pants = self.get_stable_color()
        
        return {
            'shoulder_hip': self.shoulder_hip_mem if self.shoulder_hip_mem is not None else 0.0,
            'aspect': self.aspect_mem if self.aspect_mem is not None else 1.0,
            'velocity': tuple(self.velocity_mem),
            'shirt_HSV': tuple(self.shirt_HSV_mem) if self.shirt_HSV_mem is not None else (0, 0, 0),
            'pants_HSV': tuple(self.pants_HSV_mem) if self.pants_HSV_mem is not None else (0, 0, 0),
            'stable_shirt_HSV': stable_shirt,
            'stable_pants_HSV': stable_pants,
            'age': self.frames_since_update,
            'updates': self.total_updates
        }


def compute_motion_similarity(
    velocity1: np.ndarray,
    velocity2: np.ndarray,
    tau: float = 0.2
) -> float:
    """
    Compute motion similarity using velocity vectors.
    
    Args:
        velocity1: Velocity vector 1 (pixels per frame)
        velocity2: Velocity vector 2 (pixels per frame)
        tau: Scale parameter (default 0.2 for ~12 pixels/frame threshold)
    
    Returns:
        Similarity score (0-1, 1 = same motion)
    """
    # Normalize by image diagonal (assume 1920x1080)
    img_diagonal = np.sqrt(1920**2 + 1080**2)
    
    # Compute normalized velocity difference
    delta_v = np.linalg.norm(velocity1 - velocity2) / img_diagonal
    
    # Exponential decay: similar motion -> high score
    similarity = np.exp(-delta_v / tau)
    
    return float(similarity)


def compute_geometry_similarity(
    geom1: Dict,
    geom2: Dict
) -> float:
    """
    Compute geometry similarity (shoulder_hip + aspect).
    
    Args:
        geom1: Geometry dict with 'shoulder_hip' and 'aspect'
        geom2: Geometry dict with 'shoulder_hip' and 'aspect'
    
    Returns:
        Similarity score (0-1, 1 = identical)
    """
    sh1 = geom1.get('shoulder_hip', 0.0)
    sh2 = geom2.get('shoulder_hip', 0.0)
    asp1 = geom1.get('aspect', 1.0)
    asp2 = geom2.get('aspect', 1.0)
    
    # Normalized differences
    sh_diff = abs(sh1 - sh2)
    asp_diff = abs(asp1 - asp2)
    
    # Convert to similarity (1 - normalized_diff)
    sh_sim = max(0, 1.0 - sh_diff * 2.0)  # sh_diff ~0.5 -> 0 similarity
    asp_sim = max(0, 1.0 - asp_diff * 2.0)  # asp_diff ~0.5 -> 0 similarity
    
    # Weighted average (shoulder_hip more important)
    similarity = 0.7 * sh_sim + 0.3 * asp_sim
    
    return float(similarity)


def compute_color_similarity(
    color1: Dict,
    color2: Dict,
    color_conf_threshold: float = 0.4
) -> Tuple[float, bool]:
    """
    Compute color similarity with confidence check.
    
    Args:
        color1: Color dict with 'shirt_HSV', 'pants_HSV', 'shirt_confidence', 'pants_confidence'
        color2: Color dict with similar fields
        color_conf_threshold: Minimum confidence to use color (default 0.4)
    
    Returns:
        Tuple of (similarity score, use_color flag)
    """
    # Check confidence
    conf1_shirt = color1.get('shirt_confidence', 0.0)
    conf1_pants = color1.get('pants_confidence', 0.0)
    conf2_shirt = color2.get('shirt_confidence', 0.0)
    conf2_pants = color2.get('pants_confidence', 0.0)
    
    avg_conf = (conf1_shirt + conf1_pants + conf2_shirt + conf2_pants) / 4.0
    
    # If confidence too low, don't use color
    if avg_conf < color_conf_threshold:
        return 0.0, False
    
    # Compute hue difference (circular distance)
    shirt1_H = color1.get('shirt_HSV', (0, 0, 0))[0]
    shirt2_H = color2.get('shirt_HSV', (0, 0, 0))[0]
    pants1_H = color1.get('pants_HSV', (0, 0, 0))[0]
    pants2_H = color2.get('pants_HSV', (0, 0, 0))[0]
    
    # Circular hue distance (0-180)
    def hue_diff(h1, h2):
        diff = abs(h1 - h2)
        return min(diff, 180 - diff)
    
    shirt_diff = hue_diff(shirt1_H, shirt2_H) / 180.0  # Normalize to 0-1
    pants_diff = hue_diff(pants1_H, pants2_H) / 180.0
    
    # Convert to similarity
    shirt_sim = 1.0 - shirt_diff
    pants_sim = 1.0 - pants_diff
    
    # Weighted average
    similarity = 0.6 * shirt_sim + 0.4 * pants_sim
    
    return float(similarity), True


def compute_temporal_similarity(
    detection: Dict,
    memory: FeatureMemory,
    color_conf_threshold: float = 0.4
) -> float:
    """
    Compute temporal similarity: 0.6*geom + 0.3*motion + 0.1*color.
    
    Args:
        detection: New detection dict
        memory: Feature memory of existing track
        color_conf_threshold: Minimum confidence to use color
    
    Returns:
        Similarity score (0-1)
    """
    mem_state = memory.get_state()
    
    # Geometry similarity
    geom1 = {
        'shoulder_hip': detection['shoulder_hip'],
        'aspect': detection.get('aspect', 1.0)
    }
    geom2 = {
        'shoulder_hip': mem_state['shoulder_hip'],
        'aspect': mem_state['aspect']
    }
    geom_sim = compute_geometry_similarity(geom1, geom2)
    
    # Motion similarity
    det_center = np.array(detection['bbox_center'])
    mem_velocity = np.array(mem_state['velocity'])
    
    # Predicted center based on memory velocity
    predicted_center = memory.prev_center + mem_velocity if memory.prev_center is not None else det_center
    position_diff = det_center - predicted_center
    
    # Motion similarity based on velocity consistency
    motion_sim = compute_motion_similarity(position_diff, np.array([0, 0]), tau=0.2)
    
    # Color similarity (only if confident)
    color1 = {
        'shirt_HSV': detection.get('shirt_HSV', (0, 0, 0)),
        'pants_HSV': detection.get('pants_HSV', (0, 0, 0)),
        'shirt_confidence': detection.get('shirt_confidence', 0.0),
        'pants_confidence': detection.get('pants_confidence', 0.0)
    }
    color2 = {
        'shirt_HSV': mem_state['shirt_HSV'],
        'pants_HSV': mem_state['pants_HSV'],
        'shirt_confidence': 1.0,  # Memory is always confident
        'pants_confidence': 1.0
    }
    color_sim, use_color = compute_color_similarity(color1, color2, color_conf_threshold)
    
    # Weighted combination
    if use_color:
        similarity = 0.6 * geom_sim + 0.3 * motion_sim + 0.1 * color_sim
    else:
        # If color not used, redistribute weight to geometry and motion
        similarity = 0.65 * geom_sim + 0.35 * motion_sim
    
    return float(similarity)
