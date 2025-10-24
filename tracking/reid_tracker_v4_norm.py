"""
Module: Re-ID Tracker V4 Normalized (HSV with Color Logging)
Refined tracking with HSV shirt/pants histograms and per-track color accumulation.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Optional, Tuple
from tracking.appearance_extractor_v3_norm import (
    compute_bhattacharyya_distance,
    compute_iou
)
import time


class TrackV4Norm:
    """Track with HSV shirt/pants histograms and color accumulation."""
    
    def __init__(self, track_id: int, detection: Dict):
        """
        Initialize a new track.
        
        Args:
            track_id: Unique identifier for this track
            detection: Initial detection dictionary
        """
        self.id = track_id
        self.age = 0
        self.hits = 1
        self.misses = 0
        self.max_age = 30
        
        # Geometric features
        self.shoulder_hip = detection.get('shoulder_hip', 0.0)
        self.aspect = detection.get('aspect', 1.0)
        self.bbox_center = detection.get('bbox_center', (0, 0))
        self.bbox = detection.get('bbox', (0, 0, 0, 0))
        self.conf = detection.get('conf', 0.0)
        
        # HSV-based appearance
        self.shirt_hist = detection.get('shirt_hist', np.zeros(16))
        self.pants_hist = detection.get('pants_hist', np.zeros(16))
        self.sat_mean = detection.get('sat_mean', 0.0)
        
        # Color accumulation for logging (BGR)
        self.shirt_color_mean = detection.get('shirt_color_mean', np.array([128, 128, 128]))
        self.pants_color_mean = detection.get('pants_color_mean', np.array([128, 128, 128]))
        self.color_samples = []  # List of (shirt_bgr, pants_bgr)
        self.color_samples.append((self.shirt_color_mean.copy(), self.pants_color_mean.copy()))
        
        # Feature history for EMA smoothing (keep last 5)
        self.shirt_hist_history = [self.shirt_hist.copy()]
        self.pants_hist_history = [self.pants_hist.copy()]
        
    def update(self, detection: Dict, alpha: float = 0.2):
        """
        Update track with new detection using EMA.
        
        Args:
            detection: New detection dictionary
            alpha: EMA smoothing factor (0.2 = 5-frame average)
        """
        self.age = 0
        self.hits += 1
        self.misses = 0
        
        # Update geometric features with EMA
        self.shoulder_hip = alpha * detection['shoulder_hip'] + (1 - alpha) * self.shoulder_hip
        self.aspect = alpha * detection.get('aspect', self.aspect) + (1 - alpha) * self.aspect
        
        # Update HSV histogram-based appearance with EMA
        new_shirt_hist = detection.get('shirt_hist', self.shirt_hist)
        new_pants_hist = detection.get('pants_hist', self.pants_hist)
        
        self.shirt_hist = alpha * new_shirt_hist + (1 - alpha) * self.shirt_hist
        self.pants_hist = alpha * new_pants_hist + (1 - alpha) * self.pants_hist
        
        # Normalize histograms after EMA
        self.shirt_hist = self.shirt_hist / (self.shirt_hist.sum() + 1e-6)
        self.pants_hist = self.pants_hist / (self.pants_hist.sum() + 1e-6)
        
        self.sat_mean = alpha * detection.get('sat_mean', self.sat_mean) + (1 - alpha) * self.sat_mean
        
        # Accumulate color samples
        shirt_color = detection.get('shirt_color_mean', self.shirt_color_mean)
        pants_color = detection.get('pants_color_mean', self.pants_color_mean)
        self.color_samples.append((shirt_color.copy(), pants_color.copy()))
        
        # Keep last 50 samples for averaging
        if len(self.color_samples) > 50:
            self.color_samples.pop(0)
        
        # Update current color means
        self.shirt_color_mean = shirt_color
        self.pants_color_mean = pants_color
        
        # Update position (no smoothing for bbox)
        self.bbox_center = detection['bbox_center']
        self.bbox = detection.get('bbox', self.bbox)
        self.conf = detection['conf']
        
        # Update history (keep last 5)
        self.shirt_hist_history.append(self.shirt_hist.copy())
        self.pants_hist_history.append(self.pants_hist.copy())
        
        if len(self.shirt_hist_history) > 5:
            self.shirt_hist_history.pop(0)
            self.pants_hist_history.pop(0)
    
    def get_average_colors(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get average shirt and pants colors across all samples.
        
        Returns:
            Tuple of (avg_shirt_bgr, avg_pants_bgr)
        """
        if len(self.color_samples) == 0:
            return np.array([128, 128, 128]), np.array([128, 128, 128])
        
        shirt_colors = [s[0] for s in self.color_samples]
        pants_colors = [s[1] for s in self.color_samples]
        
        avg_shirt = np.mean(shirt_colors, axis=0).astype(np.uint8)
        avg_pants = np.mean(pants_colors, axis=0).astype(np.uint8)
        
        return avg_shirt, avg_pants
    
    def mark_missed(self):
        """Mark track as missed in current frame."""
        self.age += 1
        self.misses += 1
    
    def is_dead(self) -> bool:
        """Check if track should be deleted."""
        return self.age > self.max_age
    
    def get_state(self) -> Dict:
        """Get current track state as dictionary."""
        avg_shirt, avg_pants = self.get_average_colors()
        
        return {
            'id': self.id,
            'shoulder_hip': self.shoulder_hip,
            'aspect': self.aspect,
            'bbox_center': self.bbox_center,
            'bbox': self.bbox,
            'conf': self.conf,
            'shirt_hist': self.shirt_hist.copy(),
            'pants_hist': self.pants_hist.copy(),
            'shirt_color_mean': self.shirt_color_mean.copy(),
            'pants_color_mean': self.pants_color_mean.copy(),
            'avg_shirt_color': avg_shirt,
            'avg_pants_color': avg_pants,
            'sat_mean': self.sat_mean,
            'hits': self.hits,
            'age': self.age
        }


class ReIDTrackerV4Norm:
    """Normalized Re-ID tracker with HSV shirt/pants and color logging."""
    
    def __init__(
        self,
        max_cost_threshold: float = 0.45,
        memory_cost_threshold: float = 0.30,
        memory_duration: float = 5.0,
        img_width: int = 1920,
        img_height: int = 1080
    ):
        """
        Initialize tracker.
        
        Args:
            max_cost_threshold: Maximum cost for valid matching
            memory_cost_threshold: Lower threshold for memory matching
            memory_duration: How long to keep lost tracks (seconds)
            img_width: Image width for normalization
            img_height: Image height for normalization
        """
        self.tracks: List[TrackV4Norm] = []
        self.next_id = 0
        self.max_cost_threshold = max_cost_threshold
        self.memory_cost_threshold = memory_cost_threshold
        self.memory_duration = memory_duration
        self.img_diagonal = np.sqrt(img_width**2 + img_height**2)
        
        # Short-term memory
        self.track_memory: Dict[int, tuple] = {}
        
    def compute_cost_matrix(
        self,
        detections: List[Dict],
        tracks: List[TrackV4Norm]
    ) -> np.ndarray:
        """
        Compute cost matrix with HSV shirt/pants separation.
        
        Cost formula (Week 5.5):
            cost = 0.25 × |SH_det - SH_track|
                 + 0.15 × |aspect_det - aspect_track|
                 + 0.35 × shirt_distance
                 + 0.15 × pants_distance
                 + 0.10 × motion_cost(IoU, center)
        
        Args:
            detections: List of detection dictionaries
            tracks: List of TrackV4Norm objects
        
        Returns:
            Cost matrix (n_detections x n_tracks)
        """
        if len(detections) == 0 or len(tracks) == 0:
            return np.array([]).reshape(len(detections), len(tracks))
        
        cost_matrix = np.zeros((len(detections), len(tracks)))
        
        for i, det in enumerate(detections):
            for j, track in enumerate(tracks):
                # 1. Shoulder/hip ratio difference
                sh_diff = abs(det['shoulder_hip'] - track.shoulder_hip)
                
                # 2. Aspect ratio difference
                aspect_diff = abs(det.get('aspect', 1.0) - track.aspect)
                
                # 3. Shirt (upper body) color distance
                shirt_dist = compute_bhattacharyya_distance(
                    det.get('shirt_hist', np.zeros(16)),
                    track.shirt_hist
                )
                
                # 4. Pants (lower body) color distance
                pants_dist = compute_bhattacharyya_distance(
                    det.get('pants_hist', np.zeros(16)),
                    track.pants_hist
                )
                
                # 5. Motion cost (IoU + center proximity)
                det_bbox = det.get('bbox', (0, 0, 0, 0))
                iou = compute_iou(det_bbox, track.bbox)
                
                det_center = np.array(det['bbox_center'])
                track_center = np.array(track.bbox_center)
                center_dist = np.linalg.norm(det_center - track_center) / self.img_diagonal
                
                motion_cost = 0.5 * (1 - iou) + 0.5 * center_dist
                
                # Weighted fusion (Week 5.5 weights)
                cost = (
                    0.25 * sh_diff +
                    0.15 * aspect_diff +
                    0.35 * shirt_dist +
                    0.15 * pants_dist +
                    0.10 * motion_cost
                )
                
                cost_matrix[i, j] = cost
        
        return cost_matrix
    
    def match_tracks(
        self,
        detections: List[Dict],
        tracks: List[TrackV4Norm]
    ) -> tuple:
        """Match detections to existing tracks using Hungarian algorithm."""
        if len(detections) == 0:
            return [], [], list(range(len(tracks)))
        
        if len(tracks) == 0:
            return [], list(range(len(detections))), []
        
        # Compute cost matrix
        cost_matrix = self.compute_cost_matrix(detections, tracks)
        
        # Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Filter by cost threshold
        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(range(len(tracks)))
        
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < self.max_cost_threshold:
                matched.append((i, j))
                if i in unmatched_dets:
                    unmatched_dets.remove(i)
                if j in unmatched_tracks:
                    unmatched_tracks.remove(j)
        
        return matched, unmatched_dets, unmatched_tracks
    
    def try_match_from_memory(self, detection: Dict) -> Optional[int]:
        """Try to match detection with recently lost tracks in memory."""
        if len(self.track_memory) == 0:
            return None
        
        current_time = time.time()
        best_id = None
        best_cost = self.memory_cost_threshold
        
        # Check all tracks in memory
        for track_id, (track_state, timestamp) in list(self.track_memory.items()):
            # Check if memory expired
            if current_time - timestamp > self.memory_duration:
                del self.track_memory[track_id]
                continue
            
            # Compute cost to memory track (no motion component)
            sh_diff = abs(detection['shoulder_hip'] - track_state['shoulder_hip'])
            aspect_diff = abs(detection.get('aspect', 1.0) - track_state['aspect'])
            
            shirt_dist = compute_bhattacharyya_distance(
                detection.get('shirt_hist', np.zeros(16)),
                track_state['shirt_hist']
            )
            
            pants_dist = compute_bhattacharyya_distance(
                detection.get('pants_hist', np.zeros(16)),
                track_state['pants_hist']
            )
            
            # Memory matching uses same weights but no motion
            cost = (
                0.30 * sh_diff +
                0.15 * aspect_diff +
                0.40 * shirt_dist +
                0.15 * pants_dist
            )
            
            if cost < best_cost:
                best_cost = cost
                best_id = track_id
        
        return best_id
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """Update tracker with new detections and memory management."""
        # Match detections to existing tracks
        matched, unmatched_dets, unmatched_tracks = self.match_tracks(
            detections, self.tracks
        )
        
        # Update matched tracks
        for det_idx, track_idx in matched:
            self.tracks[track_idx].update(detections[det_idx])
        
        # Mark unmatched tracks as missed
        current_time = time.time()
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
            
            # If track just died, store in memory
            if self.tracks[track_idx].is_dead():
                track_state = self.tracks[track_idx].get_state()
                self.track_memory[self.tracks[track_idx].id] = (track_state, current_time)
        
        # Try to match unmatched detections with memory
        remaining_dets = []
        for det_idx in unmatched_dets:
            memory_id = self.try_match_from_memory(detections[det_idx])
            
            if memory_id is not None:
                # Reuse ID from memory
                new_track = TrackV4Norm(memory_id, detections[det_idx])
                self.tracks.append(new_track)
                # Remove from memory
                if memory_id in self.track_memory:
                    del self.track_memory[memory_id]
            else:
                remaining_dets.append(det_idx)
        
        # Create new tracks for truly new detections
        for det_idx in remaining_dets:
            new_track = TrackV4Norm(self.next_id, detections[det_idx])
            self.tracks.append(new_track)
            self.next_id += 1
        
        # Remove dead tracks
        self.tracks = [t for t in self.tracks if not t.is_dead()]
        
        # Return current track states
        return [track.get_state() for track in self.tracks]
    
    def get_track_count(self) -> int:
        """Get number of active tracks."""
        return len(self.tracks)
    
    def get_memory_count(self) -> int:
        """Get number of tracks in memory."""
        return len(self.track_memory)
    
    def get_all_track_colors(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Get average colors for all active tracks.
        
        Returns:
            Dictionary mapping track_id to (avg_shirt_bgr, avg_pants_bgr)
        """
        color_dict = {}
        for track in self.tracks:
            avg_shirt, avg_pants = track.get_average_colors()
            color_dict[track.id] = (avg_shirt, avg_pants)
        return color_dict
    
    def reset(self):
        """Reset tracker state."""
        self.tracks = []
        self.track_memory = {}
        self.next_id = 0
