"""
Module 1: Re-ID Tracker
Identity-consistent tracking using geometric features and Hungarian matching.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Optional


class Track:
    """Represents a tracked person with persistent ID."""
    
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
        self.max_age = 30  # Max frames to keep without detection
        
        # Store features
        self.height_m = detection.get('height_m', 0.0)
        self.shoulder_hip = detection.get('shoulder_hip', 0.0)
        self.torso_leg = detection.get('torso_leg', 0.0)
        self.bbox_center = detection.get('bbox_center', (0, 0))
        self.conf = detection.get('conf', 0.0)
        
        # Feature history for smoothing
        self.height_m_history = [self.height_m]
        self.shoulder_hip_history = [self.shoulder_hip]
        self.torso_leg_history = [self.torso_leg]
        
    def update(self, detection: Dict):
        """Update track with new detection."""
        self.age = 0
        self.hits += 1
        self.misses = 0
        
        # Update features with EMA smoothing (alpha=0.3)
        alpha = 0.3
        self.height_m = alpha * detection['height_m'] + (1 - alpha) * self.height_m
        self.shoulder_hip = alpha * detection['shoulder_hip'] + (1 - alpha) * self.shoulder_hip
        self.torso_leg = alpha * detection['torso_leg'] + (1 - alpha) * self.torso_leg
        self.bbox_center = detection['bbox_center']
        self.conf = detection['conf']
        
        # Update history
        self.height_m_history.append(self.height_m)
        self.shoulder_hip_history.append(self.shoulder_hip)
        self.torso_leg_history.append(self.torso_leg)
        
        # Keep only last 10 frames
        if len(self.height_m_history) > 10:
            self.height_m_history.pop(0)
            self.shoulder_hip_history.pop(0)
            self.torso_leg_history.pop(0)
    
    def mark_missed(self):
        """Mark track as missed in current frame."""
        self.age += 1
        self.misses += 1
    
    def is_dead(self) -> bool:
        """Check if track should be deleted."""
        return self.age > self.max_age
    
    def get_state(self) -> Dict:
        """Get current track state as dictionary."""
        return {
            'id': self.id,
            'height_m': self.height_m,
            'shoulder_hip': self.shoulder_hip,
            'torso_leg': self.torso_leg,
            'bbox_center': self.bbox_center,
            'conf': self.conf,
            'hits': self.hits,
            'age': self.age
        }


class ReIDTracker:
    """Re-ID tracker using geometric features and Hungarian matching."""
    
    def __init__(
        self,
        max_cost_threshold: float = 0.35,
        img_width: int = 1920,
        img_height: int = 1080
    ):
        """
        Initialize tracker.
        
        Args:
            max_cost_threshold: Maximum cost for valid matching
            img_width: Image width for normalization
            img_height: Image height for normalization
        """
        self.tracks: List[Track] = []
        self.next_id = 0
        self.max_cost_threshold = max_cost_threshold
        self.img_diagonal = np.sqrt(img_width**2 + img_height**2)
        
    def compute_cost_matrix(
        self,
        detections: List[Dict],
        tracks: List[Track]
    ) -> np.ndarray:
        """
        Compute cost matrix for matching detections to tracks.
        
        Cost formula:
            cost = 0.5 * |h_m_det - h_m_track|
                 + 0.4 * |r_sh_det - r_sh_track|
                 + 0.1 * (distance_of_centers / img_diagonal)
        
        Args:
            detections: List of detection dictionaries
            tracks: List of Track objects
        
        Returns:
            Cost matrix (n_detections x n_tracks)
        """
        if len(detections) == 0 or len(tracks) == 0:
            return np.array([]).reshape(len(detections), len(tracks))
        
        cost_matrix = np.zeros((len(detections), len(tracks)))
        
        for i, det in enumerate(detections):
            for j, track in enumerate(tracks):
                # Height difference (normalized)
                height_diff = abs(det['height_m'] - track.height_m)
                
                # Shoulder/hip ratio difference
                sh_diff = abs(det['shoulder_hip'] - track.shoulder_hip)
                
                # Spatial distance (normalized by image diagonal)
                det_center = np.array(det['bbox_center'])
                track_center = np.array(track.bbox_center)
                spatial_dist = np.linalg.norm(det_center - track_center) / self.img_diagonal
                
                # Weighted cost
                cost = (
                    0.5 * height_diff +
                    0.4 * sh_diff +
                    0.1 * spatial_dist
                )
                
                cost_matrix[i, j] = cost
        
        return cost_matrix
    
    def match_tracks(
        self,
        detections: List[Dict],
        tracks: List[Track]
    ) -> tuple:
        """
        Match detections to existing tracks using Hungarian algorithm.
        
        Args:
            detections: List of detection dictionaries
            tracks: List of Track objects
        
        Returns:
            Tuple of (matched_indices, unmatched_detections, unmatched_tracks)
        """
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
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection dictionaries with features
        
        Returns:
            List of tracked objects with IDs
        """
        # Match detections to existing tracks
        matched, unmatched_dets, unmatched_tracks = self.match_tracks(
            detections, self.tracks
        )
        
        # Update matched tracks
        for det_idx, track_idx in matched:
            self.tracks[track_idx].update(detections[det_idx])
        
        # Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            new_track = Track(self.next_id, detections[det_idx])
            self.tracks.append(new_track)
            self.next_id += 1
        
        # Remove dead tracks
        self.tracks = [t for t in self.tracks if not t.is_dead()]
        
        # Return current track states
        return [track.get_state() for track in self.tracks]
    
    def get_track_count(self) -> int:
        """Get number of active tracks."""
        return len(self.tracks)
    
    def reset(self):
        """Reset tracker state."""
        self.tracks = []
        self.next_id = 0
