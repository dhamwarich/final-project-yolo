"""
Module: Re-ID Tracker V2 (Appearance-Enhanced)
Identity-consistent tracking with geometric + appearance + motion features.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Optional
from tracking.appearance_extractor import compute_color_distance, compute_texture_distance, compute_iou


class TrackV2:
    """Enhanced track with appearance features."""
    
    def __init__(self, track_id: int, detection: Dict):
        """
        Initialize a new track with appearance.
        
        Args:
            track_id: Unique identifier for this track
            detection: Initial detection dictionary with appearance
        """
        self.id = track_id
        self.age = 0
        self.hits = 1
        self.misses = 0
        self.max_age = 30
        
        # Geometric features
        self.height_m = detection.get('height_m', 0.0)
        self.shoulder_hip = detection.get('shoulder_hip', 0.0)
        self.torso_leg = detection.get('torso_leg', 0.0)
        self.bbox_center = detection.get('bbox_center', (0, 0))
        self.bbox = detection.get('bbox', (0, 0, 0, 0))
        self.conf = detection.get('conf', 0.0)
        
        # Appearance features
        self.top_h = detection.get('top_h', 0.0)
        self.bot_h = detection.get('bot_h', 0.0)
        self.texture = detection.get('texture', 0.0)
        
        # Feature history for EMA smoothing
        self.height_m_history = [self.height_m]
        self.torso_leg_history = [self.torso_leg]
        self.top_h_history = [self.top_h]
        self.bot_h_history = [self.bot_h]
        
    def update(self, detection: Dict, alpha: float = 0.3):
        """
        Update track with new detection using EMA.
        
        Args:
            detection: New detection dictionary
            alpha: EMA smoothing factor
        """
        self.age = 0
        self.hits += 1
        self.misses = 0
        
        # Update geometric features with EMA
        self.height_m = alpha * detection['height_m'] + (1 - alpha) * self.height_m
        self.shoulder_hip = alpha * detection['shoulder_hip'] + (1 - alpha) * self.shoulder_hip
        self.torso_leg = alpha * detection['torso_leg'] + (1 - alpha) * self.torso_leg
        
        # Update appearance features with EMA
        self.top_h = alpha * detection.get('top_h', self.top_h) + (1 - alpha) * self.top_h
        self.bot_h = alpha * detection.get('bot_h', self.bot_h) + (1 - alpha) * self.bot_h
        self.texture = alpha * detection.get('texture', self.texture) + (1 - alpha) * self.texture
        
        # Update position (no smoothing for bbox)
        self.bbox_center = detection['bbox_center']
        self.bbox = detection.get('bbox', self.bbox)
        self.conf = detection['conf']
        
        # Update history (keep last 10)
        self.height_m_history.append(self.height_m)
        self.torso_leg_history.append(self.torso_leg)
        self.top_h_history.append(self.top_h)
        self.bot_h_history.append(self.bot_h)
        
        if len(self.height_m_history) > 10:
            self.height_m_history.pop(0)
            self.torso_leg_history.pop(0)
            self.top_h_history.pop(0)
            self.bot_h_history.pop(0)
    
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
            'bbox': self.bbox,
            'conf': self.conf,
            'top_h': self.top_h,
            'bot_h': self.bot_h,
            'texture': self.texture,
            'hits': self.hits,
            'age': self.age
        }


class ReIDTrackerV2:
    """Appearance-enhanced Re-ID tracker with motion assistance."""
    
    def __init__(
        self,
        max_cost_threshold: float = 0.45,
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
        self.tracks: List[TrackV2] = []
        self.next_id = 0
        self.max_cost_threshold = max_cost_threshold
        self.img_diagonal = np.sqrt(img_width**2 + img_height**2)
        
    def compute_cost_matrix(
        self,
        detections: List[Dict],
        tracks: List[TrackV2]
    ) -> np.ndarray:
        """
        Compute enhanced cost matrix with appearance + motion.
        
        Cost formula:
            cost = 0.30 × |H_norm_det - H_norm_track|
                 + 0.20 × |SH_det - SH_track|
                 + 0.10 × |TL_det - TL_track|
                 + 0.25 × color_distance(top/bot)
                 + 0.05 × texture_distance
                 + 0.10 × motion_cost(IoU, center)
        
        Args:
            detections: List of detection dictionaries
            tracks: List of TrackV2 objects
        
        Returns:
            Cost matrix (n_detections x n_tracks)
        """
        if len(detections) == 0 or len(tracks) == 0:
            return np.array([]).reshape(len(detections), len(tracks))
        
        cost_matrix = np.zeros((len(detections), len(tracks)))
        
        for i, det in enumerate(detections):
            for j, track in enumerate(tracks):
                # 1. Height difference (normalized)
                height_diff = abs(det['height_m'] - track.height_m)
                
                # 2. Shoulder/hip ratio difference
                sh_diff = abs(det['shoulder_hip'] - track.shoulder_hip)
                
                # 3. Torso/leg ratio difference
                tl_diff = abs(det['torso_leg'] - track.torso_leg)
                
                # 4. Color distance (using appearance features)
                color_det = {'top_h': det.get('top_h', 0), 'bot_h': det.get('bot_h', 0)}
                color_track = {'top_h': track.top_h, 'bot_h': track.bot_h}
                color_dist = compute_color_distance(color_det, color_track)
                
                # 5. Texture distance
                tex_dist = compute_texture_distance(
                    det.get('texture', 0), 
                    track.texture
                )
                
                # 6. Motion cost (IoU + center proximity)
                det_bbox = det.get('bbox', (0, 0, 0, 0))
                iou = compute_iou(det_bbox, track.bbox)
                
                det_center = np.array(det['bbox_center'])
                track_center = np.array(track.bbox_center)
                center_dist = np.linalg.norm(det_center - track_center) / self.img_diagonal
                
                motion_cost = 0.5 * (1 - iou) + 0.5 * center_dist
                
                # Weighted fusion
                cost = (
                    0.30 * height_diff +
                    0.20 * sh_diff +
                    0.10 * tl_diff +
                    0.25 * color_dist +
                    0.05 * tex_dist +
                    0.10 * motion_cost
                )
                
                cost_matrix[i, j] = cost
        
        return cost_matrix
    
    def match_tracks(
        self,
        detections: List[Dict],
        tracks: List[TrackV2]
    ) -> tuple:
        """
        Match detections to existing tracks using Hungarian algorithm.
        
        Args:
            detections: List of detection dictionaries
            tracks: List of TrackV2 objects
        
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
            detections: List of detection dictionaries with appearance features
        
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
            new_track = TrackV2(self.next_id, detections[det_idx])
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
