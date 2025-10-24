"""
Module: Re-ID Tracker V3 (Memory-based, No Height)
Identity-consistent tracking with short-term memory and histogram-based appearance.
No pseudo-height estimation - uses geometry + appearance + motion only.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Optional
from tracking.appearance_extractor_v2 import (
    compute_histogram_color_distance,
    compute_texture_distance,
    compute_iou
)
import time


class TrackV3:
    """Track with histogram-based appearance and no height estimation."""
    
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
        
        # Geometric features (no height_m)
        self.shoulder_hip = detection.get('shoulder_hip', 0.0)
        self.torso_leg = detection.get('torso_leg', 0.0)
        self.aspect = detection.get('aspect', 1.0)
        self.bbox_center = detection.get('bbox_center', (0, 0))
        self.bbox = detection.get('bbox', (0, 0, 0, 0))
        self.conf = detection.get('conf', 0.0)
        
        # Histogram-based appearance
        self.top_hist = detection.get('top_hist', np.zeros(16))
        self.bot_hist = detection.get('bot_hist', np.zeros(16))
        self.texture = detection.get('texture', 0.0)
        
        # Feature history for EMA smoothing
        self.torso_leg_history = [self.torso_leg]
        self.top_hist_history = [self.top_hist.copy()]
        self.bot_hist_history = [self.bot_hist.copy()]
        
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
        self.shoulder_hip = alpha * detection['shoulder_hip'] + (1 - alpha) * self.shoulder_hip
        self.torso_leg = alpha * detection['torso_leg'] + (1 - alpha) * self.torso_leg
        self.aspect = alpha * detection.get('aspect', self.aspect) + (1 - alpha) * self.aspect
        
        # Update histogram-based appearance with EMA
        new_top_hist = detection.get('top_hist', self.top_hist)
        new_bot_hist = detection.get('bot_hist', self.bot_hist)
        
        self.top_hist = alpha * new_top_hist + (1 - alpha) * self.top_hist
        self.bot_hist = alpha * new_bot_hist + (1 - alpha) * self.bot_hist
        
        # Normalize histograms after EMA
        self.top_hist = self.top_hist / (self.top_hist.sum() + 1e-6)
        self.bot_hist = self.bot_hist / (self.bot_hist.sum() + 1e-6)
        
        self.texture = alpha * detection.get('texture', self.texture) + (1 - alpha) * self.texture
        
        # Update position (no smoothing for bbox)
        self.bbox_center = detection['bbox_center']
        self.bbox = detection.get('bbox', self.bbox)
        self.conf = detection['conf']
        
        # Update history (keep last 10)
        self.torso_leg_history.append(self.torso_leg)
        self.top_hist_history.append(self.top_hist.copy())
        self.bot_hist_history.append(self.bot_hist.copy())
        
        if len(self.torso_leg_history) > 10:
            self.torso_leg_history.pop(0)
            self.top_hist_history.pop(0)
            self.bot_hist_history.pop(0)
    
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
            'shoulder_hip': self.shoulder_hip,
            'torso_leg': self.torso_leg,
            'aspect': self.aspect,
            'bbox_center': self.bbox_center,
            'bbox': self.bbox,
            'conf': self.conf,
            'top_hist': self.top_hist.copy(),
            'bot_hist': self.bot_hist.copy(),
            'texture': self.texture,
            'hits': self.hits,
            'age': self.age
        }


class ReIDTrackerV3:
    """Memory-based Re-ID tracker without height estimation."""
    
    def __init__(
        self,
        max_cost_threshold: float = 0.45,
        memory_cost_threshold: float = 0.25,
        memory_duration: float = 5.0,
        img_width: int = 1920,
        img_height: int = 1080
    ):
        """
        Initialize tracker with memory.
        
        Args:
            max_cost_threshold: Maximum cost for valid matching
            memory_cost_threshold: Lower threshold for memory matching
            memory_duration: How long to keep lost tracks (seconds)
            img_width: Image width for normalization
            img_height: Image height for normalization
        """
        self.tracks: List[TrackV3] = []
        self.next_id = 0
        self.max_cost_threshold = max_cost_threshold
        self.memory_cost_threshold = memory_cost_threshold
        self.memory_duration = memory_duration
        self.img_diagonal = np.sqrt(img_width**2 + img_height**2)
        
        # Short-term memory: stores recently lost tracks
        # Format: {track_id: (track_state, timestamp)}
        self.track_memory: Dict[int, tuple] = {}
        
    def compute_cost_matrix(
        self,
        detections: List[Dict],
        tracks: List[TrackV3]
    ) -> np.ndarray:
        """
        Compute cost matrix without height component.
        
        Cost formula:
            cost = 0.25 × |SH_det - SH_track|
                 + 0.15 × |TL_det - TL_track|
                 + 0.10 × |aspect_det - aspect_track|
                 + 0.35 × histogram_color_distance
                 + 0.05 × texture_distance
                 + 0.10 × motion_cost(IoU, center)
        
        Args:
            detections: List of detection dictionaries
            tracks: List of TrackV3 objects
        
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
                
                # 2. Torso/leg ratio difference
                tl_diff = abs(det['torso_leg'] - track.torso_leg)
                
                # 3. Aspect ratio difference
                aspect_diff = abs(det.get('aspect', 1.0) - track.aspect)
                
                # 4. Histogram-based color distance
                color_det = {
                    'top_hist': det.get('top_hist', np.zeros(16)),
                    'bot_hist': det.get('bot_hist', np.zeros(16))
                }
                color_track = {
                    'top_hist': track.top_hist,
                    'bot_hist': track.bot_hist
                }
                color_dist = compute_histogram_color_distance(color_det, color_track)
                
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
                
                # Weighted fusion (NO HEIGHT)
                cost = (
                    0.25 * sh_diff +
                    0.15 * tl_diff +
                    0.10 * aspect_diff +
                    0.35 * color_dist +
                    0.05 * tex_dist +
                    0.10 * motion_cost
                )
                
                cost_matrix[i, j] = cost
        
        return cost_matrix
    
    def match_tracks(
        self,
        detections: List[Dict],
        tracks: List[TrackV3]
    ) -> tuple:
        """
        Match detections to existing tracks using Hungarian algorithm.
        
        Args:
            detections: List of detection dictionaries
            tracks: List of TrackV3 objects
        
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
    
    def try_match_from_memory(self, detection: Dict) -> Optional[int]:
        """
        Try to match detection with recently lost tracks in memory.
        
        Args:
            detection: Detection dictionary
        
        Returns:
            Track ID if match found, None otherwise
        """
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
            
            # Compute cost to memory track
            sh_diff = abs(detection['shoulder_hip'] - track_state['shoulder_hip'])
            tl_diff = abs(detection['torso_leg'] - track_state['torso_leg'])
            aspect_diff = abs(detection.get('aspect', 1.0) - track_state['aspect'])
            
            color_det = {
                'top_hist': detection.get('top_hist', np.zeros(16)),
                'bot_hist': detection.get('bot_hist', np.zeros(16))
            }
            color_mem = {
                'top_hist': track_state['top_hist'],
                'bot_hist': track_state['bot_hist']
            }
            color_dist = compute_histogram_color_distance(color_det, color_mem)
            
            tex_dist = compute_texture_distance(
                detection.get('texture', 0),
                track_state['texture']
            )
            
            # Memory matching uses same weights but stricter threshold
            cost = (
                0.25 * sh_diff +
                0.15 * tl_diff +
                0.10 * aspect_diff +
                0.35 * color_dist +
                0.05 * tex_dist
                # Note: No motion component for memory (person may have moved)
            )
            
            if cost < best_cost:
                best_cost = cost
                best_id = track_id
        
        return best_id
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracker with new detections and memory management.
        
        Args:
            detections: List of detection dictionaries
        
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
                new_track = TrackV3(memory_id, detections[det_idx])
                self.tracks.append(new_track)
                # Remove from memory
                if memory_id in self.track_memory:
                    del self.track_memory[memory_id]
            else:
                remaining_dets.append(det_idx)
        
        # Create new tracks for truly new detections
        for det_idx in remaining_dets:
            new_track = TrackV3(self.next_id, detections[det_idx])
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
    
    def reset(self):
        """Reset tracker state."""
        self.tracks = []
        self.track_memory = {}
        self.next_id = 0
