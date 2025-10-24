"""
Module: Re-ID Tracker V5 Temporal (Motion-Dominant Matching)
Tracking focused on geometry + motion, with color as weak contextual cue.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Optional, Tuple
from tracking.temporal_reid import FeatureMemory, compute_temporal_similarity
from utils.bbox_smoothing import smooth_bbox_ema
import time


class TrackV5Temporal:
    """Track with temporal feature memory and motion-dominant matching."""
    
    def __init__(self, track_id: int, detection: Dict):
        self.id = track_id
        self.age = 0
        self.hits = 1
        self.misses = 0
        self.max_age = 10  # Reduced from 30 for faster occlusion timeout
        
        # Geometric features (current)
        self.shoulder_hip = detection.get('shoulder_hip', 0.0)
        self.aspect = detection.get('aspect', 1.0)
        self.bbox_center = detection.get('bbox_center', (0, 0))
        self.bbox = detection.get('bbox', (0, 0, 0, 0))
        self.conf = detection.get('conf', 0.0)
        
        # Bbox smoothing
        self.prev_bbox = self.bbox
        self.bbox_smooth_beta = 0.6
        
        # Feature memory (temporal)
        self.memory = FeatureMemory(
            geom_alpha=0.3,
            motion_alpha=0.3,
            color_alpha=0.1
        )
        self.memory.update(detection)
        
        # Color confidence tracking
        self.shirt_confidence = detection.get('shirt_confidence', 0.0)
        self.pants_confidence = detection.get('pants_confidence', 0.0)
        
        # Track statistics
        self.geom_std_samples = []
        self.motion_std_samples = []
        
    def update(self, detection: Dict):
        """Update track with new detection."""
        self.age = 0
        self.hits += 1
        self.misses = 0
        
        # Update current features
        self.shoulder_hip = detection['shoulder_hip']
        self.aspect = detection.get('aspect', self.aspect)
        self.bbox_center = detection['bbox_center']
        self.conf = detection['conf']
        
        # Update confidence
        self.shirt_confidence = detection.get('shirt_confidence', self.shirt_confidence)
        self.pants_confidence = detection.get('pants_confidence', self.pants_confidence)
        
        # Update feature memory
        self.memory.update(detection)
        
        # Track geometry variance
        mem_state = self.memory.get_state()
        geom_diff = abs(self.shoulder_hip - mem_state['shoulder_hip'])
        self.geom_std_samples.append(geom_diff)
        if len(self.geom_std_samples) > 50:
            self.geom_std_samples.pop(0)
        
        # Track motion variance
        velocity = np.array(mem_state['velocity'])
        motion_mag = np.linalg.norm(velocity)
        self.motion_std_samples.append(motion_mag)
        if len(self.motion_std_samples) > 50:
            self.motion_std_samples.pop(0)
        
        # Update bbox with smoothing
        new_bbox = detection.get('bbox', self.bbox)
        self.bbox = smooth_bbox_ema(new_bbox, self.prev_bbox, self.bbox_smooth_beta)
        self.prev_bbox = self.bbox
    
    def mark_missed(self):
        """Mark track as missed (occlusion handling)."""
        self.age += 1
        self.misses += 1
        self.memory.age()
    
    def is_dead(self) -> bool:
        """Check if track should be deleted."""
        return self.age > self.max_age
    
    def get_avg_geom_std(self) -> float:
        """Get average geometry variance."""
        if len(self.geom_std_samples) == 0:
            return 0.0
        return float(np.mean(self.geom_std_samples))
    
    def get_avg_motion_std(self) -> float:
        """Get average motion magnitude."""
        if len(self.motion_std_samples) == 0:
            return 0.0
        return float(np.std(self.motion_std_samples))
    
    def get_state(self) -> Dict:
        """Get current track state."""
        mem_state = self.memory.get_state()
        stable_shirt, stable_pants = mem_state['stable_shirt_HSV'], mem_state['stable_pants_HSV']
        
        return {
            'id': self.id,
            'shoulder_hip': self.shoulder_hip,
            'aspect': self.aspect,
            'bbox_center': self.bbox_center,
            'bbox': self.bbox,
            'conf': self.conf,
            'velocity': mem_state['velocity'],
            'stable_shirt_HSV': stable_shirt,
            'stable_pants_HSV': stable_pants,
            'shirt_confidence': self.shirt_confidence,
            'pants_confidence': self.pants_confidence,
            'avg_geom_std': self.get_avg_geom_std(),
            'avg_motion_std': self.get_avg_motion_std(),
            'hits': self.hits,
            'misses': self.misses,
            'age': self.age,
            'memory_age': mem_state['age'],
            'total_updates': mem_state['updates']
        }


class ReIDTrackerV5Temporal:
    """Re-ID tracker with temporal memory and motion-dominant matching."""
    
    def __init__(
        self,
        max_similarity_threshold: float = 0.40,
        memory_similarity_threshold: float = 0.30,
        memory_duration: float = 5.0,
        color_conf_threshold: float = 0.4
    ):
        self.tracks: List[TrackV5Temporal] = []
        self.next_id = 0
        self.max_similarity_threshold = max_similarity_threshold
        self.memory_similarity_threshold = memory_similarity_threshold
        self.memory_duration = memory_duration
        self.color_conf_threshold = color_conf_threshold
        self.track_memory: Dict[int, tuple] = {}
        
        # Statistics
        self.id_switches = 0
        
    def compute_similarity_matrix(
        self,
        detections: List[Dict],
        tracks: List[TrackV5Temporal]
    ) -> np.ndarray:
        """
        Compute similarity matrix using temporal features.
        Higher score = better match (opposite of cost).
        """
        if len(detections) == 0 or len(tracks) == 0:
            return np.array([]).reshape(len(detections), len(tracks))
        
        similarity_matrix = np.zeros((len(detections), len(tracks)))
        
        for i, det in enumerate(detections):
            for j, track in enumerate(tracks):
                # Compute temporal similarity
                similarity = compute_temporal_similarity(
                    det,
                    track.memory,
                    self.color_conf_threshold
                )
                
                similarity_matrix[i, j] = similarity
        
        return similarity_matrix
    
    def match_tracks(
        self,
        detections: List[Dict],
        tracks: List[TrackV5Temporal]
    ) -> tuple:
        """Match detections to tracks using similarity (higher = better)."""
        if len(detections) == 0:
            return [], [], list(range(len(tracks)))
        
        if len(tracks) == 0:
            return [], list(range(len(detections))), []
        
        similarity_matrix = self.compute_similarity_matrix(detections, tracks)
        
        # Convert similarity to cost (Hungarian algorithm minimizes)
        cost_matrix = 1.0 - similarity_matrix
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(range(len(tracks)))
        
        for i, j in zip(row_ind, col_ind):
            # Use similarity threshold (higher similarity = lower cost)
            if similarity_matrix[i, j] >= self.max_similarity_threshold:
                matched.append((i, j))
                if i in unmatched_dets:
                    unmatched_dets.remove(i)
                if j in unmatched_tracks:
                    unmatched_tracks.remove(j)
        
        return matched, unmatched_dets, unmatched_tracks
    
    def try_match_from_memory(self, detection: Dict) -> Optional[int]:
        """Try to match detection to memory (for re-identification after occlusion)."""
        if len(self.track_memory) == 0:
            return None
        
        current_time = time.time()
        best_id = None
        best_similarity = self.memory_similarity_threshold
        
        for track_id, (memory_state, timestamp) in list(self.track_memory.items()):
            if current_time - timestamp > self.memory_duration:
                del self.track_memory[track_id]
                continue
            
            # Recreate memory from state for similarity computation
            temp_memory = FeatureMemory()
            temp_memory.shoulder_hip_mem = memory_state['shoulder_hip']
            temp_memory.aspect_mem = memory_state['aspect']
            temp_memory.velocity_mem = np.array(memory_state['velocity'])
            temp_memory.shirt_HSV_mem = np.array(memory_state['stable_shirt_HSV'])
            temp_memory.pants_HSV_mem = np.array(memory_state['stable_pants_HSV'])
            
            similarity = compute_temporal_similarity(
                detection,
                temp_memory,
                self.color_conf_threshold
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_id = track_id
        
        return best_id
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """Update tracks with new detections."""
        matched, unmatched_dets, unmatched_tracks = self.match_tracks(
            detections, self.tracks
        )
        
        # Update matched tracks
        for det_idx, track_idx in matched:
            self.tracks[track_idx].update(detections[det_idx])
        
        # Handle unmatched tracks (occlusion or exit)
        current_time = time.time()
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
            
            # If track is dead, save to memory
            if self.tracks[track_idx].is_dead():
                track_state = self.tracks[track_idx].get_state()
                self.track_memory[self.tracks[track_idx].id] = (track_state, current_time)
        
        # Try to match unmatched detections from memory first
        remaining_dets = []
        for det_idx in unmatched_dets:
            memory_id = self.try_match_from_memory(detections[det_idx])
            
            if memory_id is not None:
                # Re-identify from memory
                new_track = TrackV5Temporal(memory_id, detections[det_idx])
                self.tracks.append(new_track)
                if memory_id in self.track_memory:
                    del self.track_memory[memory_id]
            else:
                remaining_dets.append(det_idx)
        
        # Create new tracks for remaining detections
        for det_idx in remaining_dets:
            new_track = TrackV5Temporal(self.next_id, detections[det_idx])
            self.tracks.append(new_track)
            self.next_id += 1
        
        # Remove dead tracks
        self.tracks = [t for t in self.tracks if not t.is_dead()]
        
        return [track.get_state() for track in self.tracks]
    
    def get_track_count(self) -> int:
        """Get number of active tracks."""
        return len(self.tracks)
    
    def get_memory_count(self) -> int:
        """Get number of tracks in memory."""
        return len(self.track_memory)
    
    def get_all_track_stats(self) -> Dict[int, Dict]:
        """Get statistics for all active tracks."""
        stats_dict = {}
        for track in self.tracks:
            state = track.get_state()
            stats_dict[track.id] = {
                'hits': state['hits'],
                'misses': state['misses'],
                'avg_geom_std': state['avg_geom_std'],
                'avg_motion_std': state['avg_motion_std'],
                'shirt_confidence': state['shirt_confidence'],
                'pants_confidence': state['pants_confidence'],
                'stable_shirt_HSV': state['stable_shirt_HSV'],
                'stable_pants_HSV': state['stable_pants_HSV']
            }
        return stats_dict
    
    def reset(self):
        """Reset tracker state."""
        self.tracks = []
        self.track_memory = {}
        self.next_id = 0
        self.id_switches = 0
