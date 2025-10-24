"""
Module: Re-ID Tracker V4 Cropped (With HSV Color Storage)
Tracking with HSV values stored for color naming and logging.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Optional, Tuple
from tracking.appearance_extractor_v3_crop import (
    compute_bhattacharyya_distance,
    compute_iou
)
import time


class TrackV4Crop:
    """Track with HSV color storage."""
    
    def __init__(self, track_id: int, detection: Dict):
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
        
        # Histograms for matching
        self.shirt_hist = detection.get('shirt_hist', np.zeros(16))
        self.pants_hist = detection.get('pants_hist', np.zeros(16))
        
        # HSV values for color naming
        self.shirt_HSV = detection.get('shirt_HSV', (0.0, 0.0, 0.0))
        self.pants_HSV = detection.get('pants_HSV', (0.0, 0.0, 0.0))
        
        # HSV accumulation
        self.shirt_HSV_samples = [self.shirt_HSV]
        self.pants_HSV_samples = [self.pants_HSV]
        
        # BGR for visualization
        self.shirt_color_bgr = detection.get('shirt_color_bgr', np.array([128, 128, 128]))
        self.pants_color_bgr = detection.get('pants_color_bgr', np.array([128, 128, 128]))
        
    def update(self, detection: Dict, alpha: float = 0.2):
        self.age = 0
        self.hits += 1
        self.misses = 0
        
        # Update geometric features
        self.shoulder_hip = alpha * detection['shoulder_hip'] + (1 - alpha) * self.shoulder_hip
        self.aspect = alpha * detection.get('aspect', self.aspect) + (1 - alpha) * self.aspect
        
        # Update histograms
        new_shirt_hist = detection.get('shirt_hist', self.shirt_hist)
        new_pants_hist = detection.get('pants_hist', self.pants_hist)
        
        self.shirt_hist = alpha * new_shirt_hist + (1 - alpha) * self.shirt_hist
        self.pants_hist = alpha * new_pants_hist + (1 - alpha) * self.pants_hist
        
        self.shirt_hist = self.shirt_hist / (self.shirt_hist.sum() + 1e-6)
        self.pants_hist = self.pants_hist / (self.pants_hist.sum() + 1e-6)
        
        # Accumulate HSV samples
        self.shirt_HSV = detection.get('shirt_HSV', self.shirt_HSV)
        self.pants_HSV = detection.get('pants_HSV', self.pants_HSV)
        
        self.shirt_HSV_samples.append(self.shirt_HSV)
        self.pants_HSV_samples.append(self.pants_HSV)
        
        # Keep last 50 samples
        if len(self.shirt_HSV_samples) > 50:
            self.shirt_HSV_samples.pop(0)
            self.pants_HSV_samples.pop(0)
        
        # Update BGR colors
        self.shirt_color_bgr = detection.get('shirt_color_bgr', self.shirt_color_bgr)
        self.pants_color_bgr = detection.get('pants_color_bgr', self.pants_color_bgr)
        
        # Update position
        self.bbox_center = detection['bbox_center']
        self.bbox = detection.get('bbox', self.bbox)
        self.conf = detection['conf']
    
    def get_average_HSV(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Get average HSV across all samples."""
        if len(self.shirt_HSV_samples) == 0:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        
        # Average HSV (handle hue wrapping for red)
        shirt_H = np.mean([s[0] for s in self.shirt_HSV_samples])
        shirt_S = np.mean([s[1] for s in self.shirt_HSV_samples])
        shirt_V = np.mean([s[2] for s in self.shirt_HSV_samples])
        
        pants_H = np.mean([s[0] for s in self.pants_HSV_samples])
        pants_S = np.mean([s[1] for s in self.pants_HSV_samples])
        pants_V = np.mean([s[2] for s in self.pants_HSV_samples])
        
        return (shirt_H, shirt_S, shirt_V), (pants_H, pants_S, pants_V)
    
    def mark_missed(self):
        self.age += 1
        self.misses += 1
    
    def is_dead(self) -> bool:
        return self.age > self.max_age
    
    def get_state(self) -> Dict:
        avg_shirt_HSV, avg_pants_HSV = self.get_average_HSV()
        
        return {
            'id': self.id,
            'shoulder_hip': self.shoulder_hip,
            'aspect': self.aspect,
            'bbox_center': self.bbox_center,
            'bbox': self.bbox,
            'conf': self.conf,
            'shirt_hist': self.shirt_hist.copy(),
            'pants_hist': self.pants_hist.copy(),
            'shirt_HSV': self.shirt_HSV,
            'pants_HSV': self.pants_HSV,
            'avg_shirt_HSV': avg_shirt_HSV,
            'avg_pants_HSV': avg_pants_HSV,
            'shirt_color_bgr': self.shirt_color_bgr.copy(),
            'pants_color_bgr': self.pants_color_bgr.copy(),
            'hits': self.hits,
            'age': self.age
        }


class ReIDTrackerV4Crop:
    """Re-ID tracker with HSV color storage."""
    
    def __init__(
        self,
        max_cost_threshold: float = 0.45,
        memory_cost_threshold: float = 0.30,
        memory_duration: float = 5.0,
        img_width: int = 1920,
        img_height: int = 1080
    ):
        self.tracks: List[TrackV4Crop] = []
        self.next_id = 0
        self.max_cost_threshold = max_cost_threshold
        self.memory_cost_threshold = memory_cost_threshold
        self.memory_duration = memory_duration
        self.img_diagonal = np.sqrt(img_width**2 + img_height**2)
        self.track_memory: Dict[int, tuple] = {}
        
    def compute_cost_matrix(
        self,
        detections: List[Dict],
        tracks: List[TrackV4Crop]
    ) -> np.ndarray:
        if len(detections) == 0 or len(tracks) == 0:
            return np.array([]).reshape(len(detections), len(tracks))
        
        cost_matrix = np.zeros((len(detections), len(tracks)))
        
        for i, det in enumerate(detections):
            for j, track in enumerate(tracks):
                sh_diff = abs(det['shoulder_hip'] - track.shoulder_hip)
                aspect_diff = abs(det.get('aspect', 1.0) - track.aspect)
                
                shirt_dist = compute_bhattacharyya_distance(
                    det.get('shirt_hist', np.zeros(16)),
                    track.shirt_hist
                )
                
                pants_dist = compute_bhattacharyya_distance(
                    det.get('pants_hist', np.zeros(16)),
                    track.pants_hist
                )
                
                det_bbox = det.get('bbox', (0, 0, 0, 0))
                iou = compute_iou(det_bbox, track.bbox)
                
                det_center = np.array(det['bbox_center'])
                track_center = np.array(track.bbox_center)
                center_dist = np.linalg.norm(det_center - track_center) / self.img_diagonal
                
                motion_cost = 0.5 * (1 - iou) + 0.5 * center_dist
                
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
        tracks: List[TrackV4Crop]
    ) -> tuple:
        if len(detections) == 0:
            return [], [], list(range(len(tracks)))
        
        if len(tracks) == 0:
            return [], list(range(len(detections))), []
        
        cost_matrix = self.compute_cost_matrix(detections, tracks)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
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
        if len(self.track_memory) == 0:
            return None
        
        current_time = time.time()
        best_id = None
        best_cost = self.memory_cost_threshold
        
        for track_id, (track_state, timestamp) in list(self.track_memory.items()):
            if current_time - timestamp > self.memory_duration:
                del self.track_memory[track_id]
                continue
            
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
        matched, unmatched_dets, unmatched_tracks = self.match_tracks(
            detections, self.tracks
        )
        
        for det_idx, track_idx in matched:
            self.tracks[track_idx].update(detections[det_idx])
        
        current_time = time.time()
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
            
            if self.tracks[track_idx].is_dead():
                track_state = self.tracks[track_idx].get_state()
                self.track_memory[self.tracks[track_idx].id] = (track_state, current_time)
        
        remaining_dets = []
        for det_idx in unmatched_dets:
            memory_id = self.try_match_from_memory(detections[det_idx])
            
            if memory_id is not None:
                new_track = TrackV4Crop(memory_id, detections[det_idx])
                self.tracks.append(new_track)
                if memory_id in self.track_memory:
                    del self.track_memory[memory_id]
            else:
                remaining_dets.append(det_idx)
        
        for det_idx in remaining_dets:
            new_track = TrackV4Crop(self.next_id, detections[det_idx])
            self.tracks.append(new_track)
            self.next_id += 1
        
        self.tracks = [t for t in self.tracks if not t.is_dead()]
        
        return [track.get_state() for track in self.tracks]
    
    def get_track_count(self) -> int:
        return len(self.tracks)
    
    def get_memory_count(self) -> int:
        return len(self.track_memory)
    
    def get_all_track_HSV(self) -> Dict[int, Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """Get average HSV for all active tracks."""
        hsv_dict = {}
        for track in self.tracks:
            avg_shirt_HSV, avg_pants_HSV = track.get_average_HSV()
            hsv_dict[track.id] = (avg_shirt_HSV, avg_pants_HSV)
        return hsv_dict
    
    def reset(self):
        self.tracks = []
        self.track_memory = {}
        self.next_id = 0
