"""
Module: Re-ID Tracker V4 Robust (Adaptive Smoothing + Confidence Weighting)
Tracking with adaptive temporal smoothing and confidence-based matching.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Optional, Tuple
from collections import deque
from tracking.appearance_extractor_v3_robust import (
    compute_bhattacharyya_distance,
    compute_iou
)
from utils.bbox_smoothing import smooth_bbox_ema
import time


class TrackV4Robust:
    """Track with adaptive temporal smoothing and confidence weighting."""
    
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
        
        # Bbox smoothing
        self.prev_bbox = self.bbox
        self.bbox_smooth_beta = 0.6
        
        # Histograms for matching
        self.shirt_hist = detection.get('shirt_hist', np.zeros(16))
        self.pants_hist = detection.get('pants_hist', np.zeros(16))
        
        # HSV values (current frame)
        self.shirt_HSV = detection.get('shirt_HSV', (0.0, 0.0, 0.0))
        self.pants_HSV = detection.get('pants_HSV', (0.0, 0.0, 0.0))
        
        # Confidence values
        self.shirt_confidence = detection.get('shirt_confidence', 0.0)
        self.pants_confidence = detection.get('pants_confidence', 0.0)
        
        # Temporal color smoothing with adaptive buffer
        self.shirt_H_buffer = deque([self.shirt_HSV[0]], maxlen=5)
        self.shirt_S_buffer = deque([self.shirt_HSV[1]], maxlen=5)
        self.shirt_V_buffer = deque([self.shirt_HSV[2]], maxlen=5)
        
        self.pants_H_buffer = deque([self.pants_HSV[0]], maxlen=5)
        self.pants_S_buffer = deque([self.pants_HSV[1]], maxlen=5)
        self.pants_V_buffer = deque([self.pants_HSV[2]], maxlen=5)
        
        # HSV accumulation for long-term average and variance
        self.shirt_HSV_samples = [self.shirt_HSV]
        self.pants_HSV_samples = [self.pants_HSV]
        
        # Confidence accumulation
        self.shirt_conf_samples = [self.shirt_confidence]
        self.pants_conf_samples = [self.pants_confidence]
        
        # Adaptive smoothing parameter
        self.adaptive_alpha = 0.3  # Default, can increase if variance is high
        
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
        
        # Update HSV values
        new_shirt_HSV = detection.get('shirt_HSV', self.shirt_HSV)
        new_pants_HSV = detection.get('pants_HSV', self.pants_HSV)
        
        # Update confidence
        self.shirt_confidence = detection.get('shirt_confidence', self.shirt_confidence)
        self.pants_confidence = detection.get('pants_confidence', self.pants_confidence)
        
        # Add to temporal buffers
        self.shirt_H_buffer.append(new_shirt_HSV[0])
        self.shirt_S_buffer.append(new_shirt_HSV[1])
        self.shirt_V_buffer.append(new_shirt_HSV[2])
        
        self.pants_H_buffer.append(new_pants_HSV[0])
        self.pants_S_buffer.append(new_pants_HSV[1])
        self.pants_V_buffer.append(new_pants_HSV[2])
        
        # Compute current hue variance
        if len(self.shirt_HSV_samples) >= 3:
            recent_H_vals = [s[0] for s in self.shirt_HSV_samples[-10:]]
            current_variance = np.std(recent_H_vals)
            
            # Adaptive smoothing: increase alpha if variance is high
            if current_variance > 10.0:
                self.adaptive_alpha = 0.5  # More smoothing
            else:
                self.adaptive_alpha = 0.3  # Normal smoothing
        
        # Compute smoothed HSV (average of buffer)
        self.shirt_HSV = (
            float(np.mean(self.shirt_H_buffer)),
            float(np.mean(self.shirt_S_buffer)),
            float(np.mean(self.shirt_V_buffer))
        )
        
        self.pants_HSV = (
            float(np.mean(self.pants_H_buffer)),
            float(np.mean(self.pants_S_buffer)),
            float(np.mean(self.pants_V_buffer))
        )
        
        # Accumulate for long-term average
        self.shirt_HSV_samples.append(new_shirt_HSV)
        self.pants_HSV_samples.append(new_pants_HSV)
        
        self.shirt_conf_samples.append(self.shirt_confidence)
        self.pants_conf_samples.append(self.pants_confidence)
        
        # Keep last 50 samples
        if len(self.shirt_HSV_samples) > 50:
            self.shirt_HSV_samples.pop(0)
            self.pants_HSV_samples.pop(0)
            self.shirt_conf_samples.pop(0)
            self.pants_conf_samples.pop(0)
        
        # Update BGR colors
        self.shirt_color_bgr = detection.get('shirt_color_bgr', self.shirt_color_bgr)
        self.pants_color_bgr = detection.get('pants_color_bgr', self.pants_color_bgr)
        
        # Update bbox with smoothing
        new_bbox = detection.get('bbox', self.bbox)
        self.bbox = smooth_bbox_ema(new_bbox, self.prev_bbox, self.bbox_smooth_beta)
        self.prev_bbox = self.bbox
        
        # Update position
        self.bbox_center = detection['bbox_center']
        self.conf = detection['conf']
    
    def get_smoothed_HSV(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Get temporally smoothed HSV (from 5-frame buffer)."""
        return self.shirt_HSV, self.pants_HSV
    
    def get_average_HSV(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Get long-term average HSV across all samples."""
        if len(self.shirt_HSV_samples) == 0:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        
        shirt_H = np.mean([s[0] for s in self.shirt_HSV_samples])
        shirt_S = np.mean([s[1] for s in self.shirt_HSV_samples])
        shirt_V = np.mean([s[2] for s in self.shirt_HSV_samples])
        
        pants_H = np.mean([s[0] for s in self.pants_HSV_samples])
        pants_S = np.mean([s[1] for s in self.pants_HSV_samples])
        pants_V = np.mean([s[2] for s in self.pants_HSV_samples])
        
        return (shirt_H, shirt_S, shirt_V), (pants_H, pants_S, pants_V)
    
    def get_average_confidence(self) -> Tuple[float, float]:
        """Get average confidence for shirt and pants."""
        if len(self.shirt_conf_samples) == 0:
            return 0.0, 0.0
        
        shirt_conf = np.mean(self.shirt_conf_samples)
        pants_conf = np.mean(self.pants_conf_samples)
        
        return float(shirt_conf), float(pants_conf)
    
    def get_HSV_variance(self) -> Tuple[float, float]:
        """Get hue variance (std) for shirt and pants."""
        if len(self.shirt_HSV_samples) < 2:
            return 0.0, 0.0
        
        shirt_H_vals = [s[0] for s in self.shirt_HSV_samples]
        pants_H_vals = [s[0] for s in self.pants_HSV_samples]
        
        shirt_H_std = float(np.std(shirt_H_vals))
        pants_H_std = float(np.std(pants_H_vals))
        
        return shirt_H_std, pants_H_std
    
    def mark_missed(self):
        self.age += 1
        self.misses += 1
    
    def is_dead(self) -> bool:
        return self.age > self.max_age
    
    def get_state(self) -> Dict:
        smooth_shirt_HSV, smooth_pants_HSV = self.get_smoothed_HSV()
        avg_shirt_HSV, avg_pants_HSV = self.get_average_HSV()
        shirt_H_std, pants_H_std = self.get_HSV_variance()
        avg_shirt_conf, avg_pants_conf = self.get_average_confidence()
        
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
            'smooth_shirt_HSV': smooth_shirt_HSV,
            'smooth_pants_HSV': smooth_pants_HSV,
            'avg_shirt_HSV': avg_shirt_HSV,
            'avg_pants_HSV': avg_pants_HSV,
            'shirt_H_std': shirt_H_std,
            'pants_H_std': pants_H_std,
            'shirt_confidence': self.shirt_confidence,
            'pants_confidence': self.pants_confidence,
            'avg_shirt_confidence': avg_shirt_conf,
            'avg_pants_confidence': avg_pants_conf,
            'adaptive_alpha': self.adaptive_alpha,
            'shirt_color_bgr': self.shirt_color_bgr.copy(),
            'pants_color_bgr': self.pants_color_bgr.copy(),
            'hits': self.hits,
            'age': self.age
        }


class ReIDTrackerV4Robust:
    """Re-ID tracker with adaptive smoothing and confidence weighting."""
    
    def __init__(
        self,
        max_cost_threshold: float = 0.45,
        memory_cost_threshold: float = 0.30,
        memory_duration: float = 5.0,
        img_width: int = 1920,
        img_height: int = 1080
    ):
        self.tracks: List[TrackV4Robust] = []
        self.next_id = 0
        self.max_cost_threshold = max_cost_threshold
        self.memory_cost_threshold = memory_cost_threshold
        self.memory_duration = memory_duration
        self.img_diagonal = np.sqrt(img_width**2 + img_height**2)
        self.track_memory: Dict[int, tuple] = {}
        
    def compute_cost_matrix(
        self,
        detections: List[Dict],
        tracks: List[TrackV4Robust]
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
                
                # Apply confidence weighting to color distances
                det_shirt_conf = det.get('shirt_confidence', 0.5)
                det_pants_conf = det.get('pants_confidence', 0.5)
                track_shirt_conf = track.shirt_confidence
                track_pants_conf = track.pants_confidence
                
                # Average confidence for weighting
                avg_shirt_conf = (det_shirt_conf + track_shirt_conf) / 2.0
                avg_pants_conf = (det_pants_conf + track_pants_conf) / 2.0
                
                # Confidence weighting: reduce impact of low-confidence colors
                shirt_weight = 1.0 - 0.5 * (1.0 - avg_shirt_conf)
                pants_weight = 1.0 - 0.5 * (1.0 - avg_pants_conf)
                
                shirt_dist_weighted = shirt_dist * shirt_weight
                pants_dist_weighted = pants_dist * pants_weight
                
                det_bbox = det.get('bbox', (0, 0, 0, 0))
                iou = compute_iou(det_bbox, track.bbox)
                
                det_center = np.array(det['bbox_center'])
                track_center = np.array(track.bbox_center)
                center_dist = np.linalg.norm(det_center - track_center) / self.img_diagonal
                
                motion_cost = 0.5 * (1 - iou) + 0.5 * center_dist
                
                # Reduced color weight (0.15 shirt + 0.10 pants = 0.25 total)
                cost = (
                    0.30 * sh_diff +
                    0.15 * aspect_diff +
                    0.15 * shirt_dist_weighted +
                    0.10 * pants_dist_weighted +
                    0.30 * motion_cost
                )
                
                cost_matrix[i, j] = cost
        
        return cost_matrix
    
    def match_tracks(
        self,
        detections: List[Dict],
        tracks: List[TrackV4Robust]
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
                new_track = TrackV4Robust(memory_id, detections[det_idx])
                self.tracks.append(new_track)
                if memory_id in self.track_memory:
                    del self.track_memory[memory_id]
            else:
                remaining_dets.append(det_idx)
        
        for det_idx in remaining_dets:
            new_track = TrackV4Robust(self.next_id, detections[det_idx])
            self.tracks.append(new_track)
            self.next_id += 1
        
        self.tracks = [t for t in self.tracks if not t.is_dead()]
        
        return [track.get_state() for track in self.tracks]
    
    def get_track_count(self) -> int:
        return len(self.tracks)
    
    def get_memory_count(self) -> int:
        return len(self.track_memory)
    
    def get_all_track_HSV_with_confidence(self) -> Dict[int, Dict]:
        """Get average HSV, variance, and confidence for all active tracks."""
        hsv_dict = {}
        for track in self.tracks:
            avg_shirt_HSV, avg_pants_HSV = track.get_average_HSV()
            smooth_shirt_HSV, smooth_pants_HSV = track.get_smoothed_HSV()
            shirt_H_std, pants_H_std = track.get_HSV_variance()
            avg_shirt_conf, avg_pants_conf = track.get_average_confidence()
            
            hsv_dict[track.id] = {
                'avg_shirt_HSV': avg_shirt_HSV,
                'avg_pants_HSV': avg_pants_HSV,
                'smooth_shirt_HSV': smooth_shirt_HSV,
                'smooth_pants_HSV': smooth_pants_HSV,
                'shirt_H_std': shirt_H_std,
                'pants_H_std': pants_H_std,
                'avg_shirt_confidence': avg_shirt_conf,
                'avg_pants_confidence': avg_pants_conf
            }
        return hsv_dict
    
    def reset(self):
        self.tracks = []
        self.track_memory = {}
        self.next_id = 0
