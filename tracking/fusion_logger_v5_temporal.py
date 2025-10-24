"""
Module: Fusion Logger V5 Temporal (Motion-Based Metrics)
Logging with focus on temporal coherence, geometry, and motion.
"""

import csv
import os
import numpy as np
from collections import defaultdict
from typing import Dict
from utils.color_utils import hue_to_name, hsv_to_full_description


def init_temporal_logger(csv_path: str = "outputs/reid_temporal.csv") -> None:
    """Create CSV header for temporal tracking data."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'frame',
                'track_id',
                'shoulder_hip',
                'aspect',
                'velocity_x',
                'velocity_y',
                'x_center',
                'y_center',
                'conf',
                'hits',
                'misses'
            ])


def log_temporal_features(
    frame_idx: int,
    track: dict,
    csv_path: str
) -> None:
    """Append one row of temporal tracking data."""
    x_center, y_center = track['bbox_center']
    velocity_x, velocity_y = track.get('velocity', (0, 0))
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            frame_idx,
            track['id'],
            f"{track['shoulder_hip']:.2f}",
            f"{track.get('aspect', 1.0):.2f}",
            f"{velocity_x:.2f}",
            f"{velocity_y:.2f}",
            f"{x_center:.2f}",
            f"{y_center:.2f}",
            f"{track['conf']:.2f}",
            track.get('hits', 0),
            track.get('misses', 0)
        ])


def log_track_temporal_summary(
    track_stats: Dict[int, Dict],
    track_frame_counts: Dict[int, int],
    csv_path: str
) -> None:
    """
    Log per-ID temporal statistics.
    
    Args:
        track_stats: Dict mapping track_id to stats (geom_std, motion_std, conf)
        track_frame_counts: Dict mapping track_id to frame count
        csv_path: Base path for CSV file
    """
    summary_csv_path = csv_path.replace('.csv', '_temporal_summary.csv')
    
    with open(summary_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'track_id',
            'appearances',
            'hits',
            'misses',
            'avg_geom_std',
            'avg_motion_std',
            'shirt_conf_mean',
            'pants_conf_mean',
            'stable_shirt_color',
            'stable_pants_color',
            'coherence_score'
        ])
        
        for track_id in sorted(track_stats.keys()):
            stats = track_stats[track_id]
            frame_count = track_frame_counts.get(track_id, 0)
            
            hits = stats.get('hits', 0)
            misses = stats.get('misses', 0)
            avg_geom_std = stats.get('avg_geom_std', 0.0)
            avg_motion_std = stats.get('avg_motion_std', 0.0)
            shirt_conf = stats.get('shirt_confidence', 0.0)
            pants_conf = stats.get('pants_confidence', 0.0)
            
            # Get stable colors
            stable_shirt_HSV = stats.get('stable_shirt_HSV', (0, 0, 0))
            stable_pants_HSV = stats.get('stable_pants_HSV', (0, 0, 0))
            
            shirt_name = hue_to_name(stable_shirt_HSV[0])
            pants_name = hue_to_name(stable_pants_HSV[0])
            
            shirt_desc = hsv_to_full_description(stable_shirt_HSV[0], stable_shirt_HSV[1], stable_shirt_HSV[2])
            pants_desc = hsv_to_full_description(stable_pants_HSV[0], stable_pants_HSV[1], stable_pants_HSV[2])
            
            # Coherence score: low geometry variance + consistent motion
            coherence_score = 1.0 - min(avg_geom_std * 2.0, 1.0)  # Lower std = higher coherence
            coherence_ok = "✓" if coherence_score >= 0.8 else "✗"
            
            writer.writerow([
                track_id,
                frame_count,
                hits,
                misses,
                f"{avg_geom_std:.3f}",
                f"{avg_motion_std:.2f}",
                f"{shirt_conf:.2f}",
                f"{pants_conf:.2f}",
                shirt_desc,
                pants_desc,
                f"{coherence_score:.2f} {coherence_ok}"
            ])
    
    print(f"✓ Track temporal summary saved to: {summary_csv_path}")


def summarize_temporal_reid(csv_path: str) -> None:
    """Generate Re-ID summary for temporal tracking."""
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} does not exist")
        return
    
    track_data = defaultdict(list)
    total_frames = 0
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                track_id = int(row['track_id'])
                track_data[track_id].append({
                    'frame': int(row['frame']),
                    'shoulder_hip': float(row['shoulder_hip']),
                    'aspect': float(row.get('aspect', 1.0)),
                    'velocity_x': float(row.get('velocity_x', 0.0)),
                    'velocity_y': float(row.get('velocity_y', 0.0)),
                    'hits': int(row.get('hits', 0)),
                    'misses': int(row.get('misses', 0))
                })
                total_frames = max(total_frames, int(row['frame']))
            except (ValueError, KeyError):
                continue
    
    if len(track_data) == 0:
        print("Warning: No valid tracking data found")
        return
    
    # Load temporal summary data
    summary_csv_path = csv_path.replace('.csv', '_temporal_summary.csv')
    track_summary = {}
    
    if os.path.exists(summary_csv_path):
        with open(summary_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    track_id = int(row['track_id'])
                    track_summary[track_id] = {
                        'hits': int(row['hits']),
                        'misses': int(row['misses']),
                        'avg_geom_std': float(row['avg_geom_std']),
                        'avg_motion_std': float(row['avg_motion_std']),
                        'shirt_conf': float(row['shirt_conf_mean']),
                        'pants_conf': float(row['pants_conf_mean']),
                        'stable_shirt_color': row['stable_shirt_color'],
                        'stable_pants_color': row['stable_pants_color'],
                        'coherence_score': row['coherence_score']
                    }
                except (ValueError, KeyError):
                    continue
    
    summary_path = os.path.join(os.path.dirname(csv_path), 'reid_summary_temporal.txt')
    
    with open(summary_path, 'w') as f:
        f.write("="*75 + "\n")
        f.write("TEMPORAL RE-ID TRACKING (MOTION-DOMINANT MATCHING)\n")
        f.write("="*75 + "\n\n")
        
        f.write(f"Total frames processed: {total_frames + 1}\n")
        f.write(f"Unique track IDs: {len(track_data)}\n\n")
        
        f.write("-"*75 + "\n")
        f.write("PER-TRACK STATISTICS (TEMPORAL COHERENCE METRICS)\n")
        f.write("-"*75 + "\n\n")
        
        all_retention_pcts = []
        all_geom_stds = []
        all_motion_stds = []
        
        for track_id in sorted(track_data.keys()):
            data = track_data[track_id]
            
            sh_ratios = [d['shoulder_hip'] for d in data]
            aspects = [d['aspect'] for d in data]
            velocities = [(d['velocity_x'], d['velocity_y']) for d in data]
            
            sh_mean = np.mean(sh_ratios)
            sh_std = np.std(sh_ratios)
            aspect_mean = np.mean(aspects)
            
            velocity_mags = [np.sqrt(vx**2 + vy**2) for vx, vy in velocities]
            avg_velocity = np.mean(velocity_mags)
            
            retention_pct = (len(data) / (total_frames + 1)) * 100
            all_retention_pcts.append(retention_pct)
            
            f.write(f"Track ID {track_id}:\n")
            f.write(f"  Appearances: {len(data)} frames ({retention_pct:.1f}%)\n")
            f.write(f"  Shoulder/Hip: {sh_mean:.2f} ± {sh_std:.2f}\n")
            f.write(f"  Aspect ratio: {aspect_mean:.2f}\n")
            f.write(f"  Avg velocity: {avg_velocity:.1f} px/frame\n")
            
            # Add temporal summary info
            if track_id in track_summary:
                summary_info = track_summary[track_id]
                f.write(f"  Hits/Misses: {summary_info['hits']}/{summary_info['misses']}\n")
                f.write(f"  Geometry std: {summary_info['avg_geom_std']:.3f} (coherence metric)\n")
                f.write(f"  Motion std: {summary_info['avg_motion_std']:.2f} px/frame\n")
                f.write(f"  Color confidence: shirt={summary_info['shirt_conf']:.2f}, pants={summary_info['pants_conf']:.2f}\n")
                f.write(f"  Stable colors: {summary_info['stable_shirt_color']}, {summary_info['stable_pants_color']}\n")
                f.write(f"  Coherence: {summary_info['coherence_score']}\n")
                
                all_geom_stds.append(summary_info['avg_geom_std'])
                all_motion_stds.append(summary_info['avg_motion_std'])
            
            retention_ok = "✓" if retention_pct >= 85 else "✗"
            f.write(f"  ID retention: {retention_ok} (target: ≥85%)\n")
            f.write("\n")
        
        f.write("="*75 + "\n")
        f.write("OVERALL METRICS\n")
        f.write("="*75 + "\n\n")
        
        avg_retention = np.mean(all_retention_pcts)
        
        f.write(f"Average ID retention: {avg_retention:.1f}% ")
        if avg_retention >= 85:
            f.write("✓ (Target: ≥85%)\n")
        else:
            f.write("✗ (Target: ≥85%)\n")
        
        expected_people = min(6, len([r for r in all_retention_pcts if r > 15]))
        id_switches = max(0, len(track_data) - expected_people)
        switch_ok = "✓" if id_switches <= 2 else "✗"
        
        f.write(f"Estimated ID switches: {id_switches} {switch_ok} (Target: ≤2 per 350 frames)\n")
        f.write(f"Estimated real people: {expected_people}\n")
        
        # Temporal coherence metrics
        if all_geom_stds:
            avg_geom_std = np.mean(all_geom_stds)
            avg_motion_std = np.mean(all_motion_stds)
            
            f.write(f"\nTemporal Coherence Metrics:\n")
            f.write(f"  Average geometry variance: {avg_geom_std:.3f} ")
            f.write(f"{'✓' if avg_geom_std <= 0.1 else '✗'} (Target: ≤0.1)\n")
            f.write(f"  Average motion std: {avg_motion_std:.2f} px/frame\n")
        
        summary_csv_path = csv_path.replace('.csv', '_temporal_summary.csv')
        if os.path.exists(summary_csv_path):
            f.write(f"\n✓ Per-ID temporal data: {os.path.basename(summary_csv_path)}\n")
        
        f.write("\n" + "="*75 + "\n")
        f.write("WEEK 5.9 IMPROVEMENTS\n")
        f.write("="*75 + "\n\n")
        f.write("✓ Feature memory bank (geometry, motion, color with EMA)\n")
        f.write("✓ Motion-dominant matching (0.6*geom + 0.3*motion + 0.1*color)\n")
        f.write("✓ Motion similarity (exponential decay on velocity diff)\n")
        f.write("✓ Occlusion handling (max_age=10, ~0.3s buffer)\n")
        f.write("✓ Color confidence gating (ignore if conf<0.4)\n")
        f.write("✓ Long-term stable colors (15-frame EMA for visualization)\n")
        f.write("✓ Temporal coherence metrics (geometry std, motion std)\n")
        f.write("✓ Target: ID retention ≥85%, switches ≤2\n")
        f.write("\n" + "="*75 + "\n")
    
    print(f"\n✓ Temporal Re-ID summary saved to: {summary_path}")
    
    with open(summary_path, 'r') as f:
        print(f.read())
