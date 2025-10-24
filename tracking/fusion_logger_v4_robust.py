"""
Module: Fusion Logger V4 Robust (With Confidence Metrics)
Logging with hue variance and color confidence to quantify robustness.
"""

import csv
import os
import numpy as np
from collections import defaultdict
from typing import Dict
from utils.color_utils import hue_to_name, hsv_to_full_description


def init_robust_logger(csv_path: str = "outputs/reid_robust.csv") -> None:
    """Create CSV header for robust tracking data."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'frame',
                'track_id',
                'shoulder_hip',
                'aspect',
                'shirt_hist',
                'pants_hist',
                'x_center',
                'y_center',
                'conf'
            ])


def log_robust_features(
    frame_idx: int,
    track: dict,
    csv_path: str
) -> None:
    """Append one row of robust tracking data."""
    x_center, y_center = track['bbox_center']
    
    shirt_hist = track.get('shirt_hist', np.zeros(16))
    pants_hist = track.get('pants_hist', np.zeros(16))
    
    shirt_hist_str = ' '.join([f"{v:.3f}" for v in shirt_hist])
    pants_hist_str = ' '.join([f"{v:.3f}" for v in pants_hist])
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            frame_idx,
            track['id'],
            f"{track['shoulder_hip']:.2f}",
            f"{track.get('aspect', 1.0):.2f}",
            shirt_hist_str,
            pants_hist_str,
            f"{x_center:.2f}",
            f"{y_center:.2f}",
            f"{track['conf']:.2f}"
        ])


def log_track_colors_with_confidence(
    track_hsv_data: Dict[int, Dict],
    track_frame_counts: Dict[int, int],
    csv_path: str
) -> None:
    """
    Log per-ID HSV values, color names, variance, and confidence.
    
    Args:
        track_hsv_data: Dict mapping track_id to HSV data (avg, smooth, std, conf)
        track_frame_counts: Dict mapping track_id to frame count
        csv_path: Base path for CSV file
    """
    color_csv_path = csv_path.replace('.csv', '_color_stability.csv')
    
    with open(color_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'track_id',
            'frames',
            'shirt_H_mean',
            'shirt_H_std',
            'shirt_S_mean',
            'shirt_conf',
            'shirt_color_name',
            'shirt_full_desc',
            'pants_H_mean',
            'pants_H_std',
            'pants_S_mean',
            'pants_conf',
            'pants_color_name',
            'pants_full_desc',
            'hue_stability',
            'color_quality'
        ])
        
        for track_id in sorted(track_hsv_data.keys()):
            data = track_hsv_data[track_id]
            frame_count = track_frame_counts.get(track_id, 0)
            
            avg_shirt_HSV = data['avg_shirt_HSV']
            avg_pants_HSV = data['avg_pants_HSV']
            shirt_H_std = data['shirt_H_std']
            pants_H_std = data['pants_H_std']
            avg_shirt_conf = data['avg_shirt_confidence']
            avg_pants_conf = data['avg_pants_confidence']
            
            shirt_H, shirt_S, shirt_V = avg_shirt_HSV
            pants_H, pants_S, pants_V = avg_pants_HSV
            
            # Get color names
            shirt_name = hue_to_name(shirt_H)
            pants_name = hue_to_name(pants_H)
            
            # Get full descriptions
            shirt_desc = hsv_to_full_description(shirt_H, shirt_S, shirt_V)
            pants_desc = hsv_to_full_description(pants_H, pants_S, pants_V)
            
            # Hue stability check (target: ≤ 5°)
            hue_stability = "✓" if (shirt_H_std <= 5.0 and pants_H_std <= 5.0) else "✗"
            
            # Color quality check (target: S ≥ 70, conf ≥ 0.55)
            shirt_quality_ok = (shirt_S >= 70 and avg_shirt_conf >= 0.55)
            pants_quality_ok = (pants_S >= 70 and avg_pants_conf >= 0.55)
            color_quality = "✓" if (shirt_quality_ok and pants_quality_ok) else "✗"
            
            writer.writerow([
                track_id,
                frame_count,
                f"{shirt_H:.1f}",
                f"{shirt_H_std:.2f}",
                f"{shirt_S:.1f}",
                f"{avg_shirt_conf:.2f}",
                shirt_name,
                shirt_desc,
                f"{pants_H:.1f}",
                f"{pants_H_std:.2f}",
                f"{pants_S:.1f}",
                f"{avg_pants_conf:.2f}",
                pants_name,
                pants_desc,
                hue_stability,
                color_quality
            ])
    
    print(f"✓ Track color stability with confidence saved to: {color_csv_path}")


def summarize_robust_reid(csv_path: str) -> None:
    """Generate Re-ID summary for robust tracking."""
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
                    'aspect': float(row.get('aspect', 1.0))
                })
                total_frames = max(total_frames, int(row['frame']))
            except (ValueError, KeyError):
                continue
    
    if len(track_data) == 0:
        print("Warning: No valid tracking data found")
        return
    
    # Load color data with variance and confidence
    color_csv_path = csv_path.replace('.csv', '_color_stability.csv')
    track_colors = {}
    
    if os.path.exists(color_csv_path):
        with open(color_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    track_id = int(row['track_id'])
                    track_colors[track_id] = {
                        'shirt_desc': row['shirt_full_desc'],
                        'pants_desc': row['pants_full_desc'],
                        'shirt_H': float(row['shirt_H_mean']),
                        'shirt_H_std': float(row['shirt_H_std']),
                        'shirt_S': float(row['shirt_S_mean']),
                        'shirt_conf': float(row['shirt_conf']),
                        'pants_H': float(row['pants_H_mean']),
                        'pants_H_std': float(row['pants_H_std']),
                        'pants_S': float(row['pants_S_mean']),
                        'pants_conf': float(row['pants_conf']),
                        'hue_stability': row['hue_stability'],
                        'color_quality': row['color_quality']
                    }
                except (ValueError, KeyError):
                    continue
    
    summary_path = os.path.join(os.path.dirname(csv_path), 'reid_summary_robust.txt')
    
    with open(summary_path, 'w') as f:
        f.write("="*75 + "\n")
        f.write("ROBUST RE-ID TRACKING (EXPANDED REGIONS + SV GATING + CONFIDENCE)\n")
        f.write("="*75 + "\n\n")
        
        f.write(f"Total frames processed: {total_frames + 1}\n")
        f.write(f"Unique track IDs: {len(track_data)}\n\n")
        
        f.write("-"*75 + "\n")
        f.write("PER-TRACK STATISTICS (WITH ROBUSTNESS METRICS)\n")
        f.write("-"*75 + "\n\n")
        
        all_retention_pcts = []
        all_shirt_stds = []
        all_pants_stds = []
        all_shirt_confs = []
        all_pants_confs = []
        
        for track_id in sorted(track_data.keys()):
            data = track_data[track_id]
            
            sh_ratios = [d['shoulder_hip'] for d in data]
            aspects = [d['aspect'] for d in data]
            
            sh_mean = np.mean(sh_ratios)
            sh_std = np.std(sh_ratios)
            aspect_mean = np.mean(aspects)
            
            retention_pct = (len(data) / (total_frames + 1)) * 100
            all_retention_pcts.append(retention_pct)
            
            f.write(f"Track ID {track_id}:\n")
            f.write(f"  Appearances: {len(data)} frames ({retention_pct:.1f}%)\n")
            f.write(f"  Shoulder/Hip: {sh_mean:.2f} ± {sh_std:.2f}\n")
            f.write(f"  Aspect ratio: {aspect_mean:.2f}\n")
            
            # Add color information with variance and confidence
            if track_id in track_colors:
                color_info = track_colors[track_id]
                f.write(f"  Shirt color: {color_info['shirt_desc']} (H={color_info['shirt_H']:.1f}° ± {color_info['shirt_H_std']:.2f}°, S={color_info['shirt_S']:.1f}, conf={color_info['shirt_conf']:.2f})\n")
                f.write(f"  Pants color: {color_info['pants_desc']} (H={color_info['pants_H']:.1f}° ± {color_info['pants_H_std']:.2f}°, S={color_info['pants_S']:.1f}, conf={color_info['pants_conf']:.2f})\n")
                f.write(f"  Color stability: {color_info['hue_stability']} (shirt σ={color_info['shirt_H_std']:.2f}°, pants σ={color_info['pants_H_std']:.2f}°, target: ≤5°)\n")
                f.write(f"  Color quality: {color_info['color_quality']} (shirt S={color_info['shirt_S']:.1f}, conf={color_info['shirt_conf']:.2f}, target: S≥70, conf≥0.55)\n")
                
                all_shirt_stds.append(color_info['shirt_H_std'])
                all_pants_stds.append(color_info['pants_H_std'])
                all_shirt_confs.append(color_info['shirt_conf'])
                all_pants_confs.append(color_info['pants_conf'])
            
            retention_ok = "✓" if retention_pct >= 75 else "✗"
            f.write(f"  ID retention: {retention_ok} (target: ≥75%)\n")
            f.write("\n")
        
        f.write("="*75 + "\n")
        f.write("OVERALL METRICS\n")
        f.write("="*75 + "\n\n")
        
        avg_retention = np.mean(all_retention_pcts)
        
        f.write(f"Average ID retention: {avg_retention:.1f}% ")
        if avg_retention >= 75:
            f.write("✓ (Target: ≥75%)\n")
        else:
            f.write("✗ (Target: ≥75%)\n")
        
        expected_people = min(6, len([r for r in all_retention_pcts if r > 15]))
        id_switches = max(0, len(track_data) - expected_people)
        switch_ok = "✓" if id_switches <= 3 else "✗"
        
        f.write(f"Estimated ID switches: {id_switches} {switch_ok} (Target: ≤3 per 350 frames)\n")
        f.write(f"Estimated real people: {expected_people}\n")
        
        # Color robustness metrics
        if all_shirt_stds:
            avg_shirt_std = np.mean(all_shirt_stds)
            avg_pants_std = np.mean(all_pants_stds)
            avg_shirt_conf = np.mean(all_shirt_confs)
            avg_pants_conf = np.mean(all_pants_confs)
            
            f.write(f"\nColor Robustness Metrics:\n")
            f.write(f"  Average shirt hue variance: {avg_shirt_std:.2f}° ")
            f.write(f"{'✓' if avg_shirt_std <= 5.0 else '✗'} (Target: ≤5°)\n")
            f.write(f"  Average pants hue variance: {avg_pants_std:.2f}° ")
            f.write(f"{'✓' if avg_pants_std <= 5.0 else '✗'} (Target: ≤5°)\n")
            f.write(f"  Average shirt confidence: {avg_shirt_conf:.2f} ")
            f.write(f"{'✓' if avg_shirt_conf >= 0.55 else '✗'} (Target: ≥0.55)\n")
            f.write(f"  Average pants confidence: {avg_pants_conf:.2f} ")
            f.write(f"{'✓' if avg_pants_conf >= 0.55 else '✗'} (Target: ≥0.55)\n")
        
        color_csv_path = csv_path.replace('.csv', '_color_stability.csv')
        if os.path.exists(color_csv_path):
            f.write(f"\n✓ Per-ID color stability data: {os.path.basename(color_csv_path)}\n")
        
        f.write("\n" + "="*75 + "\n")
        f.write("WEEK 5.8 IMPROVEMENTS\n")
        f.write("="*75 + "\n\n")
        f.write("✓ Expanded shirt region (20-60% height, 25-75% width, 40%×50% area)\n")
        f.write("✓ Saturation/brightness gating (S>60, V∈[40,220])\n")
        f.write("✓ Adaptive temporal smoothing (α=0.3→0.5 when σ>10°)\n")
        f.write("✓ Confidence weighting in cost matrix (based on saturation)\n")
        f.write("✓ Reduced color weight (0.15 shirt + 0.10 pants = 0.25 total)\n")
        f.write("✓ Color confidence logging (quantifies color quality)\n")
        f.write("✓ Target: σ≤5°, S≥70, conf≥0.55\n")
        f.write("\n" + "="*75 + "\n")
    
    print(f"\n✓ Robust Re-ID summary saved to: {summary_path}")
    
    with open(summary_path, 'r') as f:
        print(f.read())
