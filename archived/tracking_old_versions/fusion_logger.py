"""
Module 3: Fusion Logger
Log tracking data with physical-space metrics (height_m, distance_est).
"""

import csv
import os
import numpy as np
from collections import defaultdict


def init_fusion_logger(csv_path: str = "outputs/fused_features_sim.csv") -> None:
    """
    Create CSV header for tracking data.
    
    Args:
        csv_path: Path to CSV file
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'frame',
                'track_id',
                'height_px',
                'height_m_est',
                'dist_est',
                'shoulder_hip',
                'torso_leg',
                'x_center',
                'y_center',
                'conf'
            ])


def log_tracked_features(
    frame_idx: int,
    track: dict,
    height_px: float,
    dist_est: float,
    csv_path: str
) -> None:
    """
    Append one row of tracking data.
    
    Args:
        frame_idx: Current frame index
        track: Track dictionary with ID and features
        height_px: Pixel height
        dist_est: Estimated distance (meters)
        csv_path: Path to CSV file
    """
    x_center, y_center = track['bbox_center']
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            frame_idx,
            track['id'],
            f"{height_px:.2f}",
            f"{track['height_m']:.2f}",
            f"{dist_est:.2f}",
            f"{track['shoulder_hip']:.2f}",
            f"{track['torso_leg']:.2f}",
            f"{x_center:.2f}",
            f"{y_center:.2f}",
            f"{track['conf']:.2f}"
        ])


def summarize_reid(csv_path: str) -> None:
    """
    Generate Re-ID summary with per-track statistics.
    
    Computes:
    - Mean/std height per track ID
    - ID retention rate
    - Distance consistency
    
    Args:
        csv_path: Path to CSV file to analyze
    """
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} does not exist")
        return
    
    # Read data grouped by track_id
    track_data = defaultdict(list)
    total_frames = 0
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                track_id = int(row['track_id'])
                track_data[track_id].append({
                    'frame': int(row['frame']),
                    'height_m': float(row['height_m_est']),
                    'dist_est': float(row['dist_est']),
                    'shoulder_hip': float(row['shoulder_hip']),
                    'torso_leg': float(row['torso_leg'])
                })
                total_frames = max(total_frames, int(row['frame']))
            except (ValueError, KeyError):
                continue
    
    if len(track_data) == 0:
        print("Warning: No valid tracking data found")
        return
    
    # Compute per-track statistics
    summary_path = os.path.join(os.path.dirname(csv_path), 'reid_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("RE-ID TRACKING SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Total frames processed: {total_frames + 1}\n")
        f.write(f"Unique track IDs: {len(track_data)}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("PER-TRACK STATISTICS\n")
        f.write("-"*70 + "\n\n")
        
        # Per-track analysis
        for track_id in sorted(track_data.keys()):
            data = track_data[track_id]
            
            heights = [d['height_m'] for d in data]
            dists = [d['dist_est'] for d in data]
            sh_ratios = [d['shoulder_hip'] for d in data]
            
            height_mean = np.mean(heights)
            height_std = np.std(heights)
            dist_mean = np.mean(dists)
            dist_std = np.std(dists)
            sh_mean = np.mean(sh_ratios)
            
            # Calculate retention (how many frames this ID appeared)
            retention_pct = (len(data) / (total_frames + 1)) * 100
            
            f.write(f"Track ID {track_id}:\n")
            f.write(f"  Appearances: {len(data)} frames ({retention_pct:.1f}%)\n")
            f.write(f"  Height (m): {height_mean:.2f} ± {height_std:.2f}\n")
            f.write(f"  Distance (m): {dist_mean:.2f} ± {dist_std:.2f}\n")
            f.write(f"  Shoulder/Hip: {sh_mean:.2f}\n")
            
            # Check targets
            height_ok = "✓" if height_std <= 0.15 else "✗"
            dist_var_pct = (dist_std / max(dist_mean, 0.01)) * 100
            dist_ok = "✓" if dist_var_pct <= 10 else "✗"
            
            f.write(f"  Height stability: {height_ok} (target: std ≤ 0.15m)\n")
            f.write(f"  Distance consistency: {dist_ok} (variance: {dist_var_pct:.1f}%, target: ≤10%)\n")
            f.write("\n")
        
        f.write("="*70 + "\n")
        f.write("OVERALL METRICS\n")
        f.write("="*70 + "\n\n")
        
        # Overall statistics
        all_heights = []
        all_height_stds = []
        all_retention_pcts = []
        
        for track_id, data in track_data.items():
            heights = [d['height_m'] for d in data]
            all_heights.extend(heights)
            all_height_stds.append(np.std(heights))
            all_retention_pcts.append((len(data) / (total_frames + 1)) * 100)
        
        avg_height_std = np.mean(all_height_stds)
        avg_retention = np.mean(all_retention_pcts)
        
        f.write(f"Average height stability (std): {avg_height_std:.2f}m ")
        if avg_height_std <= 0.15:
            f.write("✓ (Target: ≤0.15m)\n")
        else:
            f.write("✗ (Target: ≤0.15m)\n")
        
        f.write(f"Average ID retention: {avg_retention:.1f}% ")
        if avg_retention >= 90:
            f.write("✓ (Target: ≥90%)\n")
        else:
            f.write("✗ (Target: ≥90%)\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"\n✓ Re-ID summary saved to: {summary_path}")
    
    # Print to console
    with open(summary_path, 'r') as f:
        print(f.read())
