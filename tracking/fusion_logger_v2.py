"""
Module: Fusion Logger V2 (Appearance-Enhanced)
Log tracking data with geometric + appearance features.
"""

import csv
import os
import numpy as np
from collections import defaultdict


def init_appearance_logger(csv_path: str = "outputs/reid_appearance.csv") -> None:
    """
    Create CSV header for appearance-enhanced tracking data.
    
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
                'height_norm',
                'shoulder_hip',
                'torso_leg',
                'top_h',
                'bot_h',
                'texture',
                'x_center',
                'y_center',
                'conf'
            ])


def log_appearance_features(
    frame_idx: int,
    track: dict,
    height_px: float,
    csv_path: str
) -> None:
    """
    Append one row of appearance-enhanced tracking data.
    
    Args:
        frame_idx: Current frame index
        track: Track dictionary with ID and features
        height_px: Pixel height
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
            f"{track['shoulder_hip']:.2f}",
            f"{track['torso_leg']:.2f}",
            f"{track.get('top_h', 0.0):.3f}",
            f"{track.get('bot_h', 0.0):.3f}",
            f"{track.get('texture', 0.0):.3f}",
            f"{x_center:.2f}",
            f"{y_center:.2f}",
            f"{track['conf']:.2f}"
        ])


def summarize_appearance_reid(csv_path: str) -> None:
    """
    Generate Re-ID summary with appearance features analysis.
    
    Computes:
    - Mean/std height per track ID
    - ID retention rate
    - Color consistency
    - Overall tracking quality
    
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
                    'height_norm': float(row['height_norm']),
                    'shoulder_hip': float(row['shoulder_hip']),
                    'torso_leg': float(row['torso_leg']),
                    'top_h': float(row.get('top_h', 0)),
                    'bot_h': float(row.get('bot_h', 0)),
                    'texture': float(row.get('texture', 0))
                })
                total_frames = max(total_frames, int(row['frame']))
            except (ValueError, KeyError):
                continue
    
    if len(track_data) == 0:
        print("Warning: No valid tracking data found")
        return
    
    # Compute per-track statistics
    summary_path = os.path.join(os.path.dirname(csv_path), 'reid_summary_v2.txt')
    
    with open(summary_path, 'w') as f:
        f.write("="*75 + "\n")
        f.write("APPEARANCE-ENHANCED RE-ID TRACKING SUMMARY\n")
        f.write("="*75 + "\n\n")
        
        f.write(f"Total frames processed: {total_frames + 1}\n")
        f.write(f"Unique track IDs: {len(track_data)}\n\n")
        
        f.write("-"*75 + "\n")
        f.write("PER-TRACK STATISTICS\n")
        f.write("-"*75 + "\n\n")
        
        # Per-track analysis
        all_retention_pcts = []
        all_height_stds = []
        all_tl_vars = []
        
        for track_id in sorted(track_data.keys()):
            data = track_data[track_id]
            
            heights = [d['height_norm'] for d in data]
            tl_ratios = [d['torso_leg'] for d in data]
            sh_ratios = [d['shoulder_hip'] for d in data]
            top_hs = [d['top_h'] for d in data]
            
            height_mean = np.mean(heights)
            height_std = np.std(heights)
            tl_mean = np.mean(tl_ratios)
            tl_std = np.std(tl_ratios)
            sh_mean = np.mean(sh_ratios)
            top_h_std = np.std(top_hs)
            
            # Calculate retention (how many frames this ID appeared)
            retention_pct = (len(data) / (total_frames + 1)) * 100
            all_retention_pcts.append(retention_pct)
            all_height_stds.append(height_std)
            
            # Calculate distance variance from torso-leg ratio
            tl_var_pct = (tl_std / max(tl_mean, 0.01)) * 100
            all_tl_vars.append(tl_var_pct)
            
            f.write(f"Track ID {track_id}:\n")
            f.write(f"  Appearances: {len(data)} frames ({retention_pct:.1f}%)\n")
            f.write(f"  Height (m): {height_mean:.2f} ± {height_std:.2f}\n")
            f.write(f"  Shoulder/Hip: {sh_mean:.2f}\n")
            f.write(f"  Torso/Leg: {tl_mean:.2f} ± {tl_std:.2f}\n")
            f.write(f"  Top color std: {top_h_std:.3f}\n")
            
            # Check targets
            height_ok = "✓" if height_std <= 0.15 else "✗"
            tl_var_ok = "✓" if tl_var_pct <= 15 else "✗"
            retention_ok = "✓" if retention_pct >= 80 else "✗"
            
            f.write(f"  Height stability: {height_ok} (target: std ≤ 0.15m)\n")
            f.write(f"  TL variance: {tl_var_ok} (variance: {tl_var_pct:.1f}%, target: ≤15%)\n")
            f.write(f"  ID retention: {retention_ok} (target: ≥80%)\n")
            f.write("\n")
        
        f.write("="*75 + "\n")
        f.write("OVERALL METRICS\n")
        f.write("="*75 + "\n\n")
        
        # Overall statistics
        avg_height_std = np.mean(all_height_stds)
        avg_retention = np.mean(all_retention_pcts)
        avg_tl_var = np.mean(all_tl_vars)
        
        f.write(f"Average height stability (std): {avg_height_std:.2f}m ")
        if avg_height_std <= 0.15:
            f.write("✓ (Target: ≤0.15m)\n")
        else:
            f.write("✗ (Target: ≤0.15m)\n")
        
        f.write(f"Average ID retention: {avg_retention:.1f}% ")
        if avg_retention >= 80:
            f.write("✓ (Target: ≥80%)\n")
        else:
            f.write("✗ (Target: ≥80%)\n")
        
        f.write(f"Average distance variance: {avg_tl_var:.1f}% ")
        if avg_tl_var <= 15:
            f.write("✓ (Target: ≤15%)\n")
        else:
            f.write("✗ (Target: ≤15%)\n")
        
        # ID switches estimate
        id_switches = max(0, len(track_data) - 4)  # Assume ~4 real people
        f.write(f"\nEstimated ID switches: {id_switches}\n")
        
        f.write("\n" + "="*75 + "\n")
        f.write("IMPROVEMENTS FROM WEEK 3\n")
        f.write("="*75 + "\n\n")
        f.write("✓ Appearance features (color, texture) added\n")
        f.write("✓ Motion-assisted matching (IoU + center proximity)\n")
        f.write("✓ Enhanced cost function with 6 components\n")
        f.write("✓ Improved EMA smoothing for stability\n")
        f.write("\n" + "="*75 + "\n")
    
    print(f"\n✓ Appearance Re-ID summary saved to: {summary_path}")
    
    # Print to console
    with open(summary_path, 'r') as f:
        print(f.read())
