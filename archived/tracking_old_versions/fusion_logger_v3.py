"""
Module: Fusion Logger V3 (Tuned, No Height)
Simplified logging without pseudo-height, focusing on geometry + appearance.
"""

import csv
import os
import numpy as np
from collections import defaultdict


def init_tuned_logger(csv_path: str = "outputs/reid_tuned.csv") -> None:
    """
    Create CSV header for tuned tracking data (no height).
    
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
                'shoulder_hip',
                'torso_leg',
                'aspect',
                'top_hist',
                'bot_hist',
                'texture',
                'x_center',
                'y_center',
                'conf'
            ])


def log_tuned_features(
    frame_idx: int,
    track: dict,
    csv_path: str
) -> None:
    """
    Append one row of tuned tracking data.
    
    Args:
        frame_idx: Current frame index
        track: Track dictionary with ID and features
        csv_path: Path to CSV file
    """
    x_center, y_center = track['bbox_center']
    
    # Convert histograms to compact string representation
    top_hist = track.get('top_hist', np.zeros(16))
    bot_hist = track.get('bot_hist', np.zeros(16))
    
    # Format as space-separated values
    top_hist_str = ' '.join([f"{v:.3f}" for v in top_hist])
    bot_hist_str = ' '.join([f"{v:.3f}" for v in bot_hist])
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            frame_idx,
            track['id'],
            f"{track['shoulder_hip']:.2f}",
            f"{track['torso_leg']:.2f}",
            f"{track.get('aspect', 1.0):.2f}",
            top_hist_str,
            bot_hist_str,
            f"{track.get('texture', 0.0):.3f}",
            f"{x_center:.2f}",
            f"{y_center:.2f}",
            f"{track['conf']:.2f}"
        ])


def summarize_tuned_reid(csv_path: str) -> None:
    """
    Generate Re-ID summary for tuned tracking (no height).
    
    Computes:
    - ID retention rate
    - Torso/leg consistency
    - ID switching analysis
    - Memory effectiveness
    
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
                    'shoulder_hip': float(row['shoulder_hip']),
                    'torso_leg': float(row['torso_leg']),
                    'aspect': float(row.get('aspect', 1.0)),
                    'texture': float(row.get('texture', 0))
                })
                total_frames = max(total_frames, int(row['frame']))
            except (ValueError, KeyError):
                continue
    
    if len(track_data) == 0:
        print("Warning: No valid tracking data found")
        return
    
    # Compute per-track statistics
    summary_path = os.path.join(os.path.dirname(csv_path), 'reid_summary_tuned.txt')
    
    with open(summary_path, 'w') as f:
        f.write("="*75 + "\n")
        f.write("TUNED RE-ID TRACKING SUMMARY (NO HEIGHT)\n")
        f.write("="*75 + "\n\n")
        
        f.write(f"Total frames processed: {total_frames + 1}\n")
        f.write(f"Unique track IDs: {len(track_data)}\n\n")
        
        f.write("-"*75 + "\n")
        f.write("PER-TRACK STATISTICS\n")
        f.write("-"*75 + "\n\n")
        
        # Per-track analysis
        all_retention_pcts = []
        all_tl_vars = []
        
        for track_id in sorted(track_data.keys()):
            data = track_data[track_id]
            
            tl_ratios = [d['torso_leg'] for d in data]
            sh_ratios = [d['shoulder_hip'] for d in data]
            aspects = [d['aspect'] for d in data]
            
            tl_mean = np.mean(tl_ratios)
            tl_std = np.std(tl_ratios)
            sh_mean = np.mean(sh_ratios)
            aspect_mean = np.mean(aspects)
            
            # Calculate retention (how many frames this ID appeared)
            retention_pct = (len(data) / (total_frames + 1)) * 100
            all_retention_pcts.append(retention_pct)
            
            # Calculate TL variance
            tl_var_pct = (tl_std / max(tl_mean, 0.01)) * 100
            all_tl_vars.append(tl_var_pct)
            
            f.write(f"Track ID {track_id}:\n")
            f.write(f"  Appearances: {len(data)} frames ({retention_pct:.1f}%)\n")
            f.write(f"  Shoulder/Hip: {sh_mean:.2f}\n")
            f.write(f"  Torso/Leg: {tl_mean:.2f} ± {tl_std:.2f}\n")
            f.write(f"  Aspect ratio: {aspect_mean:.2f}\n")
            
            # Check targets
            tl_var_ok = "✓" if tl_var_pct <= 10 else "✗"
            retention_ok = "✓" if retention_pct >= 75 else "✗"
            
            f.write(f"  TL variance: {tl_var_ok} ({tl_var_pct:.1f}%, target: ≤10%)\n")
            f.write(f"  ID retention: {retention_ok} (target: ≥75%)\n")
            f.write("\n")
        
        f.write("="*75 + "\n")
        f.write("OVERALL METRICS\n")
        f.write("="*75 + "\n\n")
        
        # Overall statistics
        avg_retention = np.mean(all_retention_pcts)
        avg_tl_var = np.mean(all_tl_vars)
        
        f.write(f"Average ID retention: {avg_retention:.1f}% ")
        if avg_retention >= 75:
            f.write("✓ (Target: ≥75%)\n")
        else:
            f.write("✗ (Target: ≥75%)\n")
        
        f.write(f"Average TL variance: {avg_tl_var:.1f}% ")
        if avg_tl_var <= 10:
            f.write("✓ (Target: ≤10%)\n")
        else:
            f.write("✗ (Target: ≤10%)\n")
        
        # Estimate ID switches (assume ~4-6 real people in scene)
        expected_people = min(6, len([r for r in all_retention_pcts if r > 20]))
        id_switches = max(0, len(track_data) - expected_people)
        switch_ok = "✓" if id_switches <= 3 else "✗"
        
        f.write(f"Estimated ID switches: {id_switches} {switch_ok} (Target: ≤3 per 350 frames)\n")
        f.write(f"Estimated real people: {expected_people}\n")
        
        f.write("\n" + "="*75 + "\n")
        f.write("WEEK 4.5 IMPROVEMENTS\n")
        f.write("="*75 + "\n\n")
        f.write("✓ Removed unstable pseudo-height estimation\n")
        f.write("✓ Histogram-based color features (16 bins)\n")
        f.write("✓ Short-term memory (5s window)\n")
        f.write("✓ Simplified feature set (geometry + appearance + motion)\n")
        f.write("✓ Memory-based ID reuse for occlusions\n")
        f.write("\n" + "="*75 + "\n")
    
    print(f"\n✓ Tuned Re-ID summary saved to: {summary_path}")
    
    # Print to console
    with open(summary_path, 'r') as f:
        print(f.read())
