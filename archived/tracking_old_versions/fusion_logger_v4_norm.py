"""
Module: Fusion Logger V4 Normalized (With Color Logging)
Logging with per-ID average shirt/pants colors for validation.
"""

import csv
import os
import numpy as np
from collections import defaultdict
from typing import Dict, Tuple


def init_normalized_logger(csv_path: str = "outputs/reid_normalized.csv") -> None:
    """
    Create CSV header for normalized tracking data (HSV shirt/pants).
    
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
                'aspect',
                'shirt_hist',
                'pants_hist',
                'x_center',
                'y_center',
                'conf'
            ])


def log_normalized_features(
    frame_idx: int,
    track: dict,
    csv_path: str
) -> None:
    """
    Append one row of normalized tracking data.
    
    Args:
        frame_idx: Current frame index
        track: Track dictionary with ID and features
        csv_path: Path to CSV file
    """
    x_center, y_center = track['bbox_center']
    
    # Convert histograms to compact string representation
    shirt_hist = track.get('shirt_hist', np.zeros(16))
    pants_hist = track.get('pants_hist', np.zeros(16))
    
    # Format as space-separated values
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


def log_track_colors(
    track_colors: Dict[int, Tuple[np.ndarray, np.ndarray]],
    track_frame_counts: Dict[int, int],
    csv_path: str
) -> None:
    """
    Log average shirt/pants colors per track ID to separate CSV.
    
    Args:
        track_colors: Dictionary mapping track_id to (avg_shirt_bgr, avg_pants_bgr)
        track_frame_counts: Dictionary mapping track_id to frame count
        csv_path: Base path for CSV file
    """
    color_csv_path = csv_path.replace('.csv', '_colors.csv')
    
    with open(color_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'track_id',
            'frames',
            'shirt_B',
            'shirt_G',
            'shirt_R',
            'pants_B',
            'pants_G',
            'pants_R'
        ])
        
        for track_id in sorted(track_colors.keys()):
            avg_shirt, avg_pants = track_colors[track_id]
            frame_count = track_frame_counts.get(track_id, 0)
            
            writer.writerow([
                track_id,
                frame_count,
                int(avg_shirt[0]),
                int(avg_shirt[1]),
                int(avg_shirt[2]),
                int(avg_pants[0]),
                int(avg_pants[1]),
                int(avg_pants[2])
            ])
    
    print(f"✓ Track colors saved to: {color_csv_path}")


def summarize_normalized_reid(csv_path: str) -> None:
    """
    Generate Re-ID summary for normalized tracking (HSV with saturation masking).
    
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
                    'aspect': float(row.get('aspect', 1.0))
                })
                total_frames = max(total_frames, int(row['frame']))
            except (ValueError, KeyError):
                continue
    
    if len(track_data) == 0:
        print("Warning: No valid tracking data found")
        return
    
    # Compute per-track statistics
    summary_path = os.path.join(os.path.dirname(csv_path), 'reid_summary_normalized.txt')
    
    with open(summary_path, 'w') as f:
        f.write("="*75 + "\n")
        f.write("NORMALIZED RE-ID TRACKING SUMMARY (HSV + SAT MASKING)\n")
        f.write("="*75 + "\n\n")
        
        f.write(f"Total frames processed: {total_frames + 1}\n")
        f.write(f"Unique track IDs: {len(track_data)}\n\n")
        
        f.write("-"*75 + "\n")
        f.write("PER-TRACK STATISTICS\n")
        f.write("-"*75 + "\n\n")
        
        # Per-track analysis
        all_retention_pcts = []
        all_sh_stds = []
        
        for track_id in sorted(track_data.keys()):
            data = track_data[track_id]
            
            sh_ratios = [d['shoulder_hip'] for d in data]
            aspects = [d['aspect'] for d in data]
            
            sh_mean = np.mean(sh_ratios)
            sh_std = np.std(sh_ratios)
            aspect_mean = np.mean(aspects)
            
            # Calculate retention (how many frames this ID appeared)
            retention_pct = (len(data) / (total_frames + 1)) * 100
            all_retention_pcts.append(retention_pct)
            all_sh_stds.append(sh_std)
            
            f.write(f"Track ID {track_id}:\n")
            f.write(f"  Appearances: {len(data)} frames ({retention_pct:.1f}%)\n")
            f.write(f"  Shoulder/Hip: {sh_mean:.2f} ± {sh_std:.2f}\n")
            f.write(f"  Aspect ratio: {aspect_mean:.2f}\n")
            
            # Check targets
            retention_ok = "✓" if retention_pct >= 80 else "✗"
            
            f.write(f"  ID retention: {retention_ok} (target: ≥80%)\n")
            f.write("\n")
        
        f.write("="*75 + "\n")
        f.write("OVERALL METRICS\n")
        f.write("="*75 + "\n\n")
        
        # Overall statistics
        avg_retention = np.mean(all_retention_pcts)
        avg_sh_std = np.mean(all_sh_stds)
        
        f.write(f"Average ID retention: {avg_retention:.1f}% ")
        if avg_retention >= 80:
            f.write("✓ (Target: ≥80%)\n")
        else:
            f.write("✗ (Target: ≥80%)\n")
        
        f.write(f"Average SH stability (std): {avg_sh_std:.3f}\n")
        
        # Estimate ID switches (assume ~4-6 real people in scene)
        expected_people = min(6, len([r for r in all_retention_pcts if r > 15]))
        id_switches = max(0, len(track_data) - expected_people)
        switch_ok = "✓" if id_switches <= 3 else "✗"
        
        f.write(f"Estimated ID switches: {id_switches} {switch_ok} (Target: ≤3 per 350 frames)\n")
        f.write(f"Estimated real people: {expected_people}\n")
        
        # Check for color CSV
        color_csv_path = csv_path.replace('.csv', '_colors.csv')
        if os.path.exists(color_csv_path):
            f.write(f"\n✓ Per-ID color data available: {os.path.basename(color_csv_path)}\n")
        
        f.write("\n" + "="*75 + "\n")
        f.write("WEEK 5.5 IMPROVEMENTS\n")
        f.write("="*75 + "\n\n")
        f.write("✓ HSV color space with saturation masking (S > 30)\n")
        f.write("✓ Gray pixel exclusion for vivid color extraction\n")
        f.write("✓ Per-ID average shirt/pants colors logged\n")
        f.write("✓ Normalized histograms for consistent matching\n")
        f.write("✓ Real-time color visualization (no more gray bias)\n")
        f.write("\n" + "="*75 + "\n")
    
    print(f"\n✓ Normalized Re-ID summary saved to: {summary_path}")
    
    # Print to console
    with open(summary_path, 'r') as f:
        print(f.read())
