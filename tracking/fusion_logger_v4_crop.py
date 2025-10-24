"""
Module: Fusion Logger V4 Cropped (With Color Names)
Logging with per-ID HSV values and human-readable color names.
"""

import csv
import os
import numpy as np
from collections import defaultdict
from typing import Dict, Tuple
from utils.color_utils import hue_to_name, hsv_to_full_description


def init_colorized_logger(csv_path: str = "outputs/reid_colorized.csv") -> None:
    """Create CSV header for colorized tracking data."""
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


def log_colorized_features(
    frame_idx: int,
    track: dict,
    csv_path: str
) -> None:
    """Append one row of colorized tracking data."""
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


def log_track_colors_with_names(
    track_hsv: Dict[int, Tuple[Tuple[float, float, float], Tuple[float, float, float]]],
    track_frame_counts: Dict[int, int],
    csv_path: str
) -> None:
    """
    Log per-ID HSV values and color names.
    
    Args:
        track_hsv: Dict mapping track_id to (shirt_HSV, pants_HSV)
        track_frame_counts: Dict mapping track_id to frame count
        csv_path: Base path for CSV file
    """
    color_csv_path = csv_path.replace('.csv', '_color_summary.csv')
    
    with open(color_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'track_id',
            'frames',
            'shirt_H',
            'shirt_S',
            'shirt_V',
            'shirt_color_name',
            'shirt_full_desc',
            'pants_H',
            'pants_S',
            'pants_V',
            'pants_color_name',
            'pants_full_desc'
        ])
        
        for track_id in sorted(track_hsv.keys()):
            shirt_HSV, pants_HSV = track_hsv[track_id]
            frame_count = track_frame_counts.get(track_id, 0)
            
            shirt_H, shirt_S, shirt_V = shirt_HSV
            pants_H, pants_S, pants_V = pants_HSV
            
            # Get color names
            shirt_name = hue_to_name(shirt_H)
            pants_name = hue_to_name(pants_H)
            
            # Get full descriptions
            shirt_desc = hsv_to_full_description(shirt_H, shirt_S, shirt_V)
            pants_desc = hsv_to_full_description(pants_H, pants_S, pants_V)
            
            writer.writerow([
                track_id,
                frame_count,
                f"{shirt_H:.1f}",
                f"{shirt_S:.1f}",
                f"{shirt_V:.1f}",
                shirt_name,
                shirt_desc,
                f"{pants_H:.1f}",
                f"{pants_S:.1f}",
                f"{pants_V:.1f}",
                pants_name,
                pants_desc
            ])
    
    print(f"✓ Track color summary with names saved to: {color_csv_path}")


def summarize_colorized_reid(csv_path: str) -> None:
    """Generate Re-ID summary for colorized tracking."""
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
    
    # Load color data
    color_csv_path = csv_path.replace('.csv', '_color_summary.csv')
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
                        'shirt_H': float(row['shirt_H']),
                        'shirt_S': float(row['shirt_S']),
                        'pants_H': float(row['pants_H']),
                        'pants_S': float(row['pants_S'])
                    }
                except (ValueError, KeyError):
                    continue
    
    summary_path = os.path.join(os.path.dirname(csv_path), 'reid_summary_colorized.txt')
    
    with open(summary_path, 'w') as f:
        f.write("="*75 + "\n")
        f.write("COLORIZED RE-ID TRACKING SUMMARY (TIGHT CROP + S>50)\n")
        f.write("="*75 + "\n\n")
        
        f.write(f"Total frames processed: {total_frames + 1}\n")
        f.write(f"Unique track IDs: {len(track_data)}\n\n")
        
        f.write("-"*75 + "\n")
        f.write("PER-TRACK STATISTICS (WITH COLORS)\n")
        f.write("-"*75 + "\n\n")
        
        all_retention_pcts = []
        
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
            
            # Add color information if available
            if track_id in track_colors:
                color_info = track_colors[track_id]
                f.write(f"  Shirt color: {color_info['shirt_desc']} (H={color_info['shirt_H']:.1f}°, S={color_info['shirt_S']:.1f})\n")
                f.write(f"  Pants color: {color_info['pants_desc']} (H={color_info['pants_H']:.1f}°, S={color_info['pants_S']:.1f})\n")
            
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
        
        color_csv_path = csv_path.replace('.csv', '_color_summary.csv')
        if os.path.exists(color_csv_path):
            f.write(f"\n✓ Per-ID color data with names: {os.path.basename(color_csv_path)}\n")
        
        f.write("\n" + "="*75 + "\n")
        f.write("WEEK 5.6 IMPROVEMENTS\n")
        f.write("="*75 + "\n\n")
        f.write("✓ Tight crop (10% padding inward) to exclude background\n")
        f.write("✓ Higher saturation threshold (S > 50) for vivid colors\n")
        f.write("✓ HSV mean extraction with masking\n")
        f.write("✓ Human-readable color names (red/pink, blue, green, etc.)\n")
        f.write("✓ Vivid color patches in visualization\n")
        f.write("✓ Per-ID color CSV with names and descriptions\n")
        f.write("\n" + "="*75 + "\n")
    
    print(f"\n✓ Colorized Re-ID summary saved to: {summary_path}")
    
    with open(summary_path, 'r') as f:
        print(f.read())
