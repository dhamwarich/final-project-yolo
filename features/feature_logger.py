"""
Module 3: Feature Logger
Save per-frame metrics for analysis and later Re-ID training.
"""

import csv
import os
import numpy as np


def init_logger(csv_path: str = "outputs/features.csv") -> None:
    """
    Create CSV header if not exists.
    
    Args:
        csv_path: Path to CSV file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Check if file exists
    if not os.path.exists(csv_path):
        # Create file with header
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'frame',
                'height_px',
                'shoulder_hip',
                'torso_leg',
                'x_center',
                'y_center',
                'conf'
            ])


def log_features(
    frame_idx: int,
    features: dict,
    conf: float,
    csv_path: str
) -> None:
    """
    Append one row of data.
    
    Args:
        frame_idx: Current frame index
        features: Dictionary of computed features
        conf: Confidence score
        csv_path: Path to CSV file
    """
    x_center, y_center = features['bbox_center']
    
    # Append row to CSV
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            frame_idx,
            f"{features['height_px']:.2f}",
            f"{features['shoulder_hip']:.2f}",
            f"{features['torso_leg']:.2f}",
            f"{x_center:.2f}",
            f"{y_center:.2f}",
            f"{conf:.2f}"
        ])


def summarize_log(csv_path: str) -> None:
    """
    Compute mean, std, and outlier percentage for each metric.
    Save summary as outputs/summary.txt.
    
    Args:
        csv_path: Path to CSV file to analyze
    """
    # Read CSV data
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} does not exist")
        return
    
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                data.append({
                    'frame': int(row['frame']),
                    'height_px': float(row['height_px']),
                    'shoulder_hip': float(row['shoulder_hip']),
                    'torso_leg': float(row['torso_leg']),
                    'conf': float(row['conf'])
                })
            except (ValueError, KeyError):
                # Skip malformed rows
                continue
    
    if len(data) == 0:
        print("Warning: No valid data found in CSV")
        return
    
    # Extract metrics
    heights = [d['height_px'] for d in data]
    sh_ratios = [d['shoulder_hip'] for d in data]
    tl_ratios = [d['torso_leg'] for d in data]
    confs = [d['conf'] for d in data]
    
    # Compute statistics
    total_frames = len(data)
    
    height_mean = np.mean(heights)
    height_std = np.std(heights)
    
    sh_mean = np.mean(sh_ratios)
    sh_std = np.std(sh_ratios)
    
    tl_mean = np.mean(tl_ratios)
    tl_std = np.std(tl_ratios)
    
    conf_mean = np.mean(confs)
    
    # Compute outlier percentage (values > 2 std from mean)
    def count_outliers(values, mean, std):
        if std == 0:
            return 0
        outliers = sum(1 for v in values if abs(v - mean) > 2 * std)
        return (outliers / len(values)) * 100
    
    height_outliers = count_outliers(heights, height_mean, height_std)
    sh_outliers = count_outliers(sh_ratios, sh_mean, sh_std)
    tl_outliers = count_outliers(tl_ratios, tl_mean, tl_std)
    
    # Generate summary report
    summary_path = os.path.join(os.path.dirname(csv_path), 'summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("FEATURE SUMMARY REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Total frames processed: {total_frames}\n")
        f.write(f"Average confidence: {conf_mean:.2f}\n\n")
        
        f.write("-"*60 + "\n")
        f.write("METRIC STATISTICS\n")
        f.write("-"*60 + "\n\n")
        
        f.write(f"Height (px):\n")
        f.write(f"  Mean:     {height_mean:.2f}\n")
        f.write(f"  Std:      {height_std:.2f}\n")
        f.write(f"  Outliers: {height_outliers:.1f}%\n\n")
        
        f.write(f"Shoulder/Hip Ratio:\n")
        f.write(f"  Mean:     {sh_mean:.2f}\n")
        f.write(f"  Std:      {sh_std:.2f}\n")
        f.write(f"  Outliers: {sh_outliers:.1f}%\n\n")
        
        f.write(f"Torso/Leg Ratio:\n")
        f.write(f"  Mean:     {tl_mean:.2f}\n")
        f.write(f"  Std:      {tl_std:.2f}\n")
        f.write(f"  Outliers: {tl_outliers:.1f}%\n\n")
        
        f.write("="*60 + "\n")
        f.write("TARGET EVALUATION\n")
        f.write("="*60 + "\n\n")
        
        # Check targets
        overall_outliers = (height_outliers + sh_outliers + tl_outliers) / 3
        
        f.write(f"Overall outlier rate: {overall_outliers:.1f}% ")
        if overall_outliers < 10:
            f.write("✓ (Target: <10%)\n")
        else:
            f.write("✗ (Target: <10%)\n")
        
        f.write(f"Shoulder/Hip variance: {sh_std:.2f} ")
        if sh_std < 0.1:
            f.write("✓ (Target: <0.1)\n")
        else:
            f.write("✗ (Target: <0.1)\n")
        
        f.write(f"Torso/Leg variance: {tl_std:.2f} ")
        if tl_std < 0.1:
            f.write("✓ (Target: <0.1)\n")
        else:
            f.write("✗ (Target: <0.1)\n")
        
        f.write("\n" + "="*60 + "\n")
    
    print(f"\n✓ Summary saved to: {summary_path}")
    
    # Also print to console
    with open(summary_path, 'r') as f:
        print(f.read())
