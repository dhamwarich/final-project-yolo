"""
Module 3: Feature Logger
Save per-frame metrics for analysis and later Re-ID training.
"""

import csv
import os


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
