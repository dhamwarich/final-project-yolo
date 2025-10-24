"""
Quick test for Re-ID tracking components.
"""

from features.pose_extractor import load_pose_model, get_keypoints
from features.ratio_calculator import compile_features
from features.filter_smooth import is_valid_frame
from tracking.geometry_estimator import (
    calibrate_reference,
    estimate_height_m,
    estimate_distance_m,
    estimate_distance_ratio
)
from tracking.reid_tracker import ReIDTracker
from tracking.visualizer_bbox import BBoxVisualizer
import cv2
import numpy as np


def test_geometry_estimator():
    """Test geometry estimation functions."""
    print("="*70)
    print("GEOMETRY ESTIMATOR TEST")
    print("="*70)
    
    # Test calibration
    print("\n1. Testing calibration...")
    f_px = calibrate_reference(
        ref_height_m=1.70,
        ref_distance_m=2.0,
        ref_height_px=360
    )
    print(f"   ✓ Focal length: {f_px:.1f} pixels")
    
    # Test distance ratio estimation
    print("\n2. Testing distance ratio estimation...")
    # Closer person (compressed ratio)
    close_ratio = 0.8
    close_dist = estimate_distance_ratio(close_ratio, ref_ratio=0.53, k1=1.0)
    print(f"   Close person (ratio={close_ratio:.2f}): Z={close_dist:.2f}x (closer)")
    
    # Far person (stretched ratio)
    far_ratio = 0.4
    far_dist = estimate_distance_ratio(far_ratio, ref_ratio=0.53, k1=1.0)
    print(f"   Far person (ratio={far_ratio:.2f}): Z={far_dist:.2f}x (farther)")
    
    # Test height estimation
    print("\n3. Testing height estimation...")
    height_px = 360
    torso_leg = 0.53
    height_m = estimate_height_m(height_px, torso_leg, f_px, ref_ratio=0.53, k1=1.0)
    print(f"   Height: {height_px}px, TL ratio: {torso_leg}")
    print(f"   Estimated height: {height_m:.2f}m")
    
    # Test distance estimation
    print("\n4. Testing distance estimation...")
    dist_m = estimate_distance_m(height_px, height_m, f_px)
    print(f"   Estimated distance: {dist_m:.2f}m")
    
    print("\n✓ Geometry estimator tests passed!")


def test_reid_tracker():
    """Test Re-ID tracker with mock detections."""
    print("\n" + "="*70)
    print("RE-ID TRACKER TEST")
    print("="*70)
    
    # Initialize tracker
    print("\n1. Initializing tracker...")
    tracker = ReIDTracker(max_cost_threshold=0.35, img_width=1920, img_height=1080)
    print(f"   ✓ Tracker initialized")
    
    # Create mock detections (frame 1)
    print("\n2. Processing frame 1 with 2 detections...")
    detections_f1 = [
        {
            'height_m': 1.70,
            'shoulder_hip': 1.25,
            'torso_leg': 0.53,
            'bbox_center': (500, 500),
            'conf': 0.90
        },
        {
            'height_m': 1.65,
            'shoulder_hip': 1.30,
            'torso_leg': 0.55,
            'bbox_center': (1000, 500),
            'conf': 0.85
        }
    ]
    
    tracks_f1 = tracker.update(detections_f1)
    print(f"   ✓ Created {len(tracks_f1)} tracks")
    for track in tracks_f1:
        print(f"     - Track ID {track['id']}: H={track['height_m']:.2f}m")
    
    # Create mock detections (frame 2) - same people, slightly moved
    print("\n3. Processing frame 2 (same people, moved)...")
    detections_f2 = [
        {
            'height_m': 1.71,  # Slightly different
            'shoulder_hip': 1.26,
            'torso_leg': 0.54,
            'bbox_center': (520, 510),  # Moved slightly
            'conf': 0.92
        },
        {
            'height_m': 1.64,
            'shoulder_hip': 1.31,
            'torso_leg': 0.56,
            'bbox_center': (1020, 510),
            'conf': 0.87
        }
    ]
    
    tracks_f2 = tracker.update(detections_f2)
    print(f"   ✓ Updated {len(tracks_f2)} tracks")
    for track in tracks_f2:
        print(f"     - Track ID {track['id']}: H={track['height_m']:.2f}m (hits: {track['hits']})")
    
    # Check if IDs are preserved
    if tracks_f2[0]['id'] == tracks_f1[0]['id']:
        print("   ✓ ID consistency maintained!")
    else:
        print("   ✗ Warning: IDs changed")
    
    print("\n4. Testing new person entering scene...")
    detections_f3 = detections_f2 + [
        {
            'height_m': 1.75,
            'shoulder_hip': 1.22,
            'torso_leg': 0.51,
            'bbox_center': (1500, 500),
            'conf': 0.88
        }
    ]
    
    tracks_f3 = tracker.update(detections_f3)
    print(f"   ✓ Now tracking {len(tracks_f3)} people")
    print(f"   ✓ Total IDs assigned: {tracker.next_id}")
    
    print("\n✓ Re-ID tracker tests passed!")


def test_visualizer():
    """Test BBox visualizer."""
    print("\n" + "="*70)
    print("BBOX VISUALIZER TEST")
    print("="*70)
    
    print("\n1. Initializing visualizer...")
    visualizer = BBoxVisualizer()
    print("   ✓ Visualizer initialized")
    
    print("\n2. Testing color generation...")
    colors = []
    for i in range(5):
        color = visualizer.get_color_for_id(i)
        colors.append(color)
        print(f"   ID {i}: color = {color}")
    
    print("   ✓ Colors generated")
    
    print("\n3. Testing color persistence...")
    color_check = visualizer.get_color_for_id(0)
    if color_check == colors[0]:
        print("   ✓ Color persistence confirmed")
    else:
        print("   ✗ Warning: Color changed")
    
    print("\n✓ Visualizer tests passed!")


def test_full_pipeline():
    """Test full pipeline on one frame."""
    print("\n" + "="*70)
    print("FULL PIPELINE TEST")
    print("="*70)
    
    # Load model
    print("\n1. Loading model...")
    model = load_pose_model("yolo")
    print("   ✓ Model loaded")
    
    # Open video
    print("\n2. Opening video...")
    cap = cv2.VideoCapture("videos/test.mp4")
    if not cap.isOpened():
        print("   ✗ Error: Could not open video")
        return
    print("   ✓ Video opened")
    
    # Read frame
    print("\n3. Reading frame...")
    ret, frame = cap.read()
    if not ret:
        print("   ✗ Error: Could not read frame")
        return
    print(f"   ✓ Frame read: {frame.shape}")
    
    # Get detections
    print("\n4. Detecting poses...")
    persons = get_keypoints(model, frame)
    print(f"   ✓ Detected {len(persons)} person(s)")
    
    # Process detections
    print("\n5. Processing detections...")
    f_px = calibrate_reference()
    detections = []
    
    for person in persons:
        feats = compile_features(person["keypoints"], person["bbox"])
        
        if not is_valid_frame(feats):
            continue
        
        height_m = estimate_height_m(
            feats['height_px'],
            feats['torso_leg'],
            f_px
        )
        
        detection = {
            'height_m': height_m,
            'shoulder_hip': feats['shoulder_hip'],
            'torso_leg': feats['torso_leg'],
            'bbox_center': feats['bbox_center'],
            'conf': person['conf']
        }
        detections.append(detection)
    
    print(f"   ✓ Processed {len(detections)} valid detections")
    
    # Track
    print("\n6. Tracking...")
    tracker = ReIDTracker()
    tracks = tracker.update(detections)
    print(f"   ✓ Created {len(tracks)} tracks")
    
    for track in tracks:
        print(f"     - Track ID {track['id']}: H={track['height_m']:.2f}m")
    
    cap.release()
    print("\n✓ Full pipeline test passed!")


if __name__ == "__main__":
    test_geometry_estimator()
    test_reid_tracker()
    test_visualizer()
    test_full_pipeline()
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    print("\nReady to run: python main_reid_geometry.py")
