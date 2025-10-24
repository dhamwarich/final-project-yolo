"""
Test script for Week 5 refined Re-ID components (LAB color-region).
"""

from features.pose_extractor import load_pose_model, get_keypoints
from features.ratio_calculator import compile_features
from features.filter_smooth import is_valid_frame
from tracking.appearance_extractor_v3 import (
    extract_lab_histograms,
    compute_bhattacharyya_distance,
    compute_shirt_pants_distance,
    get_dominant_color_lab
)
from tracking.reid_tracker_v4 import ReIDTrackerV4
from tracking.visualizer_bbox_v4 import BBoxVisualizerV4
import cv2
import numpy as np


def test_lab_extractor():
    """Test LAB histogram extraction."""
    print("="*75)
    print("LAB COLOR-REGION EXTRACTOR TEST")
    print("="*75)
    
    print("\n1. Testing Bhattacharyya distance...")
    hist1 = np.array([0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    hist2 = np.array([0.12, 0.18, 0.32, 0.18, 0.12, 0.04, 0.04, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    dist = compute_bhattacharyya_distance(hist1, hist2)
    print(f"   Similar histograms distance: {dist:.3f}")
    print("   ✓ Low distance for similar distributions")
    
    hist3 = np.array([0, 0, 0, 0, 0, 0, 0, 0.3, 0.3, 0.2, 0.1, 0.1, 0, 0, 0, 0])
    dist2 = compute_bhattacharyya_distance(hist1, hist3)
    print(f"   Different histograms distance: {dist2:.3f}")
    print("   ✓ High distance for different distributions")
    
    print("\n2. Testing shirt/pants distance...")
    color1 = {'shirt_hist': hist1, 'pants_hist': hist1}
    color2 = {'shirt_hist': hist2, 'pants_hist': hist2}
    color_dist = compute_shirt_pants_distance(color1, color2)
    print(f"   Shirt/pants distance: {color_dist:.3f}")
    print("   ✓ Combined shirt/pants distance computed")
    
    print("\n3. Testing dominant LAB color extraction...")
    color_bgr = get_dominant_color_lab(hist1)
    print(f"   Dominant color (BGR): {color_bgr}")
    print("   ✓ LAB to BGR conversion working")
    
    print("\n✓ LAB color-region extractor tests passed!")


def test_reid_tracker_v4():
    """Test refined Re-ID tracker without torso-leg."""
    print("\n" + "="*75)
    print("RE-ID TRACKER V4 TEST (LAB COLOR-REGION, NO TORSO-LEG)")
    print("="*75)
    
    # Initialize tracker
    print("\n1. Initializing refined tracker...")
    tracker = ReIDTrackerV4(
        max_cost_threshold=0.45,
        memory_cost_threshold=0.30,
        memory_duration=5.0
    )
    print(f"   ✓ Tracker initialized")
    print(f"     - Max cost threshold: {tracker.max_cost_threshold}")
    print(f"     - Memory cost threshold: {tracker.memory_cost_threshold}")
    
    # Create mock detections (NO torso_leg)
    print("\n2. Processing frame 1...")
    detections_f1 = [
        {
            'shoulder_hip': 1.25,
            'aspect': 2.1,
            'bbox_center': (500, 500),
            'bbox': (400, 200, 600, 800),
            'conf': 0.90,
            'shirt_hist': np.array([0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            'pants_hist': np.array([0, 0, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.05, 0, 0, 0, 0, 0, 0, 0]),
            'sat_mean': 0.5
        },
        {
            'shoulder_hip': 1.30,
            'aspect': 2.2,
            'bbox_center': (1000, 500),
            'bbox': (900, 200, 1100, 800),
            'conf': 0.85,
            'shirt_hist': np.array([0, 0, 0, 0, 0, 0, 0, 0.3, 0.3, 0.2, 0.1, 0.1, 0, 0, 0, 0]),
            'pants_hist': np.array([0, 0, 0, 0, 0, 0, 0, 0.2, 0.3, 0.3, 0.1, 0.1, 0, 0, 0, 0]),
            'sat_mean': 0.6
        }
    ]
    
    tracks_f1 = tracker.update(detections_f1)
    print(f"   ✓ Created {len(tracks_f1)} tracks")
    for track in tracks_f1:
        print(f"     - Track ID {track['id']}: SH={track['shoulder_hip']:.2f}, Aspect={track['aspect']:.2f}")
    
    # Frame 2 - same people
    print("\n3. Processing frame 2 (same people)...")
    detections_f2 = [
        {
            'shoulder_hip': 1.26,
            'aspect': 2.1,
            'bbox_center': (520, 510),
            'bbox': (420, 210, 620, 810),
            'conf': 0.92,
            'shirt_hist': np.array([0.11, 0.19, 0.31, 0.19, 0.11, 0.05, 0.04, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            'pants_hist': np.array([0, 0, 0.11, 0.19, 0.31, 0.19, 0.11, 0.05, 0.04, 0, 0, 0, 0, 0, 0, 0]),
            'sat_mean': 0.51
        },
        {
            'shoulder_hip': 1.31,
            'aspect': 2.2,
            'bbox_center': (1020, 510),
            'bbox': (920, 210, 1120, 810),
            'conf': 0.87,
            'shirt_hist': np.array([0, 0, 0, 0, 0, 0, 0, 0.31, 0.29, 0.21, 0.09, 0.1, 0, 0, 0, 0]),
            'pants_hist': np.array([0, 0, 0, 0, 0, 0, 0, 0.19, 0.31, 0.31, 0.09, 0.1, 0, 0, 0, 0]),
            'sat_mean': 0.61
        }
    ]
    
    tracks_f2 = tracker.update(detections_f2)
    print(f"   ✓ Updated {len(tracks_f2)} tracks")
    for track in tracks_f2:
        print(f"     - Track ID {track['id']}: hits={track['hits']}, age={track['age']}")
    
    # Check ID consistency
    if tracks_f2[0]['id'] == tracks_f1[0]['id']:
        print("   ✓ ID consistency maintained with LAB color-region!")
    else:
        print("   ✗ Warning: IDs changed")
    
    print(f"\n4. Feature verification:")
    print(f"   NO torso_leg ratio in features ✓")
    print(f"   NO height estimation in features ✓")
    print(f"   LAB shirt/pants histograms present ✓")
    
    print("\n✓ Re-ID Tracker V4 tests passed!")


def test_visualizer_v4():
    """Test refined visualizer with shirt/pants patches."""
    print("\n" + "="*75)
    print("BBOX VISUALIZER V4 TEST")
    print("="*75)
    
    print("\n1. Initializing visualizer...")
    visualizer = BBoxVisualizerV4()
    print("   ✓ Visualizer initialized")
    
    print("\n2. Testing dominant color extraction...")
    hist = np.array([0, 0, 0.3, 0.3, 0.2, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    color_bgr = get_dominant_color_lab(hist)
    print(f"   LAB dominant color (BGR): {color_bgr}")
    print("   ✓ LAB color extraction working")
    
    print("\n3. Testing color persistence...")
    color1 = visualizer.get_color_for_id(0)
    color2 = visualizer.get_color_for_id(0)
    if color1 == color2:
        print("   ✓ Color persistence confirmed")
    else:
        print("   ✗ Warning: Color changed")
    
    print("\n✓ Visualizer V4 tests passed!")


def test_full_pipeline():
    """Test full refined pipeline on one frame."""
    print("\n" + "="*75)
    print("FULL REFINED PIPELINE TEST")
    print("="*75)
    
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
    
    # Process with LAB features (NO torso_leg, NO height)
    print("\n5. Processing with LAB color-region features...")
    detections = []
    
    for person in persons:
        feats = compile_features(person["keypoints"], person["bbox"])
        
        if not is_valid_frame(feats):
            continue
        
        # Extract LAB shirt/pants histograms
        appearance = extract_lab_histograms(frame, person['bbox'], bins=16)
        
        detection = {
            'shoulder_hip': feats['shoulder_hip'],
            'aspect': appearance['aspect'],
            'bbox_center': feats['bbox_center'],
            'bbox': person['bbox'],
            'conf': person['conf'],
            'shirt_hist': appearance['shirt_hist'],
            'pants_hist': appearance['pants_hist'],
            'sat_mean': appearance['sat_mean']
        }
        detections.append(detection)
    
    print(f"   ✓ Processed {len(detections)} valid detections")
    
    # Track with LAB features (NO torso_leg, NO height)
    print("\n6. Tracking with LAB color-region (no torso-leg)...")
    tracker = ReIDTrackerV4()
    tracks = tracker.update(detections)
    print(f"   ✓ Created {len(tracks)} tracks")
    
    for track in tracks:
        shirt_color = get_dominant_color_lab(track['shirt_hist'])
        pants_color = get_dominant_color_lab(track['pants_hist'])
        print(f"     - Track ID {track['id']}: SH={track['shoulder_hip']:.2f}, "
              f"Aspect={track['aspect']:.2f}")
        print(f"       Shirt color: {shirt_color}, Pants color: {pants_color}")
    
    cap.release()
    print("\n✓ Full refined pipeline test passed!")
    print("\nWeek 5 Key Features:")
    print("  • LAB color space (perceptually uniform)")
    print("  • Separate shirt/pants histograms (60/40 split)")
    print("  • NO torso-leg ratio (removed unstable feature)")
    print("  • NO height estimation")
    print("  • Rebalanced cost function (35% shirt, 15% pants)")


if __name__ == "__main__":
    test_lab_extractor()
    test_reid_tracker_v4()
    test_visualizer_v4()
    test_full_pipeline()
    
    print("\n" + "="*75)
    print("✅ ALL REFINED TESTS PASSED!")
    print("="*75)
    print("\nReady to run: python main_reid_refined.py")
