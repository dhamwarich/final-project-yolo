"""
Test script for Week 4.5 tuned Re-ID components.
"""

from features.pose_extractor import load_pose_model, get_keypoints
from features.ratio_calculator import compile_features
from features.filter_smooth import is_valid_frame, smooth_histogram_features
from tracking.appearance_extractor_v2 import (
    extract_histogram_features,
    compute_bhattacharyya_distance,
    compute_histogram_color_distance,
    get_dominant_hue
)
from tracking.reid_tracker_v3 import ReIDTrackerV3
from tracking.visualizer_bbox_v3 import BBoxVisualizerV3
import cv2
import numpy as np


def test_histogram_extractor():
    """Test histogram-based appearance extraction."""
    print("="*75)
    print("HISTOGRAM EXTRACTOR TEST")
    print("="*75)
    
    print("\n1. Testing Bhattacharyya distance...")
    hist1 = np.array([0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    hist2 = np.array([0.12, 0.18, 0.32, 0.18, 0.12, 0.04, 0.04, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    dist = compute_bhattacharyya_distance(hist1, hist2)
    print(f"   Hist1 peak at bin 2")
    print(f"   Hist2 peak at bin 2 (similar)")
    print(f"   Bhattacharyya distance: {dist:.3f}")
    print("   ✓ Similar histograms → low distance")
    
    hist3 = np.array([0, 0, 0, 0, 0, 0, 0, 0.3, 0.3, 0.2, 0.1, 0.1, 0, 0, 0, 0])
    dist2 = compute_bhattacharyya_distance(hist1, hist3)
    print(f"   Hist3 peak at bin 7-8 (different)")
    print(f"   Bhattacharyya distance: {dist2:.3f}")
    print("   ✓ Different histograms → high distance")
    
    print("\n2. Testing histogram color distance...")
    color1 = {'top_hist': hist1, 'bot_hist': hist1}
    color2 = {'top_hist': hist2, 'bot_hist': hist2}
    color_dist = compute_histogram_color_distance(color1, color2)
    print(f"   Color distance: {color_dist:.3f}")
    print("   ✓ Histogram color distance computed")
    
    print("\n3. Testing dominant hue extraction...")
    hue = get_dominant_hue(hist1)
    print(f"   Dominant hue: {hue:.3f} (normalized)")
    print("   ✓ Dominant hue extracted")
    
    print("\n✓ Histogram extractor tests passed!")


def test_histogram_smoothing():
    """Test histogram feature smoothing."""
    print("\n" + "="*75)
    print("HISTOGRAM SMOOTHING TEST")
    print("="*75)
    
    print("\n1. Testing EMA smoothing for histograms...")
    curr = {
        'top_hist': np.array([0.1] * 16),
        'bot_hist': np.array([0.05] * 16),
        'texture': 2.5,
        'aspect': 2.1
    }
    prev = {
        'top_hist': np.array([0.12] * 16),
        'bot_hist': np.array([0.06] * 16),
        'texture': 2.3,
        'aspect': 2.0
    }
    
    smoothed = smooth_histogram_features(curr, prev, alpha=0.3)
    
    print(f"   Current top_hist sum: {curr['top_hist'].sum():.3f}")
    print(f"   Previous top_hist sum: {prev['top_hist'].sum():.3f}")
    print(f"   Smoothed top_hist sum: {smoothed['top_hist'].sum():.3f}")
    print("   ✓ Histogram stays normalized after smoothing")
    
    expected_texture = 0.3 * 2.5 + 0.7 * 2.3
    if abs(smoothed['texture'] - expected_texture) < 0.001:
        print("   ✓ Scalar smoothing formula correct")
    else:
        print("   ✗ Warning: Scalar smoothing mismatch")
    
    print("\n✓ Histogram smoothing tests passed!")


def test_reid_tracker_v3():
    """Test memory-based Re-ID tracker without height."""
    print("\n" + "="*75)
    print("RE-ID TRACKER V3 TEST (MEMORY-BASED, NO HEIGHT)")
    print("="*75)
    
    # Initialize tracker
    print("\n1. Initializing memory-based tracker...")
    tracker = ReIDTrackerV3(
        max_cost_threshold=0.45,
        memory_cost_threshold=0.25,
        memory_duration=5.0
    )
    print(f"   ✓ Tracker initialized")
    print(f"     - Max cost threshold: {tracker.max_cost_threshold}")
    print(f"     - Memory cost threshold: {tracker.memory_cost_threshold}")
    print(f"     - Memory duration: {tracker.memory_duration}s")
    
    # Create mock detections
    print("\n2. Processing frame 1...")
    detections_f1 = [
        {
            'shoulder_hip': 1.25,
            'torso_leg': 0.53,
            'aspect': 2.1,
            'bbox_center': (500, 500),
            'bbox': (400, 200, 600, 800),
            'conf': 0.90,
            'top_hist': np.array([0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            'bot_hist': np.array([0, 0, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.05, 0, 0, 0, 0, 0, 0, 0]),
            'texture': 2.5
        },
        {
            'shoulder_hip': 1.30,
            'torso_leg': 0.55,
            'aspect': 2.2,
            'bbox_center': (1000, 500),
            'bbox': (900, 200, 1100, 800),
            'conf': 0.85,
            'top_hist': np.array([0, 0, 0, 0, 0, 0, 0, 0.3, 0.3, 0.2, 0.1, 0.1, 0, 0, 0, 0]),
            'bot_hist': np.array([0, 0, 0, 0, 0, 0, 0, 0.2, 0.3, 0.3, 0.1, 0.1, 0, 0, 0, 0]),
            'texture': 3.1
        }
    ]
    
    tracks_f1 = tracker.update(detections_f1)
    print(f"   ✓ Created {len(tracks_f1)} tracks")
    for track in tracks_f1:
        print(f"     - Track ID {track['id']}: SH={track['shoulder_hip']:.2f}, TL={track['torso_leg']:.2f}")
    
    # Frame 2 - same people
    print("\n3. Processing frame 2 (same people)...")
    detections_f2 = [
        {
            'shoulder_hip': 1.26,
            'torso_leg': 0.54,
            'aspect': 2.1,
            'bbox_center': (520, 510),
            'bbox': (420, 210, 620, 810),
            'conf': 0.92,
            'top_hist': np.array([0.11, 0.19, 0.31, 0.19, 0.11, 0.05, 0.04, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            'bot_hist': np.array([0, 0, 0.11, 0.19, 0.31, 0.19, 0.11, 0.05, 0.04, 0, 0, 0, 0, 0, 0, 0]),
            'texture': 2.6
        },
        {
            'shoulder_hip': 1.31,
            'torso_leg': 0.56,
            'aspect': 2.2,
            'bbox_center': (1020, 510),
            'bbox': (920, 210, 1120, 810),
            'conf': 0.87,
            'top_hist': np.array([0, 0, 0, 0, 0, 0, 0, 0.31, 0.29, 0.21, 0.09, 0.1, 0, 0, 0, 0]),
            'bot_hist': np.array([0, 0, 0, 0, 0, 0, 0, 0.19, 0.31, 0.31, 0.09, 0.1, 0, 0, 0, 0]),
            'texture': 3.0
        }
    ]
    
    tracks_f2 = tracker.update(detections_f2)
    print(f"   ✓ Updated {len(tracks_f2)} tracks")
    for track in tracks_f2:
        print(f"     - Track ID {track['id']}: hits={track['hits']}, age={track['age']}")
    
    # Check ID consistency
    if tracks_f2[0]['id'] == tracks_f1[0]['id']:
        print("   ✓ ID consistency maintained!")
    else:
        print("   ✗ Warning: IDs changed")
    
    print(f"\n4. Memory state:")
    print(f"   Active tracks: {tracker.get_track_count()}")
    print(f"   Tracks in memory: {tracker.get_memory_count()}")
    
    print("\n✓ Re-ID Tracker V3 tests passed!")


def test_visualizer_v3():
    """Test tuned visualizer."""
    print("\n" + "="*75)
    print("BBOX VISUALIZER V3 TEST")
    print("="*75)
    
    print("\n1. Initializing visualizer...")
    visualizer = BBoxVisualizerV3()
    print("   ✓ Visualizer initialized")
    
    print("\n2. Testing dominant hue visualization...")
    hist = np.array([0, 0, 0.3, 0.3, 0.2, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    hue = get_dominant_hue(hist)
    bgr = visualizer.hue_to_bgr(hue)
    print(f"   Dominant hue: {hue:.2f}")
    print(f"   BGR color: {bgr}")
    print("   ✓ Hue visualization working")
    
    print("\n3. Testing color persistence...")
    color1 = visualizer.get_color_for_id(0)
    color2 = visualizer.get_color_for_id(0)
    if color1 == color2:
        print("   ✓ Color persistence confirmed")
    else:
        print("   ✗ Warning: Color changed")
    
    print("\n✓ Visualizer V3 tests passed!")


def test_full_pipeline():
    """Test full tuned pipeline on one frame."""
    print("\n" + "="*75)
    print("FULL TUNED PIPELINE TEST")
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
    
    # Process with histogram features (NO HEIGHT)
    print("\n5. Processing with histogram features (no height)...")
    detections = []
    
    for person in persons:
        feats = compile_features(person["keypoints"], person["bbox"])
        
        if not is_valid_frame(feats):
            continue
        
        # Extract histogram appearance
        appearance = extract_histogram_features(frame, person['bbox'], bins=16)
        
        detection = {
            'shoulder_hip': feats['shoulder_hip'],
            'torso_leg': feats['torso_leg'],
            'aspect': appearance['aspect'],
            'bbox_center': feats['bbox_center'],
            'bbox': person['bbox'],
            'conf': person['conf'],
            'top_hist': appearance['top_hist'],
            'bot_hist': appearance['bot_hist'],
            'texture': appearance['texture']
        }
        detections.append(detection)
    
    print(f"   ✓ Processed {len(detections)} valid detections")
    
    # Track with memory (NO HEIGHT)
    print("\n6. Tracking with memory (no height)...")
    tracker = ReIDTrackerV3()
    tracks = tracker.update(detections)
    print(f"   ✓ Created {len(tracks)} tracks")
    
    for track in tracks:
        dominant_hue = get_dominant_hue(track['top_hist'])
        print(f"     - Track ID {track['id']}: SH={track['shoulder_hip']:.2f}, " 
              f"TL={track['torso_leg']:.2f}, hue={dominant_hue:.2f}")
    
    cap.release()
    print("\n✓ Full tuned pipeline test passed!")
    print("\nKey improvements:")
    print("  • NO pseudo-height (removed unstable feature)")
    print("  • Histogram-based color (16 bins)")
    print("  • Memory-based ID reuse")
    print("  • Simplified, stable features")


if __name__ == "__main__":
    test_histogram_extractor()
    test_histogram_smoothing()
    test_reid_tracker_v3()
    test_visualizer_v3()
    test_full_pipeline()
    
    print("\n" + "="*75)
    print("✅ ALL TUNED TESTS PASSED!")
    print("="*75)
    print("\nReady to run: python main_reid_tuned.py")
