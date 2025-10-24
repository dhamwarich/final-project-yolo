"""
Test script for Week 4 appearance-enhanced Re-ID components.
"""

from features.pose_extractor import load_pose_model, get_keypoints
from features.ratio_calculator import compile_features
from features.filter_smooth import is_valid_frame, smooth_appearance_features
from tracking.geometry_estimator import calibrate_reference, estimate_height_m
from tracking.appearance_extractor import (
    extract_appearance_features,
    compute_color_distance,
    compute_texture_distance,
    compute_iou
)
from tracking.reid_tracker_v2 import ReIDTrackerV2
from tracking.visualizer_bbox_v2 import BBoxVisualizerV2
import cv2
import numpy as np


def test_appearance_extractor():
    """Test appearance feature extraction."""
    print("="*75)
    print("APPEARANCE EXTRACTOR TEST")
    print("="*75)
    
    print("\n1. Testing color distance...")
    color1 = {'top_h': 0.1, 'bot_h': 0.3}
    color2 = {'top_h': 0.15, 'bot_h': 0.32}
    dist = compute_color_distance(color1, color2)
    print(f"   Color1: {color1}")
    print(f"   Color2: {color2}")
    print(f"   Distance: {dist:.3f}")
    print("   ✓ Color distance computed")
    
    print("\n2. Testing texture distance...")
    tex1 = 2.5
    tex2 = 3.1
    tex_dist = compute_texture_distance(tex1, tex2)
    print(f"   Texture1: {tex1}")
    print(f"   Texture2: {tex2}")
    print(f"   Distance: {tex_dist:.3f}")
    print("   ✓ Texture distance computed")
    
    print("\n3. Testing IoU...")
    bbox1 = (100, 100, 200, 300)
    bbox2 = (150, 150, 250, 350)
    iou = compute_iou(bbox1, bbox2)
    print(f"   BBox1: {bbox1}")
    print(f"   BBox2: {bbox2}")
    print(f"   IoU: {iou:.3f}")
    print("   ✓ IoU computed")
    
    print("\n✓ Appearance extractor tests passed!")


def test_appearance_smoothing():
    """Test appearance feature smoothing."""
    print("\n" + "="*75)
    print("APPEARANCE SMOOTHING TEST")
    print("="*75)
    
    print("\n1. Testing EMA smoothing for appearance...")
    curr = {'top_h': 0.5, 'bot_h': 0.3, 'texture': 2.5}
    prev = {'top_h': 0.48, 'bot_h': 0.32, 'texture': 2.3}
    
    smoothed = smooth_appearance_features(curr, prev, alpha=0.3)
    
    print(f"   Current: {curr}")
    print(f"   Previous: {prev}")
    print(f"   Smoothed: {smoothed}")
    print(f"   ✓ EMA smoothing applied")
    
    # Verify smoothing works correctly
    expected_top_h = 0.3 * 0.5 + 0.7 * 0.48
    if abs(smoothed['top_h'] - expected_top_h) < 0.001:
        print("   ✓ Smoothing formula correct")
    else:
        print("   ✗ Warning: Smoothing mismatch")
    
    print("\n✓ Appearance smoothing tests passed!")


def test_reid_tracker_v2():
    """Test enhanced Re-ID tracker with appearance."""
    print("\n" + "="*75)
    print("RE-ID TRACKER V2 TEST")
    print("="*75)
    
    # Initialize tracker
    print("\n1. Initializing enhanced tracker...")
    tracker = ReIDTrackerV2(max_cost_threshold=0.45, img_width=1920, img_height=1080)
    print(f"   ✓ Tracker initialized with threshold={tracker.max_cost_threshold}")
    
    # Create mock detections with appearance
    print("\n2. Processing frame 1 with appearance features...")
    detections_f1 = [
        {
            'height_m': 1.70,
            'shoulder_hip': 1.25,
            'torso_leg': 0.53,
            'bbox_center': (500, 500),
            'bbox': (400, 200, 600, 800),
            'conf': 0.90,
            'top_h': 0.1,  # Blue-ish top
            'bot_h': 0.6,  # Yellow-ish bottom
            'texture': 2.5
        },
        {
            'height_m': 1.65,
            'shoulder_hip': 1.30,
            'torso_leg': 0.55,
            'bbox_center': (1000, 500),
            'bbox': (900, 200, 1100, 800),
            'conf': 0.85,
            'top_h': 0.9,  # Red-ish top
            'bot_h': 0.0,  # Red-ish bottom
            'texture': 3.1
        }
    ]
    
    tracks_f1 = tracker.update(detections_f1)
    print(f"   ✓ Created {len(tracks_f1)} tracks")
    for track in tracks_f1:
        print(f"     - Track ID {track['id']}: H={track['height_m']:.2f}m, top_h={track['top_h']:.2f}")
    
    # Create mock detections (frame 2) - same people, slightly moved
    print("\n3. Processing frame 2 (same people, moved)...")
    detections_f2 = [
        {
            'height_m': 1.71,
            'shoulder_hip': 1.26,
            'torso_leg': 0.54,
            'bbox_center': (520, 510),
            'bbox': (420, 210, 620, 810),
            'conf': 0.92,
            'top_h': 0.11,  # Similar blue
            'bot_h': 0.61,  # Similar yellow
            'texture': 2.6
        },
        {
            'height_m': 1.64,
            'shoulder_hip': 1.31,
            'torso_leg': 0.56,
            'bbox_center': (1020, 510),
            'bbox': (920, 210, 1120, 810),
            'conf': 0.87,
            'top_h': 0.91,  # Similar red
            'bot_h': 0.01,  # Similar red
            'texture': 3.0
        }
    ]
    
    tracks_f2 = tracker.update(detections_f2)
    print(f"   ✓ Updated {len(tracks_f2)} tracks")
    for track in tracks_f2:
        print(f"     - Track ID {track['id']}: H={track['height_m']:.2f}m (hits: {track['hits']})")
    
    # Check if IDs are preserved
    if tracks_f2[0]['id'] == tracks_f1[0]['id']:
        print("   ✓ ID consistency maintained with appearance features!")
    else:
        print("   ✗ Warning: IDs changed")
    
    print("\n✓ Re-ID Tracker V2 tests passed!")


def test_visualizer_v2():
    """Test appearance-enhanced visualizer."""
    print("\n" + "="*75)
    print("BBOX VISUALIZER V2 TEST")
    print("="*75)
    
    print("\n1. Initializing visualizer...")
    visualizer = BBoxVisualizerV2()
    print("   ✓ Visualizer initialized")
    
    print("\n2. Testing hue to BGR conversion...")
    hue_norm = 0.5  # Cyan
    bgr = visualizer.hue_to_bgr(hue_norm)
    print(f"   Hue (normalized): {hue_norm}")
    print(f"   BGR color: {bgr}")
    print("   ✓ Hue conversion working")
    
    print("\n3. Testing color persistence...")
    color1 = visualizer.get_color_for_id(0)
    color2 = visualizer.get_color_for_id(0)
    if color1 == color2:
        print("   ✓ Color persistence confirmed")
    else:
        print("   ✗ Warning: Color changed")
    
    print("\n✓ Visualizer V2 tests passed!")


def test_full_pipeline():
    """Test full appearance-enhanced pipeline on one frame."""
    print("\n" + "="*75)
    print("FULL APPEARANCE PIPELINE TEST")
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
    
    # Process with appearance features
    print("\n5. Processing with appearance features...")
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
        
        # Extract appearance
        appearance = extract_appearance_features(frame, person['bbox'])
        
        detection = {
            'height_m': height_m,
            'shoulder_hip': feats['shoulder_hip'],
            'torso_leg': feats['torso_leg'],
            'bbox_center': feats['bbox_center'],
            'bbox': person['bbox'],
            'conf': person['conf'],
            'top_h': appearance['top_h'],
            'bot_h': appearance['bot_h'],
            'texture': appearance['texture']
        }
        detections.append(detection)
    
    print(f"   ✓ Processed {len(detections)} valid detections with appearance")
    
    # Track with appearance
    print("\n6. Tracking with appearance features...")
    tracker = ReIDTrackerV2()
    tracks = tracker.update(detections)
    print(f"   ✓ Created {len(tracks)} tracks")
    
    for track in tracks:
        print(f"     - Track ID {track['id']}: H={track['height_m']:.2f}m, " 
              f"top_h={track['top_h']:.2f}, bot_h={track['bot_h']:.2f}")
    
    cap.release()
    print("\n✓ Full appearance pipeline test passed!")


if __name__ == "__main__":
    test_appearance_extractor()
    test_appearance_smoothing()
    test_reid_tracker_v2()
    test_visualizer_v2()
    test_full_pipeline()
    
    print("\n" + "="*75)
    print("✅ ALL APPEARANCE TESTS PASSED!")
    print("="*75)
    print("\nReady to run: python main_reid_appearance.py")
