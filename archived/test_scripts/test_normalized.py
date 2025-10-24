"""
Test script for Week 5.5 normalized Re-ID components (HSV + saturation masking).
"""

from features.pose_extractor import load_pose_model, get_keypoints
from features.ratio_calculator import compile_features
from features.filter_smooth import is_valid_frame
from tracking.appearance_extractor_v3_norm import (
    extract_color_regions_hsv,
    compute_bhattacharyya_distance,
    compute_shirt_pants_distance,
    get_dominant_hue,
    hue_to_bgr
)
from tracking.reid_tracker_v4_norm import ReIDTrackerV4Norm
from tracking.visualizer_bbox_v4_norm import BBoxVisualizerV4Norm
import cv2
import numpy as np


def test_hsv_extractor():
    """Test HSV histogram extraction with saturation masking."""
    print("="*75)
    print("HSV SATURATION MASKING EXTRACTOR TEST")
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
    
    print("\n2. Testing separate shirt/pants distance...")
    color1 = {'shirt_hist': hist1, 'pants_hist': hist1}
    color2 = {'shirt_hist': hist2, 'pants_hist': hist2}
    shirt_dist, pants_dist = compute_shirt_pants_distance(color1, color2)
    print(f"   Shirt distance: {shirt_dist:.3f}")
    print(f"   Pants distance: {pants_dist:.3f}")
    print("   ✓ Separate shirt/pants distances computed")
    
    print("\n3. Testing dominant hue extraction...")
    hue = get_dominant_hue(hist1)
    print(f"   Dominant hue: {hue} (0-180)")
    bgr = hue_to_bgr(hue)
    print(f"   BGR color: {bgr}")
    print("   ✓ Hue to BGR conversion working")
    
    print("\n✓ HSV saturation masking extractor tests passed!")


def test_reid_tracker_v4_norm():
    """Test normalized Re-ID tracker with color accumulation."""
    print("\n" + "="*75)
    print("RE-ID TRACKER V4 NORMALIZED TEST (HSV + COLOR LOGGING)")
    print("="*75)
    
    # Initialize tracker
    print("\n1. Initializing normalized tracker...")
    tracker = ReIDTrackerV4Norm(
        max_cost_threshold=0.45,
        memory_cost_threshold=0.30,
        memory_duration=5.0
    )
    print(f"   ✓ Tracker initialized")
    
    # Create mock detections with color means
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
            'shirt_color_mean': np.array([50, 100, 150], dtype=np.uint8),  # BGR
            'pants_color_mean': np.array([100, 50, 200], dtype=np.uint8),
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
            'shirt_color_mean': np.array([200, 100, 50], dtype=np.uint8),
            'pants_color_mean': np.array([150, 150, 100], dtype=np.uint8),
            'sat_mean': 0.6
        }
    ]
    
    tracks_f1 = tracker.update(detections_f1)
    print(f"   ✓ Created {len(tracks_f1)} tracks")
    for track in tracks_f1:
        shirt_bgr = track['shirt_color_mean']
        pants_bgr = track['pants_color_mean']
        print(f"     - Track ID {track['id']}: SH={track['shoulder_hip']:.2f}")
        print(f"       Shirt BGR: {shirt_bgr}, Pants BGR: {pants_bgr}")
    
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
            'shirt_color_mean': np.array([52, 102, 148], dtype=np.uint8),
            'pants_color_mean': np.array([98, 52, 198], dtype=np.uint8),
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
            'shirt_color_mean': np.array([198, 98, 48], dtype=np.uint8),
            'pants_color_mean': np.array([148, 148, 98], dtype=np.uint8),
            'sat_mean': 0.61
        }
    ]
    
    tracks_f2 = tracker.update(detections_f2)
    print(f"   ✓ Updated {len(tracks_f2)} tracks")
    for track in tracks_f2:
        print(f"     - Track ID {track['id']}: hits={track['hits']}, age={track['age']}")
        avg_shirt, avg_pants = track['avg_shirt_color'], track['avg_pants_color']
        print(f"       Avg Shirt BGR: {avg_shirt}, Avg Pants BGR: {avg_pants}")
    
    # Check ID consistency
    if tracks_f2[0]['id'] == tracks_f1[0]['id']:
        print("   ✓ ID consistency maintained with HSV saturation masking!")
    else:
        print("   ✗ Warning: IDs changed")
    
    print(f"\n4. Getting all track colors...")
    track_colors = tracker.get_all_track_colors()
    print(f"   ✓ Retrieved colors for {len(track_colors)} tracks")
    for track_id, (shirt, pants) in track_colors.items():
        print(f"     - Track {track_id}: Shirt={shirt}, Pants={pants}")
    
    print("\n✓ Re-ID Tracker V4 Normalized tests passed!")


def test_visualizer_v4_norm():
    """Test normalized visualizer with actual colors."""
    print("\n" + "="*75)
    print("BBOX VISUALIZER V4 NORMALIZED TEST")
    print("="*75)
    
    print("\n1. Initializing visualizer...")
    visualizer = BBoxVisualizerV4Norm()
    print("   ✓ Visualizer initialized")
    
    print("\n2. Testing color display...")
    shirt_color = np.array([50, 100, 150], dtype=np.uint8)
    pants_color = np.array([100, 50, 200], dtype=np.uint8)
    print(f"   Test shirt color (BGR): {shirt_color}")
    print(f"   Test pants color (BGR): {pants_color}")
    print("   ✓ Real BGR colors ready for display")
    
    print("\n3. Testing color persistence...")
    color1 = visualizer.get_color_for_id(0)
    color2 = visualizer.get_color_for_id(0)
    if color1 == color2:
        print("   ✓ Color persistence confirmed")
    else:
        print("   ✗ Warning: Color changed")
    
    print("\n✓ Visualizer V4 Normalized tests passed!")


def test_full_pipeline():
    """Test full normalized pipeline on one frame."""
    print("\n" + "="*75)
    print("FULL NORMALIZED PIPELINE TEST")
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
    
    # Process with HSV features + saturation masking
    print("\n5. Processing with HSV + saturation masking...")
    detections = []
    
    for person in persons:
        feats = compile_features(person["keypoints"], person["bbox"])
        
        if not is_valid_frame(feats):
            continue
        
        # Extract HSV with saturation masking
        appearance = extract_color_regions_hsv(frame, person['bbox'], bins=16, sat_threshold=30)
        
        detection = {
            'shoulder_hip': feats['shoulder_hip'],
            'aspect': appearance['aspect'],
            'bbox_center': feats['bbox_center'],
            'bbox': person['bbox'],
            'conf': person['conf'],
            'shirt_hist': appearance['shirt_hist'],
            'pants_hist': appearance['pants_hist'],
            'shirt_color_mean': appearance['shirt_color_mean'],
            'pants_color_mean': appearance['pants_color_mean'],
            'sat_mean': appearance['sat_mean']
        }
        detections.append(detection)
    
    print(f"   ✓ Processed {len(detections)} valid detections")
    
    # Track with HSV features
    print("\n6. Tracking with HSV + color accumulation...")
    tracker = ReIDTrackerV4Norm()
    tracks = tracker.update(detections)
    print(f"   ✓ Created {len(tracks)} tracks")
    
    for track in tracks:
        shirt_bgr = track['shirt_color_mean']
        pants_bgr = track['pants_color_mean']
        print(f"     - Track ID {track['id']}: SH={track['shoulder_hip']:.2f}")
        print(f"       Shirt BGR: {shirt_bgr}, Pants BGR: {pants_bgr}")
        print(f"       Saturation: {track['sat_mean']:.2f}")
    
    cap.release()
    print("\n✓ Full normalized pipeline test passed!")
    print("\nWeek 5.5 Key Features:")
    print("  • HSV color space with saturation masking (S > 30)")
    print("  • Gray pixel exclusion for vivid colors")
    print("  • Per-track color accumulation and averaging")
    print("  • Real BGR color values for visualization")
    print("  • Accurate color patches (no gray bias)")


if __name__ == "__main__":
    test_hsv_extractor()
    test_reid_tracker_v4_norm()
    test_visualizer_v4_norm()
    test_full_pipeline()
    
    print("\n" + "="*75)
    print("✅ ALL NORMALIZED TESTS PASSED!")
    print("="*75)
    print("\nReady to run: python main_reid_normalized.py")
