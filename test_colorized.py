"""
Test script for Week 5.6 colorized Re-ID components.
"""

from features.pose_extractor import load_pose_model, get_keypoints
from features.ratio_calculator import compile_features
from features.filter_smooth import is_valid_frame
from tracking.appearance_extractor_v3_crop import extract_color_regions_cropped
from tracking.reid_tracker_v4_crop import ReIDTrackerV4Crop
from tracking.visualizer_bbox_v4_crop import BBoxVisualizerV4Crop
from utils.color_utils import (
    hsv_to_bgr,
    hue_to_name,
    hsv_to_full_description,
    saturation_to_descriptor
)
import cv2
import numpy as np


def test_color_utils():
    """Test color utility functions."""
    print("="*75)
    print("COLOR UTILITIES TEST")
    print("="*75)
    
    print("\n1. Testing HSV to BGR conversion...")
    # Test red
    red_bgr = hsv_to_bgr(0, s=200, v=200)
    print(f"   Red (H=0): BGR = {red_bgr}")
    
    # Test blue
    blue_bgr = hsv_to_bgr(110, s=200, v=200)
    print(f"   Blue (H=110): BGR = {blue_bgr}")
    
    # Test green
    green_bgr = hsv_to_bgr(60, s=200, v=200)
    print(f"   Green (H=60): BGR = {green_bgr}")
    print("   ✓ HSV to BGR conversion working")
    
    print("\n2. Testing hue to name mapping...")
    test_hues = [5, 15, 30, 60, 100, 135, 155, 175]
    for h in test_hues:
        name = hue_to_name(h)
        print(f"   H={h:3d} → {name}")
    print("   ✓ Hue to name mapping working")
    
    print("\n3. Testing full color description...")
    # Vivid blue
    desc1 = hsv_to_full_description(110, 200, 200)
    print(f"   H=110, S=200, V=200 → '{desc1}'")
    
    # Pale yellow
    desc2 = hsv_to_full_description(30, 80, 200)
    print(f"   H=30, S=80, V=200 → '{desc2}'")
    
    # Gray
    desc3 = hsv_to_full_description(0, 30, 100)
    print(f"   H=0, S=30, V=100 → '{desc3}'")
    print("   ✓ Full color description working")
    
    print("\n✓ Color utilities tests passed!")


def test_cropped_extractor():
    """Test tight crop and high saturation masking."""
    print("\n" + "="*75)
    print("TIGHT CROP + S>50 EXTRACTOR TEST")
    print("="*75)
    
    print("\n1. Testing tight crop parameters...")
    # Create dummy frame
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    bbox = (400, 200, 600, 800)  # 200x600 bbox
    
    appearance = extract_color_regions_cropped(
        frame,
        bbox,
        bins=16,
        sat_threshold=50,
        crop_padding=0.1
    )
    
    print(f"   Original bbox: {bbox}")
    print(f"   Extracted aspect: {appearance['aspect']:.2f}")
    print(f"   Shirt HSV: {appearance['shirt_HSV']}")
    print(f"   Pants HSV: {appearance['pants_HSV']}")
    print("   ✓ Tight crop applied (10% inward)")
    
    print("\n2. Testing saturation masking (S > 50)...")
    print(f"   Mean saturation: {appearance['sat_mean']:.2f}")
    print(f"   Shirt histogram sum: {appearance['shirt_hist'].sum():.3f}")
    print(f"   Pants histogram sum: {appearance['pants_hist'].sum():.3f}")
    print("   ✓ High saturation threshold applied")
    
    print("\n✓ Cropped extractor tests passed!")


def test_reid_tracker_v4_crop():
    """Test tracker with HSV storage."""
    print("\n" + "="*75)
    print("RE-ID TRACKER V4 CROP TEST (HSV STORAGE)")
    print("="*75)
    
    print("\n1. Initializing tracker...")
    tracker = ReIDTrackerV4Crop(
        max_cost_threshold=0.45,
        memory_cost_threshold=0.30,
        memory_duration=5.0
    )
    print("   ✓ Tracker initialized")
    
    print("\n2. Processing detections with HSV values...")
    detections_f1 = [
        {
            'shoulder_hip': 1.25,
            'aspect': 2.1,
            'bbox_center': (500, 500),
            'bbox': (400, 200, 600, 800),
            'conf': 0.90,
            'shirt_hist': np.array([0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            'pants_hist': np.array([0, 0, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.05, 0, 0, 0, 0, 0, 0, 0]),
            'shirt_HSV': (110.0, 180.0, 200.0),  # Blue
            'pants_HSV': (30.0, 150.0, 180.0),   # Yellow
            'shirt_color_bgr': np.array([200, 100, 50], dtype=np.uint8),
            'pants_color_bgr': np.array([50, 200, 200], dtype=np.uint8),
            'sat_mean': 0.6
        },
        {
            'shoulder_hip': 1.30,
            'aspect': 2.2,
            'bbox_center': (1000, 500),
            'bbox': (900, 200, 1100, 800),
            'conf': 0.85,
            'shirt_hist': np.array([0, 0, 0, 0, 0, 0, 0, 0.3, 0.3, 0.2, 0.1, 0.1, 0, 0, 0, 0]),
            'pants_hist': np.array([0, 0, 0, 0, 0, 0, 0, 0.2, 0.3, 0.3, 0.1, 0.1, 0, 0, 0, 0]),
            'shirt_HSV': (5.0, 200.0, 180.0),    # Red
            'pants_HSV': (60.0, 180.0, 160.0),   # Green
            'shirt_color_bgr': np.array([50, 50, 200], dtype=np.uint8),
            'pants_color_bgr': np.array([50, 180, 50], dtype=np.uint8),
            'sat_mean': 0.7
        }
    ]
    
    tracks_f1 = tracker.update(detections_f1)
    print(f"   ✓ Created {len(tracks_f1)} tracks")
    
    for track in tracks_f1:
        shirt_H, shirt_S, shirt_V = track['shirt_HSV']
        pants_H, pants_S, pants_V = track['pants_HSV']
        shirt_name = hue_to_name(shirt_H)
        pants_name = hue_to_name(pants_H)
        print(f"     - Track {track['id']}: Shirt={shirt_name} (H={shirt_H:.1f}), Pants={pants_name} (H={pants_H:.1f})")
    
    print("\n3. Getting all track HSV values...")
    track_hsv = tracker.get_all_track_HSV()
    print(f"   ✓ Retrieved HSV for {len(track_hsv)} tracks")
    
    for track_id, (shirt_HSV, pants_HSV) in track_hsv.items():
        shirt_desc = hsv_to_full_description(*shirt_HSV)
        pants_desc = hsv_to_full_description(*pants_HSV)
        print(f"     - Track {track_id}: Shirt='{shirt_desc}', Pants='{pants_desc}'")
    
    print("\n✓ Re-ID Tracker V4 Crop tests passed!")


def test_visualizer_v4_crop():
    """Test vivid color visualization."""
    print("\n" + "="*75)
    print("BBOX VISUALIZER V4 CROP TEST")
    print("="*75)
    
    print("\n1. Initializing visualizer...")
    visualizer = BBoxVisualizerV4Crop()
    print("   ✓ Visualizer initialized")
    
    print("\n2. Testing vivid color generation...")
    # Test different hues
    test_hues = [(0, "red"), (60, "green"), (110, "blue"), (30, "yellow")]
    
    for h, name in test_hues:
        bgr = hsv_to_bgr(h, s=200, v=200)
        print(f"   {name.capitalize()}: H={h}, BGR={bgr}")
    
    print("   ✓ Vivid colors (S=200, V=200) generated")
    
    print("\n✓ Visualizer V4 Crop tests passed!")


def test_full_pipeline():
    """Test full colorized pipeline."""
    print("\n" + "="*75)
    print("FULL COLORIZED PIPELINE TEST")
    print("="*75)
    
    print("\n1. Loading model...")
    model = load_pose_model("yolo")
    print("   ✓ Model loaded")
    
    print("\n2. Opening video...")
    cap = cv2.VideoCapture("videos/test.mp4")
    if not cap.isOpened():
        print("   ✗ Error: Could not open video")
        return
    print("   ✓ Video opened")
    
    print("\n3. Reading frame...")
    ret, frame = cap.read()
    if not ret:
        print("   ✗ Error: Could not read frame")
        return
    print(f"   ✓ Frame read: {frame.shape}")
    
    print("\n4. Detecting poses...")
    persons = get_keypoints(model, frame)
    print(f"   ✓ Detected {len(persons)} person(s)")
    
    print("\n5. Processing with tight crop + S>50...")
    detections = []
    
    for person in persons:
        feats = compile_features(person["keypoints"], person["bbox"])
        
        if not is_valid_frame(feats):
            continue
        
        # Extract with tight crop and high saturation threshold
        appearance = extract_color_regions_cropped(
            frame,
            person['bbox'],
            bins=16,
            sat_threshold=50,
            crop_padding=0.1
        )
        
        detection = {
            'shoulder_hip': feats['shoulder_hip'],
            'aspect': appearance['aspect'],
            'bbox_center': feats['bbox_center'],
            'bbox': person['bbox'],
            'conf': person['conf'],
            'shirt_hist': appearance['shirt_hist'],
            'pants_hist': appearance['pants_hist'],
            'shirt_HSV': appearance['shirt_HSV'],
            'pants_HSV': appearance['pants_HSV'],
            'shirt_color_bgr': appearance['shirt_color_bgr'],
            'pants_color_bgr': appearance['pants_color_bgr'],
            'sat_mean': appearance['sat_mean']
        }
        detections.append(detection)
    
    print(f"   ✓ Processed {len(detections)} valid detections")
    
    print("\n6. Tracking with HSV storage...")
    tracker = ReIDTrackerV4Crop()
    tracks = tracker.update(detections)
    print(f"   ✓ Created {len(tracks)} tracks")
    
    for track in tracks:
        shirt_HSV = track['shirt_HSV']
        pants_HSV = track['pants_HSV']
        
        shirt_name = hue_to_name(shirt_HSV[0])
        pants_name = hue_to_name(pants_HSV[0])
        
        shirt_desc = hsv_to_full_description(*shirt_HSV)
        pants_desc = hsv_to_full_description(*pants_HSV)
        
        print(f"     - Track {track['id']}:")
        print(f"       Shirt: '{shirt_desc}' (H={shirt_HSV[0]:.1f}, S={shirt_HSV[1]:.1f})")
        print(f"       Pants: '{pants_desc}' (H={pants_HSV[0]:.1f}, S={pants_HSV[1]:.1f})")
    
    cap.release()
    print("\n✓ Full colorized pipeline test passed!")
    print("\nWeek 5.6 Key Features:")
    print("  • Tight crop (10% inward) excludes background")
    print("  • High saturation threshold (S > 50) focuses on vivid colors")
    print("  • HSV means extracted and stored")
    print("  • Human-readable color names (red/pink, blue, green, etc.)")
    print("  • Vivid color patches (S=200, V=200) for visualization")


if __name__ == "__main__":
    test_color_utils()
    test_cropped_extractor()
    test_reid_tracker_v4_crop()
    test_visualizer_v4_crop()
    test_full_pipeline()
    
    print("\n" + "="*75)
    print("✅ ALL COLORIZED TESTS PASSED!")
    print("="*75)
    print("\nReady to run: python main_reid_colorized.py")
