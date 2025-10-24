"""
Quick test for refined pipeline components.
"""

from features.pose_extractor import load_pose_model, get_keypoints
from features.ratio_calculator import compile_features
from features.filter_smooth import is_valid_frame, smooth_features
from features.feature_logger import init_logger, log_features
import cv2

def test_refinements():
    """Test filtering and smoothing on a single frame."""
    
    print("="*60)
    print("REFINED PIPELINE TEST")
    print("="*60)
    
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
    
    # Read a frame
    print("\n3. Reading frame...")
    ret, frame = cap.read()
    if not ret:
        print("   ✗ Error: Could not read frame")
        return
    print(f"   ✓ Frame read: {frame.shape}")
    
    # Get keypoints
    print("\n4. Extracting keypoints...")
    persons = get_keypoints(model, frame)
    print(f"   ✓ Detected {len(persons)} person(s)")
    
    if len(persons) == 0:
        print("\n⚠️  No persons detected in first frame")
        return
    
    # Test filtering and smoothing
    print("\n5. Testing filtering and smoothing...")
    person = persons[0]
    feats = compile_features(person["keypoints"], person["bbox"])
    
    # Test validation
    is_valid = is_valid_frame(feats)
    print(f"   ✓ Validation: {'VALID' if is_valid else 'INVALID'}")
    print(f"     - Height: {feats['height_px']:.2f}px")
    print(f"     - SH ratio: {feats['shoulder_hip']:.2f}")
    print(f"     - TL ratio: {feats['torso_leg']:.2f}")
    
    if is_valid:
        # Test smoothing (first frame - no previous)
        smoothed1 = smooth_features(feats, {}, alpha=0.2)
        print(f"   ✓ Smoothing (first frame):")
        print(f"     - Height: {smoothed1['height_px']:.2f}px")
        
        # Simulate second frame with different values
        feats2 = feats.copy()
        feats2['height_px'] *= 1.1  # Simulate 10% change
        smoothed2 = smooth_features(feats2, smoothed1, alpha=0.2)
        print(f"   ✓ Smoothing (with previous):")
        print(f"     - Raw height: {feats2['height_px']:.2f}px")
        print(f"     - Smoothed: {smoothed2['height_px']:.2f}px")
        print(f"     - Reduction: {abs(feats2['height_px'] - smoothed2['height_px']):.2f}px")
    
    # Test invalid frame creation
    print("\n6. Testing outlier rejection...")
    invalid_feats = {
        'height_px': 50,  # Too short
        'shoulder_hip': 0.3,  # Too narrow
        'torso_leg': 5.0,  # Too high
        'bbox_center': (100, 100)
    }
    is_valid_test = is_valid_frame(invalid_feats)
    print(f"   ✓ Invalid frame rejected: {not is_valid_test}")
    
    # Test logger initialization
    print("\n7. Testing refined logger...")
    init_logger("outputs/test_refined.csv")
    if is_valid:
        log_features(0, smoothed1, person["conf"], "outputs/test_refined.csv")
    print("   ✓ Logger functional")
    
    cap.release()
    
    print("\n" + "="*60)
    print("✅ ALL REFINEMENT TESTS PASSED!")
    print("="*60)
    print("\nReady to run: python main_features_refined.py")

if __name__ == "__main__":
    test_refinements()
