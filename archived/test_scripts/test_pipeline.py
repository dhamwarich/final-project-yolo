"""
Quick test script to validate the pipeline without GUI.
"""

from features.pose_extractor import load_pose_model, get_keypoints
from features.ratio_calculator import compile_features
from features.feature_logger import init_logger, log_features
import cv2

def test_pipeline():
    """Test the pipeline on a single frame."""
    
    print("="*50)
    print("PIPELINE TEST")
    print("="*50)
    
    # Load model
    print("\n1. Loading YOLOv8n-pose model...")
    model = load_pose_model("yolo")
    print("   ✓ Model loaded successfully")
    
    # Open video
    print("\n2. Opening video...")
    cap = cv2.VideoCapture("videos/test_scene.MOV")
    if not cap.isOpened():
        print("   ✗ Error: Could not open video")
        return
    print("   ✓ Video opened successfully")
    
    # Read one frame
    print("\n3. Reading test frame...")
    ret, frame = cap.read()
    if not ret:
        print("   ✗ Error: Could not read frame")
        return
    print(f"   ✓ Frame read: {frame.shape}")
    
    # Get keypoints
    print("\n4. Extracting keypoints...")
    persons = get_keypoints(model, frame)
    print(f"   ✓ Detected {len(persons)} person(s)")
    
    if len(persons) > 0:
        person = persons[0]
        print(f"   - Confidence: {person['conf']:.2f}")
        print(f"   - Bbox: {person['bbox']}")
        print(f"   - Keypoints: {len(person['keypoints'])} detected")
        
        # Calculate features
        print("\n5. Calculating features...")
        feats = compile_features(person["keypoints"], person["bbox"])
        print(f"   ✓ Height: {feats['height_px']:.2f}px")
        print(f"   ✓ Shoulder/Hip ratio: {feats['shoulder_hip']:.2f}")
        print(f"   ✓ Torso/Leg ratio: {feats['torso_leg']:.2f}")
        
        # Test logger
        print("\n6. Testing CSV logger...")
        init_logger("outputs/test_features.csv")
        log_features(0, feats, person["conf"], "outputs/test_features.csv")
        print("   ✓ CSV logging works")
        
        print("\n" + "="*50)
        print("✅ ALL TESTS PASSED!")
        print("="*50)
        print("\nYou can now run: python main_features.py")
    else:
        print("\n⚠️  No persons detected in test frame")
        print("This might be normal if the first frame has no people.")
        print("Try running: python main_features.py")
    
    cap.release()

if __name__ == "__main__":
    test_pipeline()
