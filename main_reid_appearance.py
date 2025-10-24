"""
Week 4: Appearance-Enhanced Re-ID Tracking
Combines geometric + appearance + motion features for improved ID stability.
"""

from features.pose_extractor import load_pose_model, get_keypoints
from features.ratio_calculator import compile_features
from features.filter_smooth import is_valid_frame
from tracking.geometry_estimator import (
    calibrate_reference,
    estimate_height_m,
    DEFAULT_FOCAL_LENGTH_PX,
    DEFAULT_REF_RATIO,
    DEFAULT_K1
)
from tracking.appearance_extractor import extract_appearance_features
from tracking.reid_tracker_v2 import ReIDTrackerV2
from tracking.fusion_logger_v2 import (
    init_appearance_logger,
    log_appearance_features,
    summarize_appearance_reid
)
from tracking.visualizer_bbox_v2 import BBoxVisualizerV2, draw_appearance_overlay
import cv2
import time


def main():
    """Main appearance-enhanced Re-ID tracking pipeline."""
    
    # Configuration
    VIDEO_PATH = "videos/test.mp4"
    CSV_PATH = "outputs/reid_appearance.csv"
    OUTPUT_VIDEO_PATH = "outputs/reid_overlay_v2.mp4"
    
    # Calibration parameters
    FOCAL_LENGTH_PX = calibrate_reference(
        ref_height_m=1.70,
        ref_distance_m=2.0,
        ref_height_px=360
    )
    
    print("="*75)
    print("APPEARANCE-ENHANCED RE-ID TRACKING")
    print("="*75)
    print("\nWeek 4 Features:")
    print("  ✓ Geometric features (height, ratios)")
    print("  ✓ Appearance features (color, texture)")
    print("  ✓ Motion features (IoU, center proximity)")
    print("  ✓ 6-component cost function")
    print("  ✓ Enhanced EMA smoothing")
    print("  ✓ Color patch visualization")
    print(f"\nCalibration:")
    print(f"  Focal length: {FOCAL_LENGTH_PX:.1f} pixels")
    print(f"  Reference ratio: {DEFAULT_REF_RATIO}")
    print("")
    
    # Load model
    print("Loading YOLOv8n-pose model...")
    model = load_pose_model("yolo")
    print("✓ Model loaded\n")
    
    # Initialize logger
    init_appearance_logger(CSV_PATH)
    print(f"✓ Logger initialized: {CSV_PATH}\n")
    
    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"✗ Error: Could not open video {VIDEO_PATH}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))
    
    # Initialize tracker and visualizer
    tracker = ReIDTrackerV2(
        max_cost_threshold=0.45,  # Slightly higher threshold with more features
        img_width=width,
        img_height=height
    )
    visualizer = BBoxVisualizerV2()
    
    frame_id = 0
    fps_log = []
    valid_detections = 0
    invalid_detections = 0
    start_time = time.time()
    
    print("\nProcessing video...")
    print("Press 'q' to quit early\n")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Time the inference
        inference_start = time.time()
        
        # Get pose keypoints
        persons = get_keypoints(model, frame)
        
        # Process each detection
        detections_with_features = []
        bboxes = {}
        
        for person in persons:
            # Compile geometric features
            feats = compile_features(person["keypoints"], person["bbox"])
            
            # Filter invalid detections
            if not is_valid_frame(feats):
                invalid_detections += 1
                continue
            
            valid_detections += 1
            
            # Estimate real-world height
            height_m = estimate_height_m(
                feats['height_px'],
                feats['torso_leg'],
                FOCAL_LENGTH_PX,
                DEFAULT_REF_RATIO,
                DEFAULT_K1
            )
            
            # Extract appearance features
            appearance = extract_appearance_features(frame, person['bbox'])
            
            # Create detection dict with all features
            detection = {
                'height_px': feats['height_px'],
                'height_m': height_m,
                'shoulder_hip': feats['shoulder_hip'],
                'torso_leg': feats['torso_leg'],
                'bbox_center': feats['bbox_center'],
                'bbox': person['bbox'],
                'conf': person['conf'],
                # Appearance features
                'top_h': appearance['top_h'],
                'bot_h': appearance['bot_h'],
                'texture': appearance['texture']
            }
            
            detections_with_features.append(detection)
        
        # Update tracker with appearance features
        tracks = tracker.update(detections_with_features)
        
        # Log and visualize tracks
        for i, track in enumerate(tracks):
            if i < len(detections_with_features):
                det = detections_with_features[i]
                
                # Log to CSV
                log_appearance_features(
                    frame_id,
                    track,
                    det['height_px'],
                    CSV_PATH
                )
                
                # Store bbox for visualization
                bboxes[track['id']] = det['bbox']
        
        # Draw appearance-enhanced visualization
        frame = draw_appearance_overlay(frame, tracks, bboxes, visualizer)
        
        inference_end = time.time()
        inference_fps = 1 / (inference_end - inference_start)
        fps_log.append(inference_fps)
        
        # Add frame info overlay
        info_text = f"Frame: {frame_id}/{total_frames}  FPS: {inference_fps:.1f}  Tracks: {len(tracks)}"
        cv2.putText(
            frame,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        stats_text = f"Valid: {valid_detections}  Invalid: {invalid_detections}"
        cv2.putText(
            frame,
            stats_text,
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
        
        # Write to output video
        out.write(frame)
        
        # Display
        cv2.imshow("Appearance-Enhanced Re-ID Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nStopped by user")
            break
        
        frame_id += 1
        
        # Progress update every 30 frames
        if frame_id % 30 == 0:
            progress = (frame_id / total_frames) * 100
            avg_fps = sum(fps_log[-30:]) / min(30, len(fps_log))
            print(f"Progress: {progress:.1f}% | Avg FPS: {avg_fps:.1f} | Tracks: {len(tracks)}")
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Calculate statistics
    total_time = time.time() - start_time
    avg_fps = sum(fps_log) / len(fps_log) if fps_log else 0
    total_detections = valid_detections + invalid_detections
    valid_pct = (valid_detections / max(1, total_detections)) * 100
    
    # Print processing summary
    print("\n" + "="*75)
    print("PROCESSING COMPLETE")
    print("="*75)
    print(f"Total frames processed: {frame_id}")
    print(f"Total detections: {total_detections}")
    print(f"  Valid: {valid_detections} ({valid_pct:.1f}%)")
    print(f"  Invalid (filtered): {invalid_detections} ({100-valid_pct:.1f}%)")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Unique track IDs: {tracker.next_id}")
    print(f"Active tracks at end: {tracker.get_track_count()}")
    print(f"Features saved to: {CSV_PATH}")
    print(f"Annotated video saved to: {OUTPUT_VIDEO_PATH}")
    print("="*75)
    
    # Generate Re-ID summary
    print("\nGenerating appearance-enhanced Re-ID summary...")
    summarize_appearance_reid(CSV_PATH)
    
    print("\n✅ Appearance-enhanced Re-ID tracking complete!")
    print("\nImprovements over Week 3:")
    print("  • 6-component cost function (vs 3)")
    print("  • Motion-assisted matching (IoU + proximity)")
    print("  • Color & texture features for visual identity")
    print("  • Enhanced EMA smoothing for stability")
    print("  • Color patch visualization for debugging")


if __name__ == "__main__":
    main()
