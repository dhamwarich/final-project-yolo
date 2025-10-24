"""
Week 3: Re-ID Tracking with Geometry-based Height/Distance Estimation
Integrate pose detection, Re-ID tracking, and approximate height/distance estimation.
"""

from features.pose_extractor import load_pose_model, get_keypoints
from features.ratio_calculator import compile_features
from features.filter_smooth import is_valid_frame
from tracking.geometry_estimator import (
    calibrate_reference,
    estimate_height_m,
    estimate_distance_m,
    DEFAULT_FOCAL_LENGTH_PX,
    DEFAULT_REF_RATIO,
    DEFAULT_K1
)
from tracking.reid_tracker import ReIDTracker
from tracking.fusion_logger import init_fusion_logger, log_tracked_features, summarize_reid
from tracking.visualizer_bbox import BBoxVisualizer, draw_tracking_overlay
import cv2
import time


def main():
    """Main Re-ID tracking pipeline with geometry estimation."""
    
    # Configuration
    VIDEO_PATH = "videos/test.mp4"
    CSV_PATH = "outputs/fused_features_sim.csv"
    OUTPUT_VIDEO_PATH = "outputs/reid_overlay.mp4"
    
    # Calibration parameters (adjust based on your camera/scene)
    # Example: 1.70m person at 2.0m distance appears as 360px
    FOCAL_LENGTH_PX = calibrate_reference(
        ref_height_m=1.70,
        ref_distance_m=2.0,
        ref_height_px=360
    )
    
    print("="*70)
    print("RE-ID TRACKING WITH GEOMETRY ESTIMATION")
    print("="*70)
    print("\nFeatures:")
    print("  ✓ Identity-consistent tracking (Re-ID)")
    print("  ✓ Approximate height estimation (meters)")
    print("  ✓ Approximate distance estimation (meters)")
    print("  ✓ Hungarian matching with geometric features")
    print("  ✓ Color-coded bounding boxes per ID")
    print(f"\nCalibration:")
    print(f"  Focal length: {FOCAL_LENGTH_PX:.1f} pixels")
    print(f"  Reference ratio: {DEFAULT_REF_RATIO}")
    print("")
    
    # Load model
    print("Loading YOLOv8n-pose model...")
    model = load_pose_model("yolo")
    print("✓ Model loaded\n")
    
    # Initialize logger
    init_fusion_logger(CSV_PATH)
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
    tracker = ReIDTracker(
        max_cost_threshold=0.35,
        img_width=width,
        img_height=height
    )
    visualizer = BBoxVisualizer()
    
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
            
            # Create detection dict
            detection = {
                'height_px': feats['height_px'],
                'height_m': height_m,
                'shoulder_hip': feats['shoulder_hip'],
                'torso_leg': feats['torso_leg'],
                'bbox_center': feats['bbox_center'],
                'bbox': person['bbox'],
                'conf': person['conf']
            }
            
            detections_with_features.append(detection)
        
        # Update tracker
        tracks = tracker.update(detections_with_features)
        
        # Log and visualize tracks
        for i, track in enumerate(tracks):
            if i < len(detections_with_features):
                det = detections_with_features[i]
                
                # Estimate distance
                dist_est = estimate_distance_m(
                    det['height_px'],
                    track['height_m'],
                    FOCAL_LENGTH_PX
                )
                
                # Log to CSV
                log_tracked_features(
                    frame_id,
                    track,
                    det['height_px'],
                    dist_est,
                    CSV_PATH
                )
                
                # Store bbox with distance for visualization
                x1, y1, x2, y2 = det['bbox']
                bboxes[track['id']] = (x1, y1, x2, y2, dist_est)
        
        # Draw tracking visualization
        frame = draw_tracking_overlay(frame, tracks, bboxes, visualizer)
        
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
        cv2.imshow("Re-ID Tracking with Geometry", frame)
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
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
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
    print("="*70)
    
    # Generate Re-ID summary
    print("\nGenerating Re-ID tracking summary...")
    summarize_reid(CSV_PATH)
    
    print("\n✅ Re-ID tracking pipeline complete!")


if __name__ == "__main__":
    main()
