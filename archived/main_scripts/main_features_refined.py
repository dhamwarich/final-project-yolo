"""
Week 2.5: Refined Feature Extraction Pipeline
Includes outlier filtering, temporal smoothing, and per-person overlay.
"""

from features.pose_extractor import load_pose_model, get_keypoints
from features.ratio_calculator import compile_features
from features.filter_smooth import is_valid_frame, smooth_features
from features.feature_logger import init_logger, log_features, summarize_log
from features.visualize_pose import draw_pose, overlay_metrics
import cv2
import time


def main():
    """Refined pipeline with filtering, smoothing, and improved visualization."""
    
    # Configuration
    VIDEO_PATH = "videos/test.mp4"
    CSV_PATH = "outputs/features_refined.csv"
    OUTPUT_VIDEO_PATH = "outputs/refined_overlay.mp4"
    SMOOTHING_ALPHA = 0.2  # Lower = more smoothing (0-1)
    
    print("="*60)
    print("REFINED FEATURE EXTRACTION PIPELINE")
    print("="*60)
    print("\nEnhancements:")
    print("  ✓ Outlier filtering (invalid frames rejected)")
    print("  ✓ Temporal smoothing (EMA with alpha={:.1f})".format(SMOOTHING_ALPHA))
    print("  ✓ Per-person metric overlay")
    print("  ✓ Summary statistics report")
    print("")
    
    # Load model
    print("Loading YOLOv8n-pose model...")
    model = load_pose_model("yolo")
    print("✓ Model loaded\n")
    
    # Initialize logger
    init_logger(CSV_PATH)
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
    
    # Per-person smoothing state (track by person ID)
    prev_feats_per_person = {}
    
    frame_id = 0
    fps_log = []
    valid_frames = 0
    invalid_frames = 0
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
        
        inference_end = time.time()
        inference_fps = 1 / (inference_end - inference_start)
        fps_log.append(inference_fps)
        
        # Process each detected person
        for person in persons:
            person_id = person['id']
            
            # Compile features
            feats = compile_features(person["keypoints"], person["bbox"])
            
            # Filter invalid frames
            if not is_valid_frame(feats):
                invalid_frames += 1
                continue
            
            valid_frames += 1
            
            # Apply temporal smoothing
            if person_id in prev_feats_per_person:
                feats = smooth_features(
                    feats, 
                    prev_feats_per_person[person_id],
                    alpha=SMOOTHING_ALPHA
                )
            
            # Update previous features for this person
            prev_feats_per_person[person_id] = feats
            
            # Log to CSV
            log_features(frame_id, feats, person["conf"], CSV_PATH)
            
            # Visualize - draw skeleton
            frame = draw_pose(frame, person["keypoints"])
            
            # Visualize - overlay metrics near person
            frame = overlay_metrics(frame, feats, feats["bbox_center"])
        
        # Add frame info overlay
        info_text = f"Frame: {frame_id}/{total_frames}  FPS: {inference_fps:.1f}  Valid: {valid_frames}  Invalid: {invalid_frames}"
        cv2.putText(
            frame,
            info_text,
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )
        
        # Write to output video
        out.write(frame)
        
        # Display
        cv2.imshow("Refined Pose Features", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nStopped by user")
            break
        
        frame_id += 1
        
        # Progress update every 30 frames
        if frame_id % 30 == 0:
            progress = (frame_id / total_frames) * 100
            avg_fps = sum(fps_log[-30:]) / min(30, len(fps_log))
            valid_pct = (valid_frames / max(1, valid_frames + invalid_frames)) * 100
            print(f"Progress: {progress:.1f}% | Avg FPS: {avg_fps:.1f} | Valid: {valid_pct:.1f}%")
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Calculate statistics
    total_time = time.time() - start_time
    avg_fps = sum(fps_log) / len(fps_log) if fps_log else 0
    total_detections = valid_frames + invalid_frames
    valid_pct = (valid_frames / max(1, total_detections)) * 100
    
    # Print processing summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Total frames processed: {frame_id}")
    print(f"Total detections: {total_detections}")
    print(f"  Valid: {valid_frames} ({valid_pct:.1f}%)")
    print(f"  Invalid (filtered): {invalid_frames} ({100-valid_pct:.1f}%)")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Features saved to: {CSV_PATH}")
    print(f"Annotated video saved to: {OUTPUT_VIDEO_PATH}")
    print("="*60)
    
    # Generate summary statistics
    print("\nGenerating summary statistics...")
    summarize_log(CSV_PATH)
    
    print("\n✅ Refined pipeline complete!")


if __name__ == "__main__":
    main()
