"""
Week 5: Refined Re-ID with LAB Color-Region Separation
Shirt/pants histograms in LAB space, no torso-leg ratio.
"""

from features.pose_extractor import load_pose_model, get_keypoints
from features.ratio_calculator import compile_features
from features.filter_smooth import is_valid_frame
from tracking.appearance_extractor_v3 import extract_lab_histograms
from tracking.reid_tracker_v4 import ReIDTrackerV4
from tracking.fusion_logger_v4 import (
    init_refined_logger,
    log_refined_features,
    summarize_refined_reid
)
from tracking.visualizer_bbox_v4 import BBoxVisualizerV4, draw_refined_overlay
import cv2
import time


def main():
    """Main refined Re-ID tracking pipeline with LAB color-region."""
    
    # Configuration
    VIDEO_PATH = "videos/test.mp4"
    CSV_PATH = "outputs/reid_refined.csv"
    OUTPUT_VIDEO_PATH = "outputs/reid_overlay_refined.mp4"
    
    print("="*75)
    print("REFINED RE-ID TRACKING (LAB COLOR-REGION)")
    print("="*75)
    print("\nWeek 5 Features:")
    print("  ✓ LAB color space (perceptually uniform)")
    print("  ✓ Separate shirt/pants histograms (16 bins each)")
    print("  ✓ Shoulder/hip ratio + aspect ratio")
    print("  ✓ Motion features (IoU + center)")
    print("  ✓ Short-term memory (5s window)")
    print("  ✗ NO torso-leg ratio (unstable)")
    print("  ✗ NO pseudo-height")
    print("")
    
    # Load model
    print("Loading YOLOv8n-pose model...")
    model = load_pose_model("yolo")
    print("✓ Model loaded\n")
    
    # Initialize logger
    init_refined_logger(CSV_PATH)
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
    tracker = ReIDTrackerV4(
        max_cost_threshold=0.45,
        memory_cost_threshold=0.30,
        memory_duration=5.0,
        img_width=width,
        img_height=height
    )
    visualizer = BBoxVisualizerV4()
    
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
            
            # Extract LAB histogram-based appearance features
            appearance = extract_lab_histograms(frame, person['bbox'], bins=16)
            
            # Create detection dict (NO torso_leg, NO height)
            detection = {
                'shoulder_hip': feats['shoulder_hip'],
                'aspect': appearance['aspect'],
                'bbox_center': feats['bbox_center'],
                'bbox': person['bbox'],
                'conf': person['conf'],
                # LAB histogram features
                'shirt_hist': appearance['shirt_hist'],
                'pants_hist': appearance['pants_hist'],
                'sat_mean': appearance['sat_mean']
            }
            
            detections_with_features.append(detection)
        
        # Update tracker with LAB features
        tracks = tracker.update(detections_with_features)
        
        # Log and visualize tracks
        for i, track in enumerate(tracks):
            if i < len(detections_with_features):
                det = detections_with_features[i]
                
                # Log to CSV
                log_refined_features(
                    frame_id,
                    track,
                    CSV_PATH
                )
                
                # Store bbox for visualization
                bboxes[track['id']] = det['bbox']
        
        # Draw refined visualization with shirt/pants patches
        frame = draw_refined_overlay(frame, tracks, bboxes, visualizer)
        
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
        
        memory_text = f"Memory: {tracker.get_memory_count()} | IDs created: {tracker.next_id}"
        cv2.putText(
            frame,
            memory_text,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
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
        cv2.imshow("Refined Re-ID (LAB Color-Region)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nStopped by user")
            break
        
        frame_id += 1
        
        # Progress update every 30 frames
        if frame_id % 30 == 0:
            progress = (frame_id / total_frames) * 100
            avg_fps = sum(fps_log[-30:]) / min(30, len(fps_log))
            print(f"Progress: {progress:.1f}% | Avg FPS: {avg_fps:.1f} | Tracks: {len(tracks)} | Memory: {tracker.get_memory_count()}")
    
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
    print(f"Unique track IDs created: {tracker.next_id}")
    print(f"Active tracks at end: {tracker.get_track_count()}")
    print(f"Tracks in memory at end: {tracker.get_memory_count()}")
    print(f"Features saved to: {CSV_PATH}")
    print(f"Annotated video saved to: {OUTPUT_VIDEO_PATH}")
    print("="*75)
    
    # Generate Re-ID summary
    print("\nGenerating refined Re-ID summary...")
    summarize_refined_reid(CSV_PATH)
    
    print("\n✅ Refined Re-ID tracking complete!")
    print("\nWeek 5 Improvements:")
    print("  • LAB color space (perceptually uniform)")
    print("  • Separate shirt/pants histograms (70/30 weighting)")
    print("  • Removed unstable torso-leg ratio")
    print("  • Rebalanced cost function weights")
    print("  • 5-frame EMA smoothing for stability")
    print("  • Visual debugging with shirt/pants patches")


if __name__ == "__main__":
    main()
