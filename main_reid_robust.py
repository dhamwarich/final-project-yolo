"""
Week 5.8: Robust Re-ID with Expanded Regions, SV Gating & Confidence Weighting
Improves shirt color stability under pose and lighting variation.
"""

from features.pose_extractor import load_pose_model, get_keypoints
from features.ratio_calculator import compile_features
from features.filter_smooth import is_valid_frame
from tracking.appearance_extractor_v3_robust import extract_color_robust
from tracking.reid_tracker_v4_robust import ReIDTrackerV4Robust
from tracking.fusion_logger_v4_robust import (
    init_robust_logger,
    log_robust_features,
    log_track_colors_with_confidence,
    summarize_robust_reid
)
from tracking.visualizer_bbox_v4_robust import BBoxVisualizerV4Robust, draw_robust_overlay
import cv2
import time
from collections import defaultdict


def main():
    """Main robust Re-ID tracking pipeline with confidence weighting."""
    
    # Configuration
    VIDEO_PATH = "videos/test.mp4"
    CSV_PATH = "outputs/reid_robust.csv"
    OUTPUT_VIDEO_PATH = "outputs/reid_overlay_robust.mp4"
    
    print("="*75)
    print("ROBUST RE-ID TRACKING (EXPANDED REGIONS + SV GATING + CONFIDENCE)")
    print("="*75)
    print("\nWeek 5.8 Features:")
    print("  ✓ Expanded shirt region (20-60% height, 25-75% width, 40%×50% area)")
    print("  ✓ Saturation/brightness gating (S>60, V∈[40,220])")
    print("  ✓ Adaptive temporal smoothing (α=0.3→0.5 when σ>10°)")
    print("  ✓ Confidence weighting in cost matrix (based on saturation)")
    print("  ✓ Reduced color weight (0.15 shirt + 0.10 pants = 0.25 total)")
    print("  ✓ Color confidence logging (quantifies color quality)")
    print("  ✓ Target: σ≤5°, S≥70, conf≥0.55")
    print("")
    
    # Load model
    print("Loading YOLOv8n-pose model...")
    model = load_pose_model("yolo")
    print("✓ Model loaded\n")
    
    # Initialize logger
    init_robust_logger(CSV_PATH)
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
    tracker = ReIDTrackerV4Robust(
        max_cost_threshold=0.45,
        memory_cost_threshold=0.30,
        memory_duration=5.0,
        img_width=width,
        img_height=height
    )
    visualizer = BBoxVisualizerV4Robust()
    
    frame_id = 0
    fps_log = []
    valid_detections = 0
    invalid_detections = 0
    track_frame_counts = defaultdict(int)
    start_time = time.time()
    
    print("\nProcessing video...")
    print("Press 'q' to quit early\n")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
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
            
            # Extract color features with expanded regions and SV gating
            appearance = extract_color_robust(
                frame,
                person['bbox'],
                bins=16,
                sat_threshold=60,
                val_min=40,
                val_max=220
            )
            
            # Create detection dict
            detection = {
                'shoulder_hip': feats['shoulder_hip'],
                'aspect': appearance['aspect'],
                'bbox_center': feats['bbox_center'],
                'bbox': person['bbox'],
                'conf': person['conf'],
                # Histograms for matching
                'shirt_hist': appearance['shirt_hist'],
                'pants_hist': appearance['pants_hist'],
                # HSV for color smoothing
                'shirt_HSV': appearance['shirt_HSV'],
                'pants_HSV': appearance['pants_HSV'],
                # Confidence for weighting
                'shirt_confidence': appearance['shirt_confidence'],
                'pants_confidence': appearance['pants_confidence'],
                # BGR for visualization
                'shirt_color_bgr': appearance['shirt_color_bgr'],
                'pants_color_bgr': appearance['pants_color_bgr'],
                'sat_mean': appearance['sat_mean']
            }
            
            detections_with_features.append(detection)
        
        # Update tracker (with adaptive smoothing and confidence weighting)
        tracks = tracker.update(detections_with_features)
        
        # Log and visualize tracks
        for i, track in enumerate(tracks):
            if i < len(detections_with_features):
                det = detections_with_features[i]
                
                # Log to CSV
                log_robust_features(
                    frame_id,
                    track,
                    CSV_PATH
                )
                
                # Count frames
                track_frame_counts[track['id']] += 1
                
                # Store bbox (smoothed)
                bboxes[track['id']] = det['bbox']
        
        # Draw robust visualization with confidence indicators
        frame = draw_robust_overlay(frame, tracks, bboxes, visualizer)
        
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
        
        memory_text = f"Memory: {tracker.get_memory_count()} | IDs: {tracker.next_id}"
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
        cv2.imshow("Robust Re-ID (Confidence Weighted)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nStopped by user")
            break
        
        frame_id += 1
        
        # Progress update every 30 frames
        if frame_id % 30 == 0:
            progress = (frame_id / total_frames) * 100
            avg_fps = sum(fps_log[-30:]) / min(30, len(fps_log))
            print(f"Progress: {progress:.1f}% | FPS: {avg_fps:.1f} | Tracks: {len(tracks)} | Memory: {tracker.get_memory_count()}")
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Calculate statistics
    total_time = time.time() - start_time
    avg_fps = sum(fps_log) / len(fps_log) if fps_log else 0
    total_detections = valid_detections + invalid_detections
    valid_pct = (valid_detections / max(1, total_detections)) * 100
    
    # Log per-ID HSV values with variance and confidence
    print("\nLogging per-ID colors with confidence...")
    track_hsv_data = tracker.get_all_track_HSV_with_confidence()
    log_track_colors_with_confidence(track_hsv_data, track_frame_counts, CSV_PATH)
    
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
    print("\nGenerating robust Re-ID summary...")
    summarize_robust_reid(CSV_PATH)
    
    print("\n✅ Robust Re-ID tracking complete!")
    print("\nWeek 5.8 Improvements:")
    print("  • Expanded shirt region (40%×50% vs 30%×20% center patch)")
    print("  • Saturation/brightness gating (S>60, V∈[40,220])")
    print("  • Adaptive temporal smoothing (α increases with variance)")
    print("  • Confidence weighting in matching (low-sat colors downweighted)")
    print("  • Color confidence metrics (quantifies color quality)")
    print("  • Target: σ≤5°, S≥70, conf≥0.55")
    print("  • IMPROVED SHIRT COLOR STABILITY!")


if __name__ == "__main__":
    main()
