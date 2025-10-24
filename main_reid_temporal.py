"""
Week 5.9: Temporal Re-ID with Motion-Dominant Matching
Focus on geometry + motion for stable tracking, with color as weak contextual cue.
"""

from features.pose_extractor import load_pose_model, get_keypoints
from features.ratio_calculator import compile_features
from features.filter_smooth import is_valid_frame
from tracking.appearance_extractor_v3_robust import extract_color_robust
from tracking.reid_tracker_v5_temporal import ReIDTrackerV5Temporal
from tracking.fusion_logger_v5_temporal import (
    init_temporal_logger,
    log_temporal_features,
    log_track_temporal_summary,
    summarize_temporal_reid
)
from tracking.visualizer_bbox_v5_temporal import BBoxVisualizerV5Temporal, draw_temporal_overlay
import cv2
import time
from collections import defaultdict


def main():
    """Main temporal Re-ID tracking pipeline with motion-dominant matching."""
    
    # Configuration
    VIDEO_PATH = "videos/test.mp4"
    CSV_PATH = "outputs/reid_temporal.csv"
    OUTPUT_VIDEO_PATH = "outputs/reid_overlay_temporal.mp4"
    
    print("="*75)
    print("TEMPORAL RE-ID TRACKING (MOTION-DOMINANT MATCHING)")
    print("="*75)
    print("\nWeek 5.9 Features:")
    print("  ✓ Feature memory bank (geometry, motion, color with EMA)")
    print("  ✓ Motion-dominant matching (0.6*geom + 0.3*motion + 0.1*color)")
    print("  ✓ Motion similarity (exponential decay on velocity diff)")
    print("  ✓ Occlusion handling (max_age=10, ~0.3s buffer)")
    print("  ✓ Color confidence gating (ignore if conf<0.4)")
    print("  ✓ Long-term stable colors (15-frame EMA for visualization)")
    print("  ✓ Temporal coherence metrics (geometry std, motion std)")
    print("  ✓ Target: ID retention ≥85%, switches ≤2")
    print("")
    
    # Load model
    print("Loading YOLOv8n-pose model...")
    model = load_pose_model("yolo")
    print("✓ Model loaded\n")
    
    # Initialize logger
    init_temporal_logger(CSV_PATH)
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
    tracker = ReIDTrackerV5Temporal(
        max_similarity_threshold=0.40,
        memory_similarity_threshold=0.30,
        memory_duration=5.0,
        color_conf_threshold=0.4
    )
    visualizer = BBoxVisualizerV5Temporal()
    
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
            
            # Extract color features (for memory only, low weight in matching)
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
                # Color for memory and visualization (minimal weight)
                'shirt_HSV': appearance['shirt_HSV'],
                'pants_HSV': appearance['pants_HSV'],
                'shirt_confidence': appearance['shirt_confidence'],
                'pants_confidence': appearance['pants_confidence'],
                'shirt_color_bgr': appearance['shirt_color_bgr'],
                'pants_color_bgr': appearance['pants_color_bgr']
            }
            
            detections_with_features.append(detection)
        
        # Update tracker (motion-dominant matching)
        tracks = tracker.update(detections_with_features)
        
        # Log and visualize tracks
        for i, track in enumerate(tracks):
            if i < len(detections_with_features):
                det = detections_with_features[i]
                
                # Log to CSV
                log_temporal_features(
                    frame_id,
                    track,
                    CSV_PATH
                )
                
                # Count frames
                track_frame_counts[track['id']] += 1
                
                # Store bbox
                bboxes[track['id']] = det['bbox']
        
        # Draw temporal visualization with stable colors
        frame = draw_temporal_overlay(frame, tracks, bboxes, visualizer)
        
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
        cv2.imshow("Temporal Re-ID (Motion-Dominant)", frame)
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
    
    # Log per-ID temporal statistics
    print("\nLogging per-ID temporal statistics...")
    track_stats = tracker.get_all_track_stats()
    log_track_temporal_summary(track_stats, track_frame_counts, CSV_PATH)
    
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
    print("\nGenerating temporal Re-ID summary...")
    summarize_temporal_reid(CSV_PATH)
    
    print("\n✅ Temporal Re-ID tracking complete!")
    print("\nWeek 5.9 Improvements:")
    print("  • Motion-dominant matching (geometry + motion, minimal color)")
    print("  • Feature memory bank (EMA for all features)")
    print("  • Temporal coherence focus (smooth trajectories)")
    print("  • Occlusion handling (10-frame buffer)")
    print("  • Stable visualization colors (15-frame EMA)")
    print("  • Color confidence gating (ignore if <0.4)")
    print("  • Target: 85% retention, ≤2 ID switches")
    print("  • ROBUST TEMPORAL TRACKING!")


if __name__ == "__main__":
    main()
