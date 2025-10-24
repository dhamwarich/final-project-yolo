"""
Week 5.6: Colorized Re-ID with Tight Crop, S>50 Masking & Color Names
Vivid color patches and human-readable color logging.
"""

from features.pose_extractor import load_pose_model, get_keypoints
from features.ratio_calculator import compile_features
from features.filter_smooth import is_valid_frame
from tracking.appearance_extractor_v3_crop import extract_color_regions_cropped
from tracking.reid_tracker_v4_crop import ReIDTrackerV4Crop
from tracking.fusion_logger_v4_crop import (
    init_colorized_logger,
    log_colorized_features,
    log_track_colors_with_names,
    summarize_colorized_reid
)
from tracking.visualizer_bbox_v4_crop import BBoxVisualizerV4Crop, draw_colorized_overlay
import cv2
import time
from collections import defaultdict


def main():
    """Main colorized Re-ID tracking pipeline with vivid colors and names."""
    
    # Configuration
    VIDEO_PATH = "videos/test.mp4"
    CSV_PATH = "outputs/reid_colorized.csv"
    OUTPUT_VIDEO_PATH = "outputs/reid_overlay_colorized.mp4"
    
    print("="*75)
    print("COLORIZED RE-ID TRACKING (TIGHT CROP + S>50 + COLOR NAMES)")
    print("="*75)
    print("\nWeek 5.6 Features:")
    print("  ✓ Tight crop (10% inward padding) to exclude background")
    print("  ✓ High saturation threshold (S > 50) for vivid colors")
    print("  ✓ HSV mean extraction with masking")
    print("  ✓ Human-readable color names (red/pink, blue, green, etc.)")
    print("  ✓ Vivid color patches (S=200, V=200) for visibility")
    print("  ✓ Per-ID color CSV with HSV values and names")
    print("  ✗ NO torso-leg ratio")
    print("  ✗ NO pseudo-height")
    print("")
    
    # Load model
    print("Loading YOLOv8n-pose model...")
    model = load_pose_model("yolo")
    print("✓ Model loaded\n")
    
    # Initialize logger
    init_colorized_logger(CSV_PATH)
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
    tracker = ReIDTrackerV4Crop(
        max_cost_threshold=0.45,
        memory_cost_threshold=0.30,
        memory_duration=5.0,
        img_width=width,
        img_height=height
    )
    visualizer = BBoxVisualizerV4Crop()
    
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
            
            # Extract color features with tight crop and S>50 mask
            appearance = extract_color_regions_cropped(
                frame,
                person['bbox'],
                bins=16,
                sat_threshold=50,
                crop_padding=0.1
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
                # HSV for color naming
                'shirt_HSV': appearance['shirt_HSV'],
                'pants_HSV': appearance['pants_HSV'],
                # BGR for visualization
                'shirt_color_bgr': appearance['shirt_color_bgr'],
                'pants_color_bgr': appearance['pants_color_bgr'],
                'sat_mean': appearance['sat_mean']
            }
            
            detections_with_features.append(detection)
        
        # Update tracker
        tracks = tracker.update(detections_with_features)
        
        # Log and visualize tracks
        for i, track in enumerate(tracks):
            if i < len(detections_with_features):
                det = detections_with_features[i]
                
                # Log to CSV
                log_colorized_features(
                    frame_id,
                    track,
                    CSV_PATH
                )
                
                # Count frames
                track_frame_counts[track['id']] += 1
                
                # Store bbox
                bboxes[track['id']] = det['bbox']
        
        # Draw colorized visualization with vivid colors
        frame = draw_colorized_overlay(frame, tracks, bboxes, visualizer)
        
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
        cv2.imshow("Colorized Re-ID (Vivid Colors)", frame)
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
    
    # Log per-ID HSV values and color names
    print("\nLogging per-ID colors with names...")
    track_hsv = tracker.get_all_track_HSV()
    log_track_colors_with_names(track_hsv, track_frame_counts, CSV_PATH)
    
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
    print("\nGenerating colorized Re-ID summary...")
    summarize_colorized_reid(CSV_PATH)
    
    print("\n✅ Colorized Re-ID tracking complete!")
    print("\nWeek 5.6 Improvements:")
    print("  • Tight crop (10% inward) to exclude background edges")
    print("  • High saturation threshold (S > 50) for vivid color focus")
    print("  • HSV-based color patches (S=200, V=200) for visibility")
    print("  • Human-readable color names (red/pink, blue, green, etc.)")
    print("  • Per-ID color CSV with HSV values and descriptions")
    print("  • NO MORE GRAY PATCHES!")


if __name__ == "__main__":
    main()
