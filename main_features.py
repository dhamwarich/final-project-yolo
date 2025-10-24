"""
Module 5: Integration Script
End-to-end pipeline for pose feature extraction.
"""

from features.pose_extractor import load_pose_model, get_keypoints
from features.ratio_calculator import compile_features
from features.feature_logger import init_logger, log_features
from features.visualize_pose import draw_pose, overlay_metrics
import cv2
import time


def main():
    """Main pipeline for pose feature extraction."""
    
    # Configuration
    VIDEO_PATH = "videos/test_scene.MOV"
    CSV_PATH = "outputs/features.csv"
    OUTPUT_VIDEO_PATH = "outputs/annotated_video.mp4"
    
    print("Loading YOLOv8n-pose model...")
    model = load_pose_model("yolo")
    print("Model loaded successfully!")
    
    # Initialize logger
    init_logger(CSV_PATH)
    print(f"Logger initialized at {CSV_PATH}")
    
    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video {VIDEO_PATH}")
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
    
    frame_id = 0
    fps_log = []
    start_time = time.time()
    
    print("\nProcessing video...")
    print("Press 'q' to quit early")
    
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
            # Compile features
            feats = compile_features(person["keypoints"], person["bbox"])
            
            # Log to CSV
            log_features(frame_id, feats, person["conf"], CSV_PATH)
            
            # Visualize
            frame = draw_pose(frame, person["keypoints"])
            frame = overlay_metrics(frame, feats)
        
        # Add frame counter and FPS
        cv2.putText(
            frame,
            f"Frame: {frame_id}/{total_frames}  FPS: {inference_fps:.1f}",
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
        
        # Write to output video
        out.write(frame)
        
        # Display
        cv2.imshow("Pose Features", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nStopped by user")
            break
        
        frame_id += 1
        
        # Progress update every 30 frames
        if frame_id % 30 == 0:
            progress = (frame_id / total_frames) * 100
            avg_fps = sum(fps_log[-30:]) / min(30, len(fps_log))
            print(f"Progress: {progress:.1f}% | Avg FPS: {avg_fps:.1f}")
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Print statistics
    total_time = time.time() - start_time
    avg_fps = sum(fps_log) / len(fps_log) if fps_log else 0
    
    print("\n" + "="*50)
    print("PROCESSING COMPLETE")
    print("="*50)
    print(f"Total frames processed: {frame_id}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Features saved to: {CSV_PATH}")
    print(f"Annotated video saved to: {OUTPUT_VIDEO_PATH}")
    print("="*50)


if __name__ == "__main__":
    main()
