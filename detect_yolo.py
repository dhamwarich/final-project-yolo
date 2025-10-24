from ultralytics import YOLO
import cv2
import time

# Load YOLOv8n model
model = YOLO("yolov8n.pt")

# Load video
cap = cv2.VideoCapture("videos/test_scene.mp4")
fps_log = []
frame_id = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    start = time.time()
    results = model(frame)
    end = time.time()

    # Draw detections
    annotated = results[0].plot()
    cv2.imshow("YOLOv8n Detection", annotated)

    # FPS calc
    fps = 1 / (end - start)
    fps_log.append(fps)
    frame_id += 1

    # Quit with Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Average FPS: {sum(fps_log)/len(fps_log):.2f}")
