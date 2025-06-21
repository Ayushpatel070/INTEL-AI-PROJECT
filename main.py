"""
Main entrypoint for AI-Powered Real-Time Safety System for Modular Manufacturing.
Runs real-time detection and machine control logic.
"""
import cv2
import os
from detectors.yolo_detector import YOLODetector
from utils.gpio_simulator import GPIOSimulator
from utils.event_logger import EventLogger
import time

MODEL_PATH = os.path.join("models", "yolov8n.pt")
VIDEO_PATH = os.path.join("data", "sample_video.mp4")
OUTPUT_PATH = os.path.join("data", "output_annotated.mp4")

HAZARD_CLASSES = ["person", "no-helmet", "helmet"]  # 'no-helmet' if using custom model


def main():
    # Initialize modules
    detector = YOLODetector(MODEL_PATH)
    gpio = GPIOSimulator()
    logger = EventLogger(os.path.join("data", "event_log.csv"))

    # Try to open camera first
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[INFO] Camera not detected. Using test video instead.")
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open camera or test video at {VIDEO_PATH}.")
            return
    else:
        print("[INFO] Using live camera feed.")

    # Create and position the window
    try:
        cv2.namedWindow("Safety System", cv2.WINDOW_AUTOSIZE)
        cv2.setWindowProperty("Safety System", cv2.WND_PROP_TOPMOST, 1)
        print("[INFO] Camera window created. Look for 'Safety System' window.")
        print("[INFO] If you don't see the window, check your taskbar or try Alt+Tab.")
    except Exception as e:
        print(f"[WARNING] Could not create window: {e}")
    
    # Give the window time to appear
    time.sleep(3)

    # Prepare video writer
    try:
        # Try different methods to create the fourcc codec
        if hasattr(cv2, 'VideoWriter_fourcc'):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
        else:
            # Alternative approach for some OpenCV versions
            fourcc = 0x7634706d  # mp4v codec as integer
    except Exception as e:
        print(f"[WARNING] Video codec error: {e}")
        fourcc = 0x7634706d  # Fallback to mp4v codec
    out = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        detections = detector.detect(frame)
        hazard_detected = False
        alert_msgs = []

        for det in detections:
            class_name = det['class_name']
            conf = det['conf']
            x1, y1, x2, y2 = map(int, det['bbox'])
            color = (0, 255, 0)
            label = f"{class_name} {conf:.2f}"
            if class_name == "person":
                color = (0, 0, 255)
                hazard_detected = True
                alert_msgs.append("Human detected in hazard zone!")
            if class_name == "no-helmet":
                color = (0, 165, 255)
                hazard_detected = True
                alert_msgs.append("No helmet detected!")
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Handle hazard
        if hazard_detected:
            if gpio.machine_running:
                gpio.stop_machine()
                logger.log_event("Hazard Detected", "; ".join(alert_msgs))
            cv2.putText(frame, "ALERT: MACHINE STOPPED!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        else:
            if not gpio.machine_running:
                gpio.start_machine()
                logger.log_event("Machine Restarted", "No hazard detected.")
            cv2.putText(frame, "Machine Status: RUNNING", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        # Show frame
        cv2.imshow("Safety System", frame)

        # Write output video
        if out is None:
            h, w = frame.shape[:2]
            out = cv2.VideoWriter(OUTPUT_PATH, fourcc, 20.0, (w, h))
        out.write(frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 