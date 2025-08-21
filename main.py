"""
Main entrypoint for AI-Powered Real-Time Safety System for Modular Manufacturing.
Runs real-time detection and machine control logic.
"""
import cv2
import os
import glob
import time
from detectors.yolo_detector import YOLODetector
from utils.gpio_simulator import GPIOSimulator
from utils.event_logger import EventLogger

MODEL_PATH = os.path.join("models", "yolov8n.pt")
VIDEO_PATH = os.path.join("data", "sample_video.mp4")
OUTPUT_PATH = os.path.join("data", "output_annotated.mp4")

def main():
    # Auto-select best available model (prefer helmet-capable)
    selected_model_path = MODEL_PATH
    try:
        helmet_candidates = []
        helmet_candidates += glob.glob(os.path.join("models", "*helmet*.pt"))
        helmet_candidates += glob.glob(os.path.join("models", "*hardhat*.pt"))
        helmet_candidates += glob.glob(os.path.join("runs", "detect", "**", "weights", "best.pt"), recursive=True)
        helmet_candidates = [p for p in helmet_candidates if os.path.exists(p)]
        if helmet_candidates:
            helmet_candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            selected_model_path = helmet_candidates[0]
        print(f"[INFO] Using model: {selected_model_path}")
    except Exception as e:
        print(f"[WARN] Auto-select model failed: {e}")

    # Initialize
    detector = YOLODetector(selected_model_path)
    gpio = GPIOSimulator()
    logger = EventLogger(os.path.join("data", "event_log.csv"))

    # Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[INFO] Camera not detected. Using test video instead.")
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open camera or test video at {VIDEO_PATH}.")
            return
    else:
        print("[INFO] Using live camera feed.")

    # Window
    try:
        cv2.namedWindow("Safety System", cv2.WINDOW_AUTOSIZE)
        cv2.setWindowProperty("Safety System", cv2.WND_PROP_TOPMOST, 1)
    except Exception as e:
        print(f"[WARNING] Could not create window: {e}")

    time.sleep(2)

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    # Debounce
    HAZARD_ON_FRAMES = 6
    HAZARD_OFF_FRAMES = 8
    hazard_frames_streak = 0
    safe_frames_streak = 0

    # Label sets
    helmet_label_candidates = {
        "helmet", "hardhat", "safety helmet", "helmet_on",
        "helmet worn", "helmet-worn", "hard hat"
    }
    no_helmet_label_candidates = {
        "no-helmet", "no_helmet", "without_helmet", "helmet off",
        "helmet_off", "no helmet", "not wearing helmet",
        "helmet missing", "helmet absent", "nohelmet",
        "no-hardhat", "no hardhat"
    }

    def normalize(name: str) -> str:
        return str(name).strip().lower()

    def is_helmet_label(name: str) -> bool:
        n = normalize(name)
        return n in helmet_label_candidates

    def is_no_helmet_label(name: str) -> bool:
        n = normalize(name)
        return n in no_helmet_label_candidates

    # Print model classes
    try:
        model_classes = set(detector.class_names.values())
        print(f"[INFO] Model classes: {sorted(list(model_classes))[:50]}")
    except Exception:
        pass

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame, conf=0.15)
        hazard_detected = False
        alert_msgs = []

        persons, helmets, explicit_no_helmets = [], [], []

        # Categorize detections
        for det in detections:
            cname = normalize(det['class_name'])
            if cname == "person":
                persons.append(det)
            elif is_helmet_label(cname):
                helmets.append(det)
            elif is_no_helmet_label(cname):
                explicit_no_helmets.append(det)

        # Draw all detections
        for det in detections:
            cname = normalize(det['class_name'])
            x1, y1, x2, y2 = map(int, det['bbox'])
            conf = det['conf']
            if is_helmet_label(cname):
                color = (0, 200, 0)  # green
            elif is_no_helmet_label(cname):
                color = (0, 0, 255)  # red
            elif cname == "person":
                color = (255, 255, 0)
            else:
                color = (200, 200, 200)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{det['class_name']} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Safety logic: frame-level helmet check
        if explicit_no_helmets:
            hazard_detected = True
            alert_msgs.append("No helmet detected!")
        else:
            if persons:
                if helmets:
                    # At least one helmet anywhere in frame → safe
                    hazard_detected = False
                    for person in persons:
                        px1, py1, px2, py2 = map(int, person['bbox'])
                        label = f"person {person['conf']:.2f} | Helmet"
                        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 200, 0), 2)
                        cv2.putText(frame, label, (px1, py1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
                else:
                    hazard_detected = True
                    alert_msgs.append("Person without helmet detected!")
                    for person in persons:
                        px1, py1, px2, py2 = map(int, person['bbox'])
                        label = f"person {person['conf']:.2f}"   # no “NO HELMET” suffix
                        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
                        cv2.putText(frame, label, (px1, py1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)



        # Handle hazard with debounce
        if hazard_detected:
            hazard_frames_streak += 1
            safe_frames_streak = 0
        else:
            safe_frames_streak += 1
            hazard_frames_streak = 0

        # Machine state
        if hazard_frames_streak >= HAZARD_ON_FRAMES:
            if gpio.machine_running:
                gpio.stop_machine()
                logger.log_event("Hazard Detected", "; ".join(alert_msgs))
            cv2.putText(frame, "ALERT: MACHINE STOPPED!", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        elif safe_frames_streak >= HAZARD_OFF_FRAMES:
            if not gpio.machine_running:
                gpio.start_machine()
                logger.log_event("Machine Restarted", "No hazard detected.")
            cv2.putText(frame, "Machine Status: RUNNING", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "Checking helmet status...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Show frame
        cv2.imshow("Safety System", frame)

        if out is None:
            h, w = frame.shape[:2]
            out = cv2.VideoWriter(OUTPUT_PATH, fourcc, 20.0, (w, h))
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
