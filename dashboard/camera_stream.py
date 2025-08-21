"""
Real-time camera streaming component for Streamlit dashboard.
Now includes helmet detection and machine status overlays (RUNNING/STOPPED).
"""

import cv2
import numpy as np
import streamlit as st
from threading import Thread, Lock
import time
import os
import glob

# These imports rely on sys.path adjustment done in streamlit_app.py
from detectors.yolo_detector import YOLODetector
from utils.gpio_simulator import GPIOSimulator
from utils.event_logger import EventLogger

class CameraStream:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.frame = None
        self.lock = Lock()
        self.running = False
        self.thread = None
    
    def start(self):
        """Start the camera stream in a separate thread."""
        if self.running:
            return
        
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            st.error("❌ Cannot open camera!")
            return False
        
        self.running = True
        self.thread = Thread(target=self._update_frame, daemon=True)
        self.thread.start()
        return True
    
    def _update_frame(self):
        """Update frame in background thread."""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            time.sleep(0.03)  # ~30 FPS
    
    def get_frame(self):
        """Get the current frame."""
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
        return None
    
    def stop(self):
        """Stop the camera stream."""
        self.running = False
        if self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()


class LiveHelmetMonitor:
    """Encapsulates helmet detection and machine control for live frames."""
    def __init__(self):
        # Select model: prefer any model with 'helmet' in name under models/, then newest runs/**/weights/best.pt
        default_model = os.path.join("models", "yolov8n.pt")
        selected_model = default_model
        try:
            candidates = []
            candidates += glob.glob(os.path.join("models", "*helmet*.pt"))
            candidates += glob.glob(os.path.join("models", "*hardhat*.pt"))
            candidates += glob.glob(os.path.join("runs", "detect", "**", "weights", "best.pt"), recursive=True)
            candidates = [p for p in candidates if os.path.exists(p)]
            if candidates:
                candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                selected_model = candidates[0]
        except Exception:
            pass

        self.detector = YOLODetector(selected_model)
        self.gpio = GPIOSimulator()
        self.logger = EventLogger(os.path.join("data", "event_log.csv"))

        # Label handling
        try:
            class_names = set(self.detector.class_names.values()) if hasattr(self.detector, 'class_names') else set()
        except Exception:
            class_names = set()

        # Broad label matching
        self.helmet_label_candidates = {"helmet", "hardhat", "safety helmet", "helmet_on", "helmet worn", "helmet-worn"}
        self.no_helmet_label_candidates = {"no-helmet", "no_helmet", "without_helmet", "helmet off", "helmet_off", "no helmet", "not wearing helmet", "helmet missing", "helmet absent", "nohelmet", "no-hardhat", "no hardhat"}

        def normalize_label(name: str) -> str:
            return str(name).strip().lower()

        self.normalize_label = normalize_label
        self.helmet_supported = any(self._is_helmet_label(n) for n in class_names)

        # Debounce (live camera: stop immediately on violation, recover quickly on safe)
        self.HAZARD_ON_FRAMES = 1
        self.HAZARD_OFF_FRAMES = 3
        self.hazard_frames_streak = 0
        self.safe_frames_streak = 0

    def _is_helmet_label(self, name: str) -> bool:
        n = self.normalize_label(name)
        if n in self.helmet_label_candidates:
            return True
        return ("helmet" in n) and not any(w in n for w in ["no", "without", "not", "absent", "missing", "none", "off"])

    def _is_no_helmet_label(self, name: str) -> bool:
        n = self.normalize_label(name)
        if n in self.no_helmet_label_candidates:
            return True
        return any(p in n for p in ["no helmet", "no-helmet", "without helmet", "helmet off", "helmet_off", "not wearing helmet", "helmet missing", "helmet absent", "nohelmet"])

    @staticmethod
    def _iou(box_a, box_b):
        x1a, y1a, x2a, y2a = box_a
        x1b, y1b, x2b, y2b = box_b
        inter_x1 = max(x1a, x1b)
        inter_y1 = max(y1a, y1b)
        inter_x2 = min(x2a, x2b)
        inter_y2 = min(y2a, y2b)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area_a = max(0, x2a - x1a) * max(0, y2a - y1a)
        area_b = max(0, x2b - x1b) * max(0, y2b - y1b)
        union = area_a + area_b - inter_area if (area_a + area_b - inter_area) > 0 else 1e-6
        return inter_area / union

    @staticmethod
    def _center_inside(inner_box, outer_box):
        xi1, yi1, xi2, yi2 = inner_box
        xo1, yo1, xo2, yo2 = outer_box
        cx = (xi1 + xi2) / 2.0
        cy = (yi1 + yi2) / 2.0
        return (xo1 <= cx <= xo2) and (yo1 <= cy <= yo2)

    def analyze_and_annotate(self, frame):
        # Detect
        detections = self.detector.detect(frame, conf=0.2)
        persons, helmets, explicit_no_helmets = [], [], []
        for det in detections:
            cname = self.normalize_label(det['class_name'])
            if cname == "person":
                persons.append(det)
            elif self._is_helmet_label(cname):
                helmets.append(det)
            elif self._is_no_helmet_label(cname):
                explicit_no_helmets.append(det)

        # Draw helmet/no-helmet tags
        for det in detections:
            cname = self.normalize_label(det['class_name'])
            if cname == "person":
                continue
            x1, y1, x2, y2 = map(int, det['bbox'])
            color = (0, 200, 0) if self._is_helmet_label(cname) else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{det['class_name']} {det['conf']:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        hazard_detected = False
        alert_msgs = []

        # Check each person for helmet (head region top 60%) and any explicit no-helmet tag that overlaps the person
        violating_people = 0
        for person in persons:
            px1, py1, px2, py2 = map(int, person['bbox'])
            person_box = (px1, py1, px2, py2)
            head_box = (px1, py1, px2, py1 + int(0.60 * (py2 - py1)))

            has_helmet = False
            has_no_helmet_tag = False
            if self.helmet_supported and helmets:
                for h in helmets:
                    hx1, hy1, hx2, hy2 = map(int, h['bbox'])
                    hb = (hx1, hy1, hx2, hy2)
                    # Use head box IoU or center-in-head for a tighter match
                    if self._center_inside(hb, head_box) or self._iou(hb, head_box) > 0.02:
                        has_helmet = True
                        break

            # Associate explicit "no-helmet" tags to the person before flagging
            if explicit_no_helmets:
                for nh in explicit_no_helmets:
                    nx1, ny1, nx2, ny2 = map(int, nh['bbox'])
                    nb = (nx1, ny1, nx2, ny2)
                    if self._center_inside(nb, person_box) or self._iou(nb, person_box) > 0.10:
                        has_no_helmet_tag = True
                        break

            if self.helmet_supported:
                if has_helmet and not has_no_helmet_tag:
                    color = (0, 200, 0)
                    suffix = " | Helmet"
                else:
                    color = (0, 165, 255)
                    suffix = " | NO HELMET"
                    violating_people += 1
            else:
                color = (0, 255, 0)
                suffix = " | (helmet model not available)"

            cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
            cv2.putText(frame, f"person {person['conf']:.2f}{suffix}", (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Immediate hazard if explicit no-helmet tag is present anywhere
        if explicit_no_helmets:
            hazard_detected = True
            alert_msgs.append("No helmet detected!")
        else:
            if persons:
                if violating_people > 0:
                    hazard_detected = True
                    alert_msgs.append("Person without helmet detected!")
            else:
                # Conservative fallback when no person class is available: if no helmets seen, stop
                if not helmets:
                    hazard_detected = True
                    alert_msgs.append("No helmet visible in frame")

        # Debounce machine state
        if hazard_detected:
            self.hazard_frames_streak += 1
            self.safe_frames_streak = 0
        else:
            self.safe_frames_streak += 1
            self.hazard_frames_streak = 0

        # Helmet support notice and debug model info
        model_name = getattr(self.detector.model, 'model', None)
        try:
            selected_name = os.path.basename(self.detector.model.pt_path if hasattr(self.detector.model, 'pt_path') else getattr(self.detector.model, 'ckpt_path', 'model'))
        except Exception:
            selected_name = 'model'
        if not self.helmet_supported:
            cv2.putText(frame, "Model lacks 'helmet' class — cannot enforce helmet rule", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Model: {selected_name}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Apply state
        if self.hazard_frames_streak >= self.HAZARD_ON_FRAMES:
            if self.gpio.machine_running:
                self.gpio.stop_machine()
                self.logger.log_event("Hazard Detected", "; ".join(alert_msgs))
            cv2.putText(frame, "ALERT: MACHINE STOPPED!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        elif self.safe_frames_streak >= self.HAZARD_OFF_FRAMES:
            if not self.gpio.machine_running:
                self.gpio.start_machine()
                self.logger.log_event("Machine Restarted", "No hazard detected.")
            cv2.putText(frame, "Machine Status: RUNNING", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "Checking helmet status...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        return frame

def show_live_camera():
    """Display live camera feed in Streamlit."""
    st.header("📹 Live Camera Feed")
    
    # Initialize camera stream
    if 'camera_stream' not in st.session_state:
        st.session_state.camera_stream = CameraStream()
    
    # Camera controls
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎥 Start Camera"):
            if st.session_state.camera_stream.start():
                st.success("✅ Camera started!")
            else:
                st.error("❌ Failed to start camera!")
    
    with col2:
        if st.button("⏹️ Stop Camera"):
            st.session_state.camera_stream.stop()
            st.success("✅ Camera stopped!")
    
    # Initialize live helmet monitor
    if 'helmet_monitor' not in st.session_state:
        st.session_state.helmet_monitor = LiveHelmetMonitor()

    # Display only the latest frame (non-blocking) with AI overlays
    camera_placeholder = st.empty()
    if st.session_state.camera_stream.running:
        frame = st.session_state.camera_stream.get_frame()
        if frame is not None:
            annotated = st.session_state.helmet_monitor.analyze_and_annotate(frame)
            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        else:
            camera_placeholder.warning("Waiting for camera frames...")
    else:
        camera_placeholder.info("🎥 Click 'Start Camera' to begin live streaming")

if __name__ == "__main__":
    show_live_camera() 