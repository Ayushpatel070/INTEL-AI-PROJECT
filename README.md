## 🤖🛡️ AI-Powered Real-Time Safety System

### 🧭 Overview
Real-time safety monitoring using computer vision (Ultralytics YOLO) to detect people and helmet compliance and to automatically stop/start a simulated machine. Includes a Streamlit dashboard for live monitoring and event logs.

### ✨ Key Features
- ⚡ **Real-time detection**: person, helmet/no-helmet tags (if supported by the model)
- 🧰 **Machine control**: simulated start/stop via GPIO abstraction
- 📝 **Event logging**: CSV with timestamps and details
- 📊 **Dashboard**: live feed with overlays, machine status, and event history

### 📦 Requirements
- Python 3.9–3.11
- Windows/macOS/Linux (Webcam required for live features)
- Install dependencies:
```bash
pip install -r requirements.txt
```

### 📁 Project Structure
```
INTEL-AI-PROJECT/
├─ main.py                      # Real-time detection + annotated recording
├─ dashboard/
│  ├─ streamlit_app.py         # Streamlit UI (tabs: live, status, logs)
│  └─ camera_stream.py         # Live camera + helmet logic + overlays
├─ detectors/
│  └─ yolo_detector.py         # Ultralytics YOLO wrapper
├─ utils/
│  ├─ gpio_simulator.py        # Simulated relay (machine start/stop)
│  └─ event_logger.py          # CSV logger
├─ data/
│  ├─ output_annotated.mp4     # Saved annotated video (from main.py)
│  └─ event_log.csv            # Event log
├─ models/
│  ├─ helmet-best.pt           # Example helmet-capable model (optional)
│  ├─ hardhat-best.pt          # Example hardhat-capable model (optional)
│  └─ yolov8n.pt               # Default model fallback
├─ runs/detect/.../weights/    # Optional: training outputs (best.pt)
├─ test_camera.py              # Simple webcam sanity check
└─ README.md
```

### 🚀 Quick Start
- 🎥 **Run with annotated recording (OpenCV window):**
```bash
python main.py
```
  - Press `q` to quit. Output saved to `data/output_annotated.mp4`. Events in `data/event_log.csv`.

- 🌐 **Run the Streamlit dashboard (recommended):**
```bash
streamlit run dashboard/streamlit_app.py
```
  - 🔗 Open `http://localhost:8501`.
  - 🧭 Tabs: Camera Feed (recorded video or live), System Status, Event Log.

### 🧠 Model Selection Logic
Both `main.py` and the dashboard auto-pick the most suitable model:
- 🎯 Prefer `models/*helmet*.pt` or `models/*hardhat*.pt`
- 🕒 Otherwise, prefer latest `runs/detect/**/weights/best.pt`
- ↩️ Fallback to `models/yolov8n.pt`

💡 If your chosen model lacks an explicit helmet class, live view will show a notice and only person detection will be used.

### 🔍 How Detection and Safety Logic Work
- 🧩 Detector: `detectors/ yolo_detector.py` wraps Ultralytics YOLO, returning `class_id`, `class_name`, `conf`, `bbox`.

- 📼 In `main.py` (record + window):
  - 🎚️ Confidence: `conf=0.15`
  - ⏱️ Debounce: `HAZARD_ON_FRAMES=6`, `HAZARD_OFF_FRAMES=8`
  - 🚨 Hazard if: explicit no-helmet tag is present, or person(s) detected without any helmet in frame.
  - 🖥️ Machine state overlay: “ALERT: MACHINE STOPPED!” or “Machine Status: RUNNING”.

- 🎦 In `dashboard/camera_stream.py` (live stream):
  - 🎚️ Confidence: `conf=0.2`
  - ⏱️ Debounce: `HAZARD_ON_FRAMES=1`, `HAZARD_OFF_FRAMES=3` (faster response)
  - 🧠 Associates helmets with a person’s head region (top 60% of the person box) using center-in-head or small IoU.
  - 🚨 Hazard if: explicit no-helmet tag, or a person without an associated helmet, or fallback when neither persons nor helmets are visible.

### ⚙️ Machine Control Abstraction
- 🔌 `utils/gpio_simulator.GPIOSimulator`
  - ▶️ `.start_machine()` sets relay pin to ON, ⏹️ `.stop_machine()` sets to OFF
  - 🔁 Replace with real GPIO (e.g., RPi.GPIO) by implementing the same methods.

### 📊 Logs and Outputs
- 🎞️ Annotated video: `data/output_annotated.mp4` (from `main.py`)
- 📝 Event log: `data/event_log.csv` with columns `[timestamp, event, details]`
- 📋 Dashboard “Event Log” tab displays recent and full history and lets you export CSV.

### 🔧 Configuration
- 🎚️ Change detection confidence in:
  - `main.py` → `detector.detect(frame, conf=0.15)`
  - `dashboard/camera_stream.py` → `conf=0.2`
- ⏱️ Adjust debounce:
  - `main.py` → `HAZARD_ON_FRAMES=6`, `HAZARD_OFF_FRAMES=8`
  - `dashboard/camera_stream.py` → `HAZARD_ON_FRAMES=1`, `HAZARD_OFF_FRAMES=3`
- 🎥 Camera index: `0` by default (`cv2.VideoCapture(0)`); change if you have multiple cameras.

### 🧪 Testing Your Camera
```bash
python test_camera.py
```
✅ If a window opens and frames update, your webcam works.

### 🛠️ Troubleshooting
- 📷 **Camera cannot open**: Ensure no other app uses the webcam; try camera index `1` or `2`.
- 🛑 **Live shows running after helmet removal**: Live logic stops on no-helmet tags, on people-without-helmets, and conservatively when neither persons nor helmets are visible. If needed, lower `HAZARD_OFF_FRAMES` or increase `conf`.
- 🪖 **Model lacks helmet class**: You’ll see a yellow notice. Use `models/helmet-best.pt` or place your trained `best.pt` under `runs/detect/**/weights/`.
- 🐢 **Performance issues**: Reduce input resolution or use a smaller model (e.g., `yolov8n.pt`). GPU support improves FPS.
- 💾 **No video saved**: Only `main.py` saves video. The dashboard shows frames but does not record by default.

### 🧩 Extending/Integrating
- 🔌 Replace `GPIOSimulator` with your hardware driver, preserving `.start_machine()/.stop_machine()` and `.machine_running`.
- 🧠 Add classes or retrain YOLO. Place weights under `models/` or `runs/detect/**/weights/` for auto-pickup.

### 📄 License
For educational/demo use. Add your preferred license file if distributing.

### 🙏 Acknowledgements
- 🤖 Ultralytics YOLO (object detection)
- 🎥 OpenCV (video I/O and drawing)
- 🌐 Streamlit (dashboard)