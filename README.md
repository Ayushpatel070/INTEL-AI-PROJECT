# AI-Powered Real-Time Safety System for Modular Manufacturing

## Overview
This system uses computer vision (YOLOv8, MediaPipe) to monitor factory activities in real-time, detect hazards (humans, no-helmet, falls), and automatically stop machines to prevent accidents. It features a Streamlit dashboard for live monitoring, alerts, and logs.

## Features
- Real-time object/human/helmet detection (YOLOv8)
- Fall/posture detection (MediaPipe, optional)
- Machine stop/start simulation
- Event logging with timestamps
- Streamlit dashboard: live video, alerts, logs, machine status

## Project Structure
```
main.py
requirements.txt
detectors/
  yolo_detector.py
utils/
  gpio_simulator.py
  event_logger.py
dashboard/
  streamlit_app.py
models/
data/
  sample_video.mp4
```

## Setup
1. Install dependencies:
   ```
pip install -r requirements.txt
   ```
2. Download YOLOv8 weights (e.g., `yolov8n.pt`, `helmet.pt`) into `models/`.
3. Place a sample video in `data/sample_video.mp4` (or use a webcam).

## Running
- **Detection & Control:**
  ```
python main.py
  ```
- **Dashboard:**
  ```
streamlit run dashboard/streamlit_app.py
  ```

## Notes
- For Raspberry Pi/Jetson Nano: Replace `gpio_simulator.py` with hardware GPIO code.
- For helmet detection, use a YOLOv8 model trained for helmets (see README links). 