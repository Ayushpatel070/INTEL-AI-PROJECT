import cv2
from ultralytics import YOLO


class YOLODetector:
    def __init__(self, model_path: str):
        print(f"[INFO] Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)
        # class_names is a dict: {id: class_name}
        self.class_names = self.model.model.names if hasattr(self.model.model, "names") else {}

        print(f"[INFO] Model loaded. Classes: {self.class_names}")

    def detect(self, frame, conf: float = 0.25):
        """
        Run detection on a single frame.
        Returns a list of dicts with keys: class_id, class_name, conf, bbox
        """
        results = self.model.predict(frame, conf=conf, verbose=False)

        detections = []
        if not results:
            return detections

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                class_name = self.class_names.get(cls_id, str(cls_id))
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                detection = {
                    "class_id": cls_id,
                    "class_name": class_name,
                    "conf": confidence,
                    "bbox": (x1, y1, x2, y2),
                }
                detections.append(detection)

                # --- DEBUG PRINT for every detection ---
                print(f"[DETECT] {class_name} ({cls_id}) conf={confidence:.2f} "
                      f"bbox=({x1},{y1},{x2},{y2})")

        return detections
