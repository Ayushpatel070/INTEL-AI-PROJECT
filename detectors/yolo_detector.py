"""
YOLOv8 Detector module for object, human, and helmet detection.
"""
from ultralytics import YOLO
import numpy as np

class YOLODetector:
    def __init__(self, model_path):
        """Load YOLOv8 model from given path."""
        self.model = YOLO(model_path)
        self.class_names = self.model.names  # e.g., {0: 'person', 1: 'helmet', ...}

    def detect(self, frame, conf=0.3):
        """
        Run detection on a frame.
        Returns list of dicts: [{bbox, class_id, class_name, conf}, ...]
        """
        results = self.model(frame, conf=conf)
        detections = []
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, 'xyxy') else []
            scores = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, 'conf') else []
            class_ids = r.boxes.cls.cpu().numpy().astype(int) if hasattr(r.boxes, 'cls') else []
            for box, score, class_id in zip(boxes, scores, class_ids):
                detections.append({
                    'bbox': box,  # [x1, y1, x2, y2]
                    'class_id': class_id,
                    'class_name': self.class_names[class_id] if class_id in self.class_names else str(class_id),
                    'conf': float(score)
                })
        return detections 