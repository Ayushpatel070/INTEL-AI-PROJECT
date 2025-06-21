"""
Logs detected hazard events with timestamps to a CSV file.
"""
import pandas as pd
from datetime import datetime
import os

class EventLogger:
    def __init__(self, log_path="event_log.csv"):
        self.log_path = log_path
        if not os.path.exists(self.log_path):
            df = pd.DataFrame(columns=["timestamp", "event", "details"])
            df.to_csv(self.log_path, index=False)

    def log_event(self, event, details=""):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df = pd.DataFrame([[timestamp, event, details]], columns=["timestamp", "event", "details"])
        df.to_csv(self.log_path, mode="a", header=False, index=False) 