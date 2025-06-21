"""
Real-time camera streaming component for Streamlit dashboard.
"""

import cv2
import numpy as np
import streamlit as st
from threading import Thread, Lock
import time

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
    
    # Display camera feed
    camera_placeholder = st.empty()
    
    if st.session_state.camera_stream.running:
        while st.session_state.camera_stream.running:
            frame = st.session_state.camera_stream.get_frame()
            if frame is not None:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                camera_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
            time.sleep(0.1)
    else:
        camera_placeholder.info("🎥 Click 'Start Camera' to begin live streaming")

if __name__ == "__main__":
    show_live_camera() 