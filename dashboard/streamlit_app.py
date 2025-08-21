"""
Streamlit dashboard for real-time safety monitoring.
"""

import streamlit as st
import pandas as pd
import cv2
import os
import sys
import time
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.gpio_simulator import GPIOSimulator
try:
    from dashboard.camera_stream import show_live_camera  # when importing as a package
except Exception:
    from camera_stream import show_live_camera  # when running inside the dashboard dir

VIDEO_PATH = os.path.join("data", "output_annotated.mp4")
LOG_PATH = os.path.join("data", "event_log.csv")

st.set_page_config(page_title="Safety System Dashboard", layout="wide")
st.title("🤖 AI-Powered Real-Time Safety System Dashboard")

# Sidebar for controls
st.sidebar.header("System Controls")
auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 1, 30, 5)

# Machine status (simulate by reading last event)
def get_machine_status():
    if os.path.exists(LOG_PATH):
        df = pd.read_csv(LOG_PATH)
        if not df.empty and "Hazard Detected" in df["event"].iloc[-1]:
            return "Auto-stopped"
    return "Running"

# Create tabs
tab1, tab2, tab3 = st.tabs(["📹 Camera Feed", "⚙️ System Status", "📋 Event Log"])

with tab1:
    st.header("📹 Camera Feed Options")
    
    # Choose between recorded video and live stream
    feed_type = st.radio(
        "Choose camera feed type:",
        ["📼 Recorded Video (with AI detection)", "🎥 Live Camera Stream"],
        help="Recorded video shows AI detection results, Live stream shows raw camera feed"
    )
    
    if feed_type == "📼 Recorded Video (with AI detection)":
        # Show recorded video
        if os.path.exists(VIDEO_PATH):
            # Get file modification time
            mod_time = os.path.getmtime(VIDEO_PATH)
            mod_time_str = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
            
            st.info(f"📹 Latest recorded video with AI detection (updated: {mod_time_str})")
            video_file = open(VIDEO_PATH, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)
            
            # Add download button
            st.download_button(
                label="📥 Download Video",
                data=video_bytes,
                file_name=f"safety_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                mime="video/mp4"
            )
            
            st.info("💡 This video shows real-time AI detection with bounding boxes and safety alerts.")
        else:
            st.warning("⚠️ No recorded video available.")
            st.info("💡 Run 'python main.py' to start recording with AI detection.")
    
    else:  # Live Camera Stream
        show_live_camera()

with tab2:
    st.header("⚙️ System Status")
    
    # Show machine status
    status = get_machine_status()
    if status == "Running":
        st.success(f"🟢 Machine Status: {status}")
    else:
        st.error(f"🔴 Machine Status: {status}")
    
    # System info
    st.subheader("📊 System Info")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Detection Model", "YOLOv8")
    with col2:
        st.metric("Camera Status", "🟢 Active" if os.path.exists(VIDEO_PATH) else "🔴 Inactive")
    with col3:
        st.metric("Recording Status", "🟢 Recording" if os.path.exists(VIDEO_PATH) else "🔴 Not Recording")
    
    # Quick actions
    st.subheader("🚀 Quick Actions")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Refresh Now"):
            st.rerun()
    
    with col2:
        if st.button("📊 View System Logs"):
            st.info("System logs are shown in the Event Log tab.")

with tab3:
    st.header("📋 Hazard/Event Log")
    if os.path.exists(LOG_PATH):
        df = pd.read_csv(LOG_PATH)
        if not df.empty:
            # Add timestamp column if not exists
            if 'timestamp' not in df.columns:
                df['timestamp'] = pd.to_datetime(df.index, unit='s')
            
            # Show recent events
            st.subheader("Recent Events")
            recent_events = df.tail(10)
            for idx, row in recent_events.iterrows():
                details = row.get('details', row.get('description', ''))
                if "Hazard Detected" in str(row['event']):
                    st.error(f"🚨 {row['event']} - {details}")
                else:
                    st.success(f"✅ {row['event']} - {details}")
            
            # Show full log as table
            st.subheader("Full Event History")
            st.dataframe(df.tail(20), use_container_width=True)
            
            # Export button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Export Event Log",
                data=csv,
                file_name=f"event_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("📝 No events logged yet. The system will log events when hazards are detected.")
    else:
        st.info("📝 No events logged yet. Run main.py to start logging events.")

# Auto-refresh
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()

"""
Note: This file is executed directly by Streamlit using `streamlit run dashboard/streamlit_app.py`.
Do not call an undefined `main()` here.
"""