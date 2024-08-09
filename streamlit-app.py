import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Custom HGRs",
    page_icon="🖐️",
    initial_sidebar_state="expanded",
    layout="wide",
)

# Define pages
pages = [
    st.Page("home.py", title="Home", icon="🏠"),
    st.Page("yolov8-classify.py", title="YOLOv8 Classify", icon="👋"),
    st.Page("yolov8-classify-photo.py", title="YOLOv8 Classify - Photo", icon="📸"),
    st.Page("yolov8-classify-video.py", title="YOLOv8 Classify - Video", icon="🎞️"),
    st.Page(
        "yolov8-classify-video-realtime.py",
        title="YOLOv8 Classify - Video RTP",
        icon="🎥",
    ),
    st.Page("yolov8-classify-webcam.py", title="YOLOv8 Classify - Webcam", icon="📷"),
    st.Page("yolov8-detect.py", title="YOLOv8 Detect", icon="👋"),
    st.Page("yolov8-detect-photo.py", title="YOLOv8 Detect - Photo", icon="📸"),
    st.Page("yolov8-detect-video.py", title="YOLOv8 Detect - Video", icon="🎞️"),
    st.Page("yolov8-detect-webcam.py", title="YOLOv8 Detect - Webcam", icon="📷"),
    st.Page("mediapipe-hgr-task.py", title="MP Gesture Recognition", icon="👋"),
    st.Page(
        "mediapipe-hgr-task-photo.py", title="MP Gesture Recognition - Photo", icon="📸"
    ),
    st.Page(
        "mediapipe-hgr-task-video.py",
        title="MP Gesture Recognition - Video",
        icon="🎞️",
    ),
    st.Page(
        "mediapipe-hgr-task-webcam.py",
        title="MP Gesture Recognition - Webcam",
        icon="📷",
    ),
]

# Create and run navigation
navigation = st.navigation(
    {
        "HOME": [pages[0]],
        "YOLOv8 CLASSIFY": pages[1:6],
        "YOLOv8 DETECT": pages[6:10],
        "MEDIAPIPE (MP)": pages[10:14],
    }
)
navigation.run()
