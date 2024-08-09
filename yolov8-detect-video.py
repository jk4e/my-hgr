import streamlit as st
from ultralytics import YOLO
import cv2
import time
import torch

# Local Modules
import content
import utils

# Sidebar
model_path, confidence, iou = utils.yolo_detect_sidebar_options()

# Load YOLO model
with st.spinner("Model is downloading..."):
    model = YOLO(model_path)
    class_names = list(
        model.names.values()
    )  # Convert dictionary to list of class names
st.success("Model loaded successfully!", icon="✅")

# Content
content.content_yolov8_detect_video()

# Video selection
video_path = utils.video_selector()

# Class selection
selected_classes = st.sidebar.multiselect(
    "Classes", class_names, default=class_names[:3]
)
selected_ind = [class_names.index(option) for option in selected_classes]

# Create two columns for display
col1, col2 = st.columns(2)

with col1:
    st.markdown("## 🎞️ Video Feed")
    original_frame_placeholder = st.empty()
    col11, col12 = st.columns([1, 1])
    with col11:
        run_button = st.button(
            ":green[Run]", type="secondary", use_container_width=True
        )
    with col12:
        stop_button = st.button("Stop", type="primary", use_container_width=True)

with col2:
    st.markdown("## 📊 Annotated Video Feed")
    annotated_frame_placeholder = st.empty()

fps_placeholder = st.sidebar.empty()

st.sidebar.markdown(
    "Click the 'Start' button to start the video feed. Click the 'Stop' button to stop the video feed."
)

if run_button:
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("Could not open video source.")
    else:

        while cap.isOpened() and not stop_button:
            success, frame = cap.read()
            if not success:
                st.warning("End of video or failed to read frame.", icon="⚠️")
                break

            start_time = time.time()

            # Perform detection
            results = model(frame, conf=confidence, iou=iou, classes=selected_ind)

            annotated_frame = results[0].plot()

            # Display frames
            original_frame_placeholder.image(
                frame, channels="BGR", use_column_width=True
            )
            annotated_frame_placeholder.image(
                annotated_frame, channels="BGR", use_column_width=True
            )

            # Calculate and display FPS
            fps = 1 / (time.time() - start_time)
            fps_placeholder.metric("FPS", f"{fps:.2f}")

            if stop_button:
                break

        cap.release()
        torch.cuda.empty_cache()

# Clear CUDA memory
torch.cuda.empty_cache()
