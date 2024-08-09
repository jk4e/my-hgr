import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from collections import deque

# Local Modules
import content
import utils

# Sidebar
model_path = utils.yolo_classify_sidebar_options()

# Load YOLO model
with st.spinner("Model is downloading..."):
    model = YOLO(model_path)
    class_names = list(
        model.names.values()
    )  # Convert dictionary to list of class names
st.success("Model loaded successfully!", icon="✅")

# Content
content.content_yolov8_classify_video()

# Video selection
video_path = utils.video_selector()


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    results = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        result = model.predict(frame)[0]
        results.append(result)

    cap.release()
    return frames, results


# Process video<
with st.spinner("Processing video..."):
    frames, results = process_video(video_path)

# Display results
if frames and results:
    # Create two columns for plots
    col1, col2 = st.columns(2)

    # Initialize plots
    with col1:
        st.markdown("## 🎞️ Video Feed")
        video_placeholder = st.empty()

    with col2:
        st.markdown("## 📊 Classification Results")
        bar_chart_placeholder = st.empty()
        line_chart_placeholder = st.empty()

    # Initialize probability history
    class_names = list(results[0].names.values())
    prob_history = {class_name: [] for class_name in class_names}

    with col1:
        # Slider for frame selection
        frame_index = st.slider("**Select frame**", 0, len(frames) - 1, 0)

    # Update plots based on selected frame
    current_frame = frames[frame_index]
    current_result = results[frame_index]

    # Update video feed
    video_placeholder.image(current_frame, channels="BGR", use_column_width=True)

    # Update bar chart
    probabilities = current_result.probs.data.tolist()
    df_bar = pd.DataFrame({"Class": class_names, "Probability": probabilities})
    fig_bar = go.Figure(data=[go.Bar(x=df_bar["Class"], y=df_bar["Probability"])])
    fig_bar.update_layout(
        title="Current Frame Probabilities",
        xaxis_title="Class",
        yaxis_title="Probability",
        yaxis_range=[0, 1],
    )
    bar_chart_placeholder.plotly_chart(fig_bar, use_container_width=True)

    # Update line chart
    for i, result in enumerate(results):
        probs = result.probs.data.tolist()
        for class_name, prob in zip(class_names, probs):
            prob_history[class_name].append(prob)

    fig_line = go.Figure()
    for class_name, probs in prob_history.items():
        fig_line.add_trace(go.Scatter(y=probs, mode="lines", name=class_name))
    fig_line.update_layout(
        title="Probability History",
        xaxis_title="Frame",
        yaxis_title="Probability",
        yaxis_range=[0, 1],
    )
    line_chart_placeholder.plotly_chart(fig_line, use_container_width=True)

    st.success(
        "Video processing complete. Use the slider to explore different frames.",
        icon="✅",
    )
else:
    st.warning("No video processed. Please upload a video or select a predefined one.")
