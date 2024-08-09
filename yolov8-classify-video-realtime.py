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
content.content_yolov8_classify_video_realtime()

# Video selection
video_path = utils.video_selector()

# Initialize video capture
cap = cv2.VideoCapture(video_path)

# Create three columns for plots
col1, col2, col3 = st.columns(3)

# Initialize plots
with col1:
    st.markdown("## 🎥 Video Feed")
    video_placeholder = st.empty()

with col2:
    st.markdown("## 📊 Classification Results")
    bar_chart_placeholder = st.empty()

with col3:
    st.markdown("## 📈 Probability History")
    line_chart_placeholder = st.empty()

# Initialize deque for storing historical probabilities
history_length = 250
prob_history = {
    class_name: deque(maxlen=history_length) for class_name in model.names.values()
}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform prediction
    results = model.predict(frame)

    for r in results:
        names = r.names
        probs = r.probs

        if probs is not None:
            # Extract class names and probabilities
            class_names = list(names.values())
            probabilities = probs.data.tolist()

            # Update probability history
            for class_name, prob in zip(class_names, probabilities):
                prob_history[class_name].append(prob)

            # Create DataFrame for bar chart
            df_bar = pd.DataFrame({"Class": class_names, "Probability": probabilities})

            # Create bar chart
            fig_bar = go.Figure(
                data=[go.Bar(x=df_bar["Class"], y=df_bar["Probability"])]
            )
            fig_bar.update_layout(
                title="Current Frame Probabilities",
                xaxis_title="Class",
                yaxis_title="Probability",
                yaxis_range=[0, 1],
            )

            # Create line chart
            fig_line = go.Figure()
            for class_name, probs in prob_history.items():
                fig_line.add_trace(
                    go.Scatter(y=list(probs), mode="lines", name=class_name)
                )
            fig_line.update_layout(
                title="Probability History",
                xaxis_title="Frame",
                yaxis_title="Probability",
                yaxis_range=[0, 1],
                xaxis_range=[0, 250],
            )

            # Update plots
            with col1:
                video_placeholder.image(frame, channels="BGR", use_column_width=True)
            with col2:
                bar_chart_placeholder.plotly_chart(fig_bar, use_container_width=True)
                line_chart_placeholder.plotly_chart(fig_line, use_container_width=True)

# Release video capture
cap.release()

st.success("Video processing complete.", icon="✅")
