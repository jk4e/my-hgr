import streamlit as st
import cv2
import torch
from ultralytics import YOLO
import time
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Local Modules
import content
import utils


# Sidebar
model_path = utils.yolo_classify_sidebar_options()

fps_placeholder = st.sidebar.empty()

st.sidebar.markdown(
    "Click the 'Run' button to start the webcam feed. Click the 'Stop' button to stop the webcam feed."
)

# Load YOLO model
with st.spinner("Model is downloading..."):
    model = YOLO(model_path)
    class_names = list(
        model.names.values()
    )  # Convert dictionary to list of class names
st.success("Model loaded successfully!", icon="✅")

# Content
content.content_yolov8_classify_webcam()


col1, col2 = st.columns(2)

with col1:
    st.markdown("## 📷 Webcam Feed")
    video_placeholder = st.empty()
    col11, col12 = st.columns([1, 1])
    with col11:
        run_button = st.button(
            ":green[Run]", type="secondary", use_container_width=True
        )
    with col12:
        stop_button = st.button("Stop", type="primary", use_container_width=True)

with col2:
    st.markdown("## 📊 Classification Results")
    bar_chart_placeholder = st.empty()
    line_chart_placeholder = st.empty()

# Initialize probability history
class_names = list(model.names.values())
prob_history = {class_name: [] for class_name in class_names}

if run_button:
    with st.spinner("Open webcam..."):
        cap = cv2.VideoCapture(0)  # Use 0 for default webcam
    if not cap.isOpened():
        st.error("Could not open webcam.")

    while not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to capture frame from webcam.")
            break

        start_time = time.time()

        # Perform prediction
        results = model.predict(frame)
        annotated_frame = results[0].plot()  # Get the annotated frame

        # Update video feed with annotated frame
        video_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)

        # Calculate and display FPS
        fps = 1 / (time.time() - start_time)
        fps_placeholder.metric("FPS", f"{fps:.2f}")

        # Extract probabilities
        probabilities = results[0].probs.data.tolist()

        # Update bar chart
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
        for class_name, prob in zip(class_names, probabilities):
            prob_history[class_name].append(prob)
            if len(prob_history[class_name]) > 50:  # Keep last 50 frames
                prob_history[class_name].pop(0)

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

        time.sleep(0.05)  # Add a small delay to reduce computational load

    cap.release()
    st.success("Webcam stream ended.")

# Clear CUDA memory
torch.cuda.empty_cache()
