import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go

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
content.content_yolov8_classify_photo()

# Create two columns for display
col1, col2 = st.columns(2)

# Camera input
with col1:
    st.markdown("## 📸 Camera Input")
    img_file_buffer = st.camera_input("Take a picture", label_visibility="collapsed")

with col2:
    st.markdown("## 📊 Classification Results")

if img_file_buffer is not None:
    # To read image file buffer as a numpy array:
    bytes_data = img_file_buffer.getvalue()
    cv2_image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    with col2:
        # Perform classification
        results = model(cv2_image)

        # Get classification results
        summary = results[0].summary()
        probs = results[0].probs.data.tolist()
        names = results[0].names
        speeds = results[0].speed
        original_shape = results[0].orig_shape

        # Display results
        annotated_img = results[0].plot()
        st.image(annotated_img, channels="BGR", use_column_width=True)

    # Create a DataFrame for Plotly
    df = pd.DataFrame({"Class": class_names, "Probability": probs})

    # Create Plotly figure
    fig = go.Figure(
        data=[
            go.Bar(
                x=df["Class"],
                y=df["Probability"],
                text=df["Probability"].apply(lambda x: f"{x:.4f}"),
                textposition="outside",
                textfont=dict(size=18),
                hoverinfo="text",
                hovertext=[
                    f"Class: {cls}<br>Probability: {prob:.4f}"
                    for cls, prob in zip(df["Class"], df["Probability"])
                ],
            )
        ]
    )

    # Update layout with larger font sizes
    fig.update_layout(
        title={
            "text": "Class Probabilities",
            "font": {"size": 20},
        },
        xaxis_title="Class",
        yaxis_title="Probability",
        xaxis={
            "title": {"font": {"size": 24}},
            "tickfont": {"size": 18},
        },
        yaxis={
            "title": {"font": {"size": 24}},
            "tickfont": {"size": 18},
            "range": [0, 1],
        },
        width=800,
        height=500,
        font=dict(size=14),
    )

    # Display the plot
    st.markdown("## 📊 Classification Results in bar plot:")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    md_prediciton = f"""
    ## 🗂️ Model Prediction:
    ## Model predicts the class **:red[{summary[0]['name']}]** with a probability of **:red[{summary[0]['confidence']:.2f}]**
    """

    st.markdown(md_prediciton)

    st.divider()

    utils.show_inference(speeds=speeds)

    st.divider()

    utils.show_data_objects(
        speeds=speeds,
        original_shape=original_shape,
        probabilities=probs,
        class_names=class_names,
    )


else:
    st.info("Waiting for camera input. Please take a picture. 📸")
