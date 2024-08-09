import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

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
content.content_yolov8_detect_photo()

# Class selection
selected_classes = st.sidebar.multiselect(
    "Classes", class_names, default=class_names[:3]
)
selected_ind = [class_names.index(option) for option in selected_classes]

# Create two columns for display
col1, col2 = st.columns(2)


# Camera input
with col1:
    st.markdown("## 📸 Camera Input")
    img_file_buffer = st.camera_input("Take a picture", label_visibility="collapsed")

with col2:
    st.markdown("## Annotated Image")

if img_file_buffer is not None:
    # To read image file buffer as a numpy array:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    with col2:
        # Perform detection
        results = model(cv2_img, conf=confidence, iou=iou, classes=selected_ind)

        # Plot results
        annotated_img = results[0].plot()

        # Display annotated image
        st.image(annotated_img, channels="BGR", use_column_width=True)

    # Display detection information
    st.markdown("### Detection Results")
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            st.write(f"{class_name}: {confidence:.2f}")

else:
    st.info("Waiting for camera input. Please take a picture.📸")
