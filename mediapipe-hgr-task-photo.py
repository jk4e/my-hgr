import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from PIL import Image
import cv2

# Local Modules
import content
import utils
import plotting

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


@st.cache_resource
def load_model(model_path, num_hands=1):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=VisionRunningMode.IMAGE,
        num_hands=num_hands,
    )
    return vision.GestureRecognizer.create_from_options(options)


# Load the model
model_path, num_hands = utils.mediapipe_sidebar_options()
recognizer = load_model(model_path, num_hands)

# Content
content.content_mediapipe_hgr_task_photo()

# Create three columns
col1, col2, col3 = st.columns(3)

# Camera input
with col1:
    st.markdown("## 📸 Camera Input")
    img_file_buffer = st.camera_input("Take a picture", label_visibility="collapsed")

with col2:
    st.markdown("## 📊 Annotated Image")

with col3:
    st.markdown("## 📈 3D Visualization")

if img_file_buffer is not None:
    # To read image file buffer as a numpy array:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # Convert BGR to RGB
    image_np = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)

    # Run inference
    recognition_result = recognizer.recognize(mp_image)

    # Display annotated image in column 2
    with col2:
        st.markdown("## Annotated Image")
        annotated_image = plotting.draw_landmarks_on_image(image_np, recognition_result)
        st.image(annotated_image, caption="Annotated Image", use_column_width=True)

    # Display 3D plot in column 3
    with col3:
        st.markdown("## 3D Visualization")
        if recognition_result.gestures and recognition_result.hand_landmarks:
            results = [
                (recognition_result.gestures[0][0], recognition_result.hand_landmarks)
            ]
            fig = plotting.plot_hand_landmarks_3d_plotly(results)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No hand landmarks detected for 3D visualization.")
        else:
            st.write("No gestures or hand landmarks detected for 3D visualization.")

    # Display gesture recognition results
    st.divider()
    st.markdown("## 🗂️ Model Prediction")
    st.markdown("### Hand Gestures:")
    if recognition_result.gestures:
        for hand_index, gestures in enumerate(recognition_result.gestures):
            for gesture in gestures:
                st.markdown(
                    f"### ✋ Hand {hand_index + 1}: Model predicts the Class/Gesture **:red[{gesture.category_name}]** with a Probability/Score of **:red[{gesture.score:.2f}]**"
                )
    else:
        st.write("No gestures detected")

    # Display handedness
    st.subheader("Handedness:")
    if recognition_result.handedness:
        for hand_index, handedness in enumerate(recognition_result.handedness):
            for category in handedness:
                st.markdown(
                    f"### Hand {hand_index + 1}: **:orange[{category.category_name}]** hand (Score: **:orange[{category.score:.2f}]**)"
                )
    else:
        st.write("No hands detected")

else:
    st.info("Waiting for camera input. Please take a picture.📸")
