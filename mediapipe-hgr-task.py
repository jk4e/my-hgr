import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from PIL import Image

# Local Modules
import content
import utils
import plotting

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# Set up MediaPipe
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
content.content_mediapipe_hgr_task()

# Image selection
uploaded_file = utils.image_selector()

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)

    # Run inference
    recognition_result = recognizer.recognize(mp_image)

    # Create two columns
    col1, col2, col3 = st.columns(3)

    # Display input image in column 1
    with col1:
        st.markdown("## 🖼️ Input Image")
        st.image(image, caption="Input Image", use_column_width=True)

    # Display annotated image in column 2
    with col2:
        st.markdown("## 📊 Annotated Image")
        annotated_image = plotting.draw_landmarks_on_image(image_np, recognition_result)
        st.image(annotated_image, caption="Annotated Image", use_column_width=True)

    # Display 3D plot in column 3
    with col3:
        st.markdown("## 📊 3D Hand Landmarks")
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

    st.divider()
    # Display gesture recognition results
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
                    f"### ✋ Hand {hand_index + 1}: **:orange[{category.category_name}]** hand (Score: **:orange[{category.score:.2f}]**)"
                )
    else:
        st.write("No hands detected")
