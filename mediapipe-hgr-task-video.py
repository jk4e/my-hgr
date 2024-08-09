import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import tempfile
import os
import time

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
        running_mode=VisionRunningMode.VIDEO,
        num_hands=num_hands,
    )
    return vision.GestureRecognizer.create_from_options(options)


# Sidebar
model_path, num_hands = utils.mediapipe_sidebar_options()

# Init recognizer
recognizer = load_model(model_path, num_hands)

# Content
content.content_mediapipe_hgr_task_video()

# Video selection
video_path = utils.video_selector()

col1, col2 = st.columns(2)

with col1:
    # Create a placeholder for the video
    video_placeholder = st.empty()

with col1:
    st.markdown("## 🗂️ Model Prediction:")
    st.markdown("### Hand Gestures:")

# Create a placeholder for the gesture text outside the video processing loop
gesture_text = st.empty()

if video_path is not None:

    # Check if video_path is a string (file path) or a file-like object
    if isinstance(video_path, str):
        # It's a file path, so we can use it directly
        cap = cv2.VideoCapture(video_path)
    else:
        # It's a file-like object, so we need to save it to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_path.read())
        tfile.close()
        cap = cv2.VideoCapture(tfile.name)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Process the video
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Convert the frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Run inference
        recognition_result = recognizer.recognize_for_video(
            mp_image, int(time.time() * 1000)
        )

        # Draw landmarks and gestures on the frame
        annotated_frame = plotting.draw_landmarks_on_image(
            frame_rgb, recognition_result
        )

        # Display the annotated frame
        with col1:
            video_placeholder.image(
                annotated_frame, channels="RGB", use_column_width=True
            )

        with col1:
            # Display gesture recognition results
            if recognition_result.gestures:
                gesture_info = []
                for hand_index, gestures in enumerate(recognition_result.gestures):
                    for gesture in gestures:
                        gesture_info.append(
                            f"### ✋ Hand {hand_index + 1}: Model predicts the Class/Gesture **:red[{gesture.category_name}]** "
                            f"with a Probability/Score of **:red[{gesture.score:.2f}]**"
                        )

                all_gestures = "\n\n".join(gesture_info)

                # Update the placeholder with the new text
                gesture_text.markdown(all_gestures)
            else:
                # If no gestures are detected
                gesture_text.markdown("No gestures detected")
    cap.release()

    # If created a temporary file, remove
    if not isinstance(video_path, str):
        os.unlink(tfile.name)

else:
    st.write("Please upload a video file.")
