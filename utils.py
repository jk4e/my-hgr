import streamlit as st


def image_selector() -> str:
    """
    Select an image from the predefined list or upload a new image.

    Returns:
        str: The path of the selected image.
    """
    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload an image...", type=["jpg", "jpeg", "png", "webp", "bmp"]
    )

    # Predefined images
    image_list = [
        "hand1.jpg",
        "hand2.jpg",
        "hand3.jpg",
        "hand4.jpg",
        "hand5.jpg",
        "hand6.jpg",
        "hand7.jpg",
        "hand8.jpg",
        "hand9.jpg",
        "hand10.jpg",
        "hand11.jpg",
        "hand12.jpg",
        "hand13.jpg",
        "hand14.jpg",
        "hand15.jpg",
        "hand16.jpg",
        "ai-hand1.jpg",
        "ai-hand2.jpg",
        "ai-hand3.jpg",
        "ai-hand4.jpg",
        "ai-hand5.jpg",
        "ai-hand6.jpg",
        "ai-hand7.jpeg",
        "background1.jpg",
        "background2.jpg",
        "background3.jpg",
        "background4.jpg",
        "background5.jpg",
        "graphic1.jpg",
        "graphic2.jpg",
        "graphic3.jpg",
        "person1.jpg",
        "person2.jpg",
        "person3.jpg",
        "person4.jpg",
        "person5.jpg",
    ]

    selected_image = st.sidebar.selectbox("Select an predefined image", image_list)

    if uploaded_file is None:
        uploaded_file = f"images/{selected_image}"

    return uploaded_file


def video_selector() -> str:
    """
    Select a video from the predefined list or upload a new video.

    Returns:
        str: The path of the selected video.
    """
    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload a video...", type=["mp4", "avi", "mov"]
    )

    # Predefined videos
    video_list = [
        "video1.mp4",
        "video2.mp4",
        "video3.mp4",
        "video4.mp4",
        "video5.mp4",
        "video6.mp4",
        "video7.mp4",
        "video8.mp4",
        "video9.mp4",
        "video10.mp4",
        "video11.mp4",
        "video12.mp4",
        "video13.mp4",
    ]

    selected_video = st.sidebar.selectbox("Select a video", video_list)

    if uploaded_file is None:
        video_path = f"videos/{selected_video}"
    else:
        video_path = uploaded_file

    return video_path


def yolo_classify_sidebar_options() -> tuple:
    """
    Sidebar options for YOLO model selection and confidence threshold.

    Returns:
        tuple: The selected model type and confidence threshold.
    """
    st.sidebar.header("Model Configuration")

    # Model Options
    model_type = st.sidebar.radio(
        "Model (more ℹ️ see Home)",
        [
            "RPS✔️",
            "ASL Count✔️",
            "ASL Alphabet✔️",
            "HaGRID✔️",
            "Hands✔️",
        ],
    )

    # Selecting model path based on model type
    model_paths = {
        "RPS✔️": "models/rps_classify.pt",
        "ASL Count✔️": "models/asl_count_classify.pt",
        "ASL Alphabet✔️": "models/asl_alphabet_classify.pt",
        "HaGRID✔️": "models/hagrid_classify.pt",
        "Hands✔️": "models/hand_classify.pt",
    }
    model_path = model_paths[model_type]

    return model_path


def yolo_detect_sidebar_options() -> tuple:
    """
    Sidebar options for YOLO model selection and confidence threshold.

    Returns:
        tuple: The selected model type and confidence threshold.
    """
    st.sidebar.header("Model Configuration")

    # Model Options
    model_type = st.sidebar.radio(
        "Model (more ℹ️ see Home)",
        [
            "RPS✔️",
            "ASL Alphabet✔️",
            "AI Open Palm✔️",
            "AI Hands✔️",
            "AI Hands2✔️",
        ],
    )
    confidence = float(
        st.sidebar.slider(
            "Confidence Threshold (default is 0.25)", 0.0, 1.0, 0.25, 0.01
        )
    )

    iou = float(
        st.sidebar.slider("IoU Threshold (default is 0.7)", 0.0, 1.0, 0.7, 0.01)
    )

    # Selecting model path based on model type
    model_paths = {
        "RPS✔️": "models/rps_detect.pt",
        "ASL Alphabet✔️": "models/asl_alphabet_detect.pt",
        "AI Open Palm✔️": "models/ai_open_palm_detect",
        "AI Hands✔️": "models/ai_hands_detect.pt",
        "AI Hands2✔️": "models/ai_hands2_detect.pt",
    }
    model_path = model_paths[model_type]

    return model_path, confidence, iou


def mediapipe_sidebar_options() -> tuple:
    """
    Sidebar options for YOLO model selection and confidence threshold.

    Returns:
        tuple: The selected model type and confidence threshold.
    """
    st.sidebar.header("Model Configuration")

    # Model Options
    model_type = st.sidebar.radio(
        "Model (more ℹ️ see Home)",
        [
            "DEFAULT✔️",
            "RPS✔️",
            "ASL Count✔️",
            "ASL Alphabet✔️",
        ],
    )

    # Selecting model path based on model type
    model_paths = {
        "DEFAULT✔️": "models/gesture_recognizer.task",
        "RPS✔️": "models/rps.task",
        "ASL Count✔️": "models/asl_count.task",
        "ASL Alphabet✔️": "models/asl_alphabet.task",
    }
    model_path = model_paths[model_type]

    num_hands = int(
        st.sidebar.slider("The maximum number of hands to detect", 0, 10, 1, 1)
    )

    return model_path, num_hands


def show_inference(speeds) -> None:
    md_inference_speed = f"""
    ## ⏱️ Inference timings / Speed in ms
    - Preprocess: :blue-background[{speeds['preprocess']:.2f}] ms
    - Inference: :blue-background[{speeds['inference']:.2f}] ms
    - Postprocess: :blue-background[{speeds['postprocess']:.2f}] ms
    """
    st.markdown(md_inference_speed)


def show_data_objects(
    speeds=None, original_shape=None, probabilities=None, class_names=None
) -> None:
    st.markdown("## 🗂️ Data Objects")
    if class_names is not None:
        st.markdown("### Class Names:")
        st.write(class_names)
    if probabilities is not None:
        st.markdown("### Probabilities:")
        st.write(probabilities)
    if speeds is not None:
        st.markdown("### Speed:")
        st.write(speeds)
    if original_shape is not None:
        st.markdown("### Original image shape (height, width):")
        st.write(original_shape)
    pass
