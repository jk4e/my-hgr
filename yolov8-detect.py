import streamlit as st
from ultralytics import YOLO
from PIL import Image

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
content.content_yolov8_detect()

# Image selection
uploaded_file = utils.image_selector()

col1, col2 = st.columns(2)

with col1:
    # Display the uploaded image
    st.markdown("## 🖼️ Input Image")
    st.image(uploaded_file, caption="Uploaded Image | Input", use_column_width="auto")

    # Open the image using PIL
    image = Image.open(uploaded_file)

# Perform prediction
results = model.predict(image, conf=confidence, iou=iou)
# summary = results[0].summary()

with col2:
    # Display results
    st.markdown("## 📊 Detection Results")
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        st.image(
            im_array,
            channels="BGR",
            caption="Detection Results | Detection Model Output",
            use_column_width="auto",
        )

# Process results and create bar chart
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    probs = result.probs  # Probs object for classification outputs
    speeds = result.speed  # Speed object for inference times
    original_shape = result.orig_shape  # Original image shape

    if boxes is not None:

        # TODO: better way to display the results in a dynamic string format
        # st.markdown("## 🗂️ Model Prediction:")
        # st.markdown(
        #     f"## Model predicts the class **:red[{summary[0]['name']}]** with a probability of **:red[{summary[0]['confidence']:.2f}]**"
        # )

        st.divider()

        utils.show_inference(speeds=speeds)

        st.divider()

        utils.show_data_objects(speeds=speeds, original_shape=original_shape)

    else:
        st.write("No detections found in this image.")
        st.info("Maybe try another image or adjust the confidence threshold", icon="ℹ️")
