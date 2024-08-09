import streamlit as st
from ultralytics import YOLO
from PIL import Image
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
content.content_yolov8_classify()

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
results = model.predict(image)

# Get classification results
summary = results[0].summary()
probs = results[0].probs.data.tolist()
names = results[0].names
speeds = results[0].speed  # Speed object for inference times
original_shape = results[0].orig_shape  # Original image shape

with col2:
    # Display results
    st.markdown("## 📊 Classification Results")
    annotated_img = results[0].plot()  # plot a BGR numpy array of predictions
    st.image(
        annotated_img,
        channels="BGR",
        caption="Classification Results | Classification Model Output",
        use_column_width="auto",
    )

# Extract class names and probabilities
class_names = list(names.values())

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

# Display the plot in Streamlit
st.markdown("## 📊 Bar chart of classification results")
st.plotly_chart(fig, use_container_width=True)

st.divider()

md_prediciton = f"""
## 🗂️ Model Prediction
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
