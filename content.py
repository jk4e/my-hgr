import streamlit as st


# -- Warnings -- #
def preview_warning() -> None:
    st.warning(
        "Streamlit app is in Preview (early stage of development). Not all models are currently available (marked with ✔️, ❌). More models will be added soon. Stay tuned! 📺",
        icon="⚠️",
    )
    pass


# -- Home Page -- #
def content_home() -> None:
    md_content = """
    # :rainbow[Custom Hand Gesture Recognition Models (HGR)]🖐️ with :rainbow[_YOLOv8_]🚀 and :rainbow[_MediaPipe_]👋
    Author: [jk4e](https://github.com/jk4e) | Last Update: August 2024

    ## Introduction
    This is a collection of demos as a Streamlit app showcasing the capabilities of custom-trained HGR models on freely available SDKs and frameworks for various HGR tasks. Select a model configuration from the sidebar to try out the different models and tasks.
    Have fun exploring!

    ## Features
    - Support for Images🖼️, Videos🎞️, and Webcam📷 Photos and Video Streams
    - Custom YOLOv8 detection model for hand detection task (HGR)
    - Custom YOLOv8 classification model for hand classification task (HGR)
    - Custom MediaPipe hand gesture recognition task (HGR)

    ## How to Use
    1. Select the model and task from the sidebar
    2. Upload or choose an image, video, or use your webcam to start the model prediction
    3. See the model predictions and information in the main window
    4. Play around with the different models and confidence thresholds to see how they affect the model predictions, inference time, and accuracy. Have fun!🎉

    ## Models and Tasks
    - **YOLOv8 Hand Detection**
    - **YOLOv8 Hand Classification**
    - **MediaPipe Hand Gesture Recognition (HGR)**

    ## Model Configuration Sidebar
    - Choose from the available pre-trained models and adjust the model parameters if available

    ## Background
    The goal of this project is to showcase the capabilities of custom-trained HGR models on freely available SDKs and frameworks for various HGR tasks. The project demonstrates solutions using computer vision and machine learning to build and customize HGRs for diverse applications.

    ## About
    Research revealed a lack of freely available HGR models and SDKs for custom HGR tasks. This project aims to provide a solution by creating a collection of custom-trained HGR models using freely available SDKs and frameworks for various HGR tasks.
    The solutions found were **Ultralytics YOLOv8** and **Google MediaPipe** SDKs. These are relatively easy to use, with a low-code and high-level abstraction approach, and have the possibility of customizing gesture recognition models. Both are available in Python 🐍 and have good documentation and community support.
    The models were trained on custom datasets by collecting our own images, using freely available datasets from platforms like Roboflow, Kaggle, GitHub, and Google Dataset Search, as well as a new approach of training with AI-generated images from image generators like Stable Diffusion and DALL-E.
    The models were trained on as much diverse data as possible, but at relatively low resolutions on local machines and Google Colab instances with GPUs in the free plan. The following steps were taken to train the models:

    1. Data collection: own images, freely available datasets, AI-generated images
    2. Labeling: label images with bounding boxes and classes if necessary using the free tool CVAT.ai
    3. Data preprocessing: image resizing, splitting into train/val/test sets, creating necessary folder structure for training
    4. Data augmentation: flip, rotate, crop, resize, color jitter, blur, noise, etc. (if not already done in the model training pipeline itself)
    5. Model training: train the model on the custom dataset with the selected model architecture and default parameters. Then try some hyperparameter tuning to improve model performance. Use of pre-trained models for transfer learning was applied.
    6. Model evaluation: evaluate the model on the test set and calculate performance metrics
    7. Model testing: test the model on new images and videos to see how it performs in real-world scenarios and how easy it is to use for end-users
    8. Model deployment

    In this process, various challenges were faced and had to be solved. For training and evaluation, custom scripts and Jupyter notebooks were created.
    """
    preview_warning()

    st.image("assets/banner.jpg", use_column_width=True)

    st.markdown(md_content)

    st.info(
        "The models were trained using a combination of data sets. To respect privacy and licenses, the sources of the training data cannot be published. To achieve good results, the models were trained on as much diverse data as possible.",
        icon="ℹ️",
    )

    asl_manual_link = "https://en.wikipedia.org/wiki/American_manual_alphabet"

    md_models = f"""
    ## Available models for hand gestures:
    - **RPS**: :orange[Rock]✊, :green[Paper]✋, :blue[Scissors]✌️ (famous game)
    - **ASL Count**: Digits, see [ASL/American manual alphabet]({asl_manual_link})
    - **ASL Alphabet**: Letters, see [ASL/American manual alphabet]({asl_manual_link}) (Letters Z and J are not included because they require dynamic movements)
    - **Default**: Victory, Closed Fist, Thumbs Up, Thumbs Down, Open Palm, I Love you, Pointing Up (👍, 👎, ✌️, ☝️, ✊, 👋, 🤟)(common hand gestures)
    - **HaGRID**: see [HaGRID - HAnd Gesture Recognition Image Dataset](https://github.com/hukenovs/hagrid)
    - **Hand**: detect if in imaga a hand is present or not


    ## Model Information (TODO: Section is not yet complete)
    - Num. of Images: all images for model training. Include train, validation, and test sets.

    ### YOLOv8 Classify:
    | Model    | Available| Test Accuracy | Model Input Shape | Pretrained on | Size (MB) | Num. of Images | Num. of Classes | None/Background class | Class names |
    | -------- | -------- | ------------- | ----------------- | ------------- | --------- | ------------- | --------------- | --------------------- | ----------- |
    | RPS      | ✔️       | -             | (1, 3, 224, 224)       |               |        |           | 4               | ✔️                   | `rock`, `paper`, `scissors`, `background`|
    | ASL Count| ✔️       | -             | (1, 3, 224, 224)       |               |        |           | 11              | ✔️                   | `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `background`|
    | ASL Alphabet| ✔️    | -             | (1, 3, 224, 224)       |               |        |           | 25              | ✔️                   | `A`, `B`, `C`, `D`, `E`, `F`, `G`, `H`, `I`, `K`, `L`, `M`, `N`, `O`, `P`, `Q`, `R`, `S`, `T`, `U`, `V`, `W`, `X`, `Y`, `background`|

    ### YOLOv8 Detect:

    ### MediaPipe HGR Task:
    | Model    | Available | Test Accuracy | Model Input Shape | Pretrained on | Size (MB) | Num. of Images | Num. of Classes | None/Background class | Class names |
    | -------- | --------- | ------------- | ----------------- | ------------- | --------- | ------------- | --------------- | --------------------- | ----------- |
    | RPS      | ✔️        | 0.90          | 224 x 224         | -             | 8.3       | 8,100          | 4               | ✔️                   | `rock`, `paper`, `scissors`, `none`|
    | ASL Count| ✔️        | 0.98          | 224 x 224         | -             | 8.3       | 7,100          | 10              | ❌                   | `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `none`|
    | ASL Alphabet| ✔️     | 0.96          | 224 x 224         | -             | 8.3       | 8,400          | 24              | ❌                   | `A`, `B`, `C`, `D`, `E`, `F`, `G`, `H`, `I`, `K`, `L`, `M`, `N`, `O`, `P`, `Q`, `R`, `S`, `T`, `U`, `V`, `W`, `X`, `Y`, `none`|
    | Default*  | ✔️        | -             | 192 x 192, 224 x 224 | -          | 8.3       | 3000           | 8               | -                     | `Unknown`, `Closed_Fist`, `Open_Palm`, `Pointing_Up`, `Thumb_Down`, `Thumb_Up`, `Victory`, `ILoveYou`|

    *Default model is provided by MediaPipe (Model name: `HandGestureClassifier` or `Default` (in MP Studio )). For more information see [Model card for Hand Gesture Classification](https://storage.googleapis.com/mediapipe-assets/gesture_recognizer/model_card_hand_gesture_classification_with_faireness_2022.pdf)



    """
    st.markdown(md_models)

    st.info(
        "Note: The class label for unclassified hand gestures should be `background` or `None` as classified by the model",
        icon="ℹ️",
    )

    st.warning(
        "Note: not all models are currently available for every task or even available at all",
        icon="⚠️",
    )

    st.error(
        "The trained models do not currently achieve the anticipated level of accuracy in real-world applications.The objective of the app is to illustrate the fundamental concepts and potential applications. The comparatively poor accuracy and generational control of the Yolo model in particular is likely the result of a number of factors, including an insufficient quantity of annotated training data, erroneous data, a lack of diversity in the data set, and a partial lack of data augmentation. However, the most significant challenge lies in the limited hardware resources available. The project encountered several obstacles, including the lack of a powerful GPU with CUDA for training (free plan Google Colab), insufficient RAM, and other hardware limitations. These constraints hindered the ability to perform hyperparameter tuning and model customization and also to train the models long enough (300 epochs were suggested as a starting point; training was only conducted for 10-50 epochs). Additionally, the limited training time necessitated reducing the image size to 224x224.",
        icon="⚠️",
    )

    st.divider()

    md_description = """
    ## Description:
    Upload your own image or select one of the predefined images to start classifying hands.
    The model will classify the hand gesture.🔎 Most of the models have only been trained on images with a visible hand, so make sure your uploaded image meets this requirement.

    ## Results:
    The model will output the results with a class label and a confidence value, sometimes with an additional bar chart with all class names and probabilities. 
    A probability of 1 or close to 1 indicates a high confidence of the model in classifying a particular class.
    If the results are more spread over the classes, the model is not as confident about the prediction.
    The model also outputs the inference times in milliseconds for each step (preprocess, inference, postprocess).
    """
    st.markdown(md_description)

    pass


# -- YOLOv8 Classify Page -- #
def content_yolov8_classify() -> None:
    md_content = """
    # :rainbow[_YOLOv8_]🚀 Hand Classification🖐️
    Author: [jk4e](https://github.com/jk4e) | Last Update: August 2024  
    """
    preview_warning()
    st.markdown(md_content)
    st.divider()
    pass


# -- YOLOv8 Classify Photo Page -- #
def content_yolov8_classify_photo() -> None:
    md_content = """
    # :rainbow[_YOLOv8_]🚀 Hand Classification Photo (Webcam)🖐️
    Author: [jk4e](https://github.com/jk4e) | Last Update: August 2024  
    """
    preview_warning()
    st.markdown(md_content)
    st.divider()
    pass


# -- YOLOv8 Classify Video Page -- #
def content_yolov8_classify_video() -> None:
    md_content = """
    # :rainbow[_YOLOv8_]🚀 Hand Classification Video🖐️
    Author: [jk4e](https://github.com/jk4e) | Last Update: August 2024  
    """
    preview_warning()
    st.markdown(md_content)
    st.divider()
    pass


# -- YOLOv8 Classify Video Realtime Processing Page -- #
def content_yolov8_classify_video_realtime() -> None:
    md_content = """
    # :rainbow[_YOLOv8_]🚀 Hand Classification Video with Real-Time Processing🖐️  
    Author: [jk4e](https://github.com/jk4e) | Last Update: August 2024  
    """
    preview_warning()
    st.markdown(md_content)
    st.warning(
        "This demo will be lagging due to the realtime processing of each frame with the model on the CPU",
        icon="⚠️",
    )
    st.divider()
    pass


# -- YOLOv8 Classify Webcam Page -- #
def content_yolov8_classify_webcam() -> None:
    md_content = """
    # :rainbow[_YOLOv8_]🚀 Hand Classification Webcam📷  
    Author: [jk4e](https://github.com/jk4e) | Last Update: August 2024 
    """
    preview_warning()
    st.markdown(md_content)
    st.warning(
        "This demo will may be lagging due to the realtime processing of each frame with the model on the CPU",
        icon="⚠️",
    )
    st.warning(
        "On **Windows** open the Webcam may take a while. Wait more than 1 minute if webcam stream do not show up",
        icon="⚠️",
    )
    st.divider()
    pass


# -- YOLOv8 Detect Page -- #
def content_yolov8_detect() -> None:
    md_content = """
    # :rainbow[_YOLOv8_]🚀 Hand Detection🖐️  
    Author: [jk4e](https://github.com/jk4e) | Last Update: August 2024  
    """
    preview_warning()
    st.markdown(md_content)
    st.divider()
    pass


# -- YOLOv8 Detect Photo Page -- #
def content_yolov8_detect_photo() -> None:
    md_content = """
    # :rainbow[_YOLOv8_]🚀 Hand Detection Photo🖐️   
    Author: [jk4e](https://github.com/jk4e) | Last Update: August 2024  
    """
    preview_warning()
    st.markdown(md_content)
    st.divider()
    pass


# -- YOLOv8 Detect Video Page -- #
def content_yolov8_detect_video() -> None:
    md_content = """
    # :rainbow[_YOLOv8_]🚀 Hand Detection Video🖐️   
    Author: [jk4e](https://github.com/jk4e) | Last Update: August 2024  
    """
    preview_warning()
    st.markdown(md_content)
    st.divider()
    pass


# -- YOLOv8 Detect Webcam Page -- #
def content_yolov8_detect_webcam() -> None:
    md_content = """
    # :rainbow[_YOLOv8_]🚀 Hand Detection Webcam Video🖐️  
    Author: [jk4e](https://github.com/jk4e) | Last Update: August 2024  
    """
    preview_warning()
    st.markdown(md_content)
    st.warning(
        "Note: In the demo, Due to non-availability of GPUs, you may encounter slow video inferencing and lagging frames.",
        icon="⚠️",
    )
    st.warning(
        "On **Windows** open the Webcam may take a while. Wait more than 1 minute if webcam stream do not show up",
        icon="⚠️",
    )
    st.divider()
    pass


# -- MediaPipe HGR Task Page -- #
def content_mediapipe_hgr_task() -> None:
    md_content = """
    # :rainbow[_MediaPipe_]🚀 Hand Gesture Recognition (HGR)🖐️  
    Author: [jk4e](https://github.com/jk4e) | Last Update: August 2024  
    """
    preview_warning()
    st.markdown(md_content)
    st.divider()
    pass


# -- MediaPipe HGR Task Photo Page -- #
def content_mediapipe_hgr_task_photo() -> None:
    md_content = """
    # :rainbow[_MediaPipe_]🚀 HGR Photo 🖐️  
    Author: [jk4e](https://github.com/jk4e) | Last Update: August 2024  
    """
    preview_warning()
    st.markdown(md_content)
    st.divider()
    pass


# -- MediaPipe HGR Task Video Page -- #
def content_mediapipe_hgr_task_video() -> None:
    md_content = """
    # :rainbow[_MediaPipe_]🚀 HGR Video🖐️  
    Author: [jk4e](https://github.com/jk4e) | Last Update: August 2024  
    """
    preview_warning()
    st.markdown(md_content)
    st.divider()
    pass


# -- MediaPipe HGR Task Webcam Page -- #
def content_mediapipe_hgr_task_webcam() -> None:
    md_content = """
    # :rainbow[_MediaPipe_]🚀 HGR Webcam🖐️  
    Author: [jk4e](https://github.com/jk4e) | Last Update: August 2024  
    """
    preview_warning()
    st.markdown(md_content)
    st.divider()
    pass
