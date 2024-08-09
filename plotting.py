import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import cv2
import numpy as np
from PIL import Image
import time
import sys
import plotly.graph_objects as go

# Set up MediaPipe for drawing landmarks
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Define your color tuples in RGB
_RED = "rgb(255, 48, 48)"
_GREEN = "rgb(48, 255, 48)"
_BLUE = "rgb(21, 101, 192)"
_YELLOW = "rgb(255, 204, 0)"
_GRAY = "rgb(128, 128, 128)"
_PURPLE = "rgb(128, 64, 128)"
_PEACH = "rgb(255, 229, 180)"
_WHITE = "rgb(224, 224, 224)"
_CYAN = "rgb(48, 255, 192)"
_MAGENTA = "rgb(255, 48, 192)"

# Define a list of colors for each keypoint
colors = [
    _RED,
    _RED,
    _PEACH,
    _PEACH,
    _PEACH,
    _RED,
    _PURPLE,
    _PURPLE,
    _PURPLE,
    _RED,
    _YELLOW,
    _YELLOW,
    _YELLOW,
    _RED,
    _GREEN,
    _GREEN,
    _GREEN,
    _RED,
    _BLUE,
    _BLUE,
    _BLUE,
]

# Hand landmarks
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_FINGER_MCP = 5
INDEX_FINGER_PIP = 6
INDEX_FINGER_DIP = 7
INDEX_FINGER_TIP = 8
MIDDLE_FINGER_MCP = 9
MIDDLE_FINGER_PIP = 10
MIDDLE_FINGER_DIP = 11
MIDDLE_FINGER_TIP = 12
RING_FINGER_MCP = 13
RING_FINGER_PIP = 14
RING_FINGER_DIP = 15
RING_FINGER_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

# Landmark groups
_PALM_LANDMARKS = [
    WRIST,
    INDEX_FINGER_MCP,
    MIDDLE_FINGER_MCP,
    RING_FINGER_MCP,
    PINKY_MCP,
    WRIST,
]
_PALM_LANDMARKS_2 = [WRIST, THUMB_CMC]
_THUMB_LANDMARKS = [THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP]
_INDEX_FINGER_LANDMARKS = [
    INDEX_FINGER_MCP,
    INDEX_FINGER_PIP,
    INDEX_FINGER_DIP,
    INDEX_FINGER_TIP,
]
_MIDDLE_FINGER_LANDMARKS = [
    MIDDLE_FINGER_MCP,
    MIDDLE_FINGER_PIP,
    MIDDLE_FINGER_DIP,
    MIDDLE_FINGER_TIP,
]
_RING_FINGER_LANDMARKS = [
    RING_FINGER_MCP,
    RING_FINGER_PIP,
    RING_FINGER_DIP,
    RING_FINGER_TIP,
]
_PINKY_FINGER_LANDMARKS = [PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP]


def draw_landmarks_on_image(image, recognition_result):
    annotated_image = image.copy()

    if recognition_result.hand_landmarks:

        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        for hand_landmarks in recognition_result.hand_landmarks:
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(
                        x=landmark.x, y=landmark.y, z=landmark.z
                    )
                    for landmark in hand_landmarks
                ]
            )
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    return annotated_image


def plot_hand_landmarks_3d_plotly(results):
    for gesture, hand_landmarks_list in results:
        if hand_landmarks_list and len(hand_landmarks_list) > 0:
            # Extract x, y, z coordinates from the first (and only) list of landmarks
            landmarks = hand_landmarks_list[0]
            x = [landmark.x for landmark in landmarks]
            y = [landmark.y for landmark in landmarks]
            z = [landmark.z for landmark in landmarks]

            # Create the 3D scatter plot with custom colors
            scatter = go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(size=10, color=colors, opacity=1),
                text=[
                    f"Landmark {i}<br>X: {x[i]:.2f}<br>Y: {y[i]:.2f}<br>Z: {z[i]:.2f}"
                    for i in range(len(x))
                ],
                hoverinfo="text",
            )

            # Function to draw lines between landmarks
            def draw_lines(landmarks, color):
                line_data = []
                for i in range(len(landmarks) - 1):
                    line_data.append(
                        go.Scatter3d(
                            x=[x[landmarks[i]], x[landmarks[i + 1]]],
                            y=[y[landmarks[i]], y[landmarks[i + 1]]],
                            z=[z[landmarks[i]], z[landmarks[i + 1]]],
                            mode="lines",
                            line=dict(color=color, width=10),
                        )
                    )
                return line_data

            # Draw lines for each group
            line_traces = []
            line_traces.extend(draw_lines(_PALM_LANDMARKS, _GRAY))
            line_traces.extend(draw_lines(_PALM_LANDMARKS_2, _GRAY))
            line_traces.extend(draw_lines(_THUMB_LANDMARKS, _PEACH))
            line_traces.extend(draw_lines(_INDEX_FINGER_LANDMARKS, _PURPLE))
            line_traces.extend(draw_lines(_MIDDLE_FINGER_LANDMARKS, _YELLOW))
            line_traces.extend(draw_lines(_RING_FINGER_LANDMARKS, _GREEN))
            line_traces.extend(draw_lines(_PINKY_FINGER_LANDMARKS, _BLUE))

            # Create the layout
            layout = go.Layout(
                scene=dict(
                    xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"
                ),
                title=f"Gesture: {gesture.category_name}",
            )

            # Create the figure and add the scatter plot and lines
            fig = go.Figure(data=[scatter] + line_traces, layout=layout)

            fig.update_layout(showlegend=False, autosize=True, width=600, height=600)

            return fig
        else:
            return None
