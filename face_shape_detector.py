import streamlit as st
import cv2
import numpy as np
import face_recognition

# Define the shape of different face types
face_shapes = {
    0: "Undefined",
    1: "Oval",
    2: "Round",
    3: "Square",
    4: "Heart",
    5: "Diamond",
}
face_ratios = {
    1: [1.75, 1.55, 1.15, 1.6, 1.9],
    2: [1.5, 1.6, 1.2, 1.5, 1.8],
    3: [1.6, 1.6, 1.2, 1.7, 2.2],
    4: [1.6, 2.0, 1.15, 1.6, 2.0],
    5: [1.5, 1.9, 1.2, 1.4, 1.7],
}

# Define a function to detect the face shape from a given image
def detect_face_shape(image):
    # Detect faces and facial landmarks using face_recognition library
    face_locations = face_recognition.face_locations(image, model="hog")
    face_landmarks = face_recognition.face_landmarks(image, face_locations, model="large")

    # If no face is detected, return undefined
    if len(face_landmarks) == 0:
        return "Undefined"

    # Get the facial landmarks for the first face detected
    landmarks = face_landmarks[0]

    # Calculate the ratios of different facial features
    jawline = landmarks["chin"]
    nose = landmarks["nose_tip"]
    eyebrows = landmarks["left_eyebrow"] + landmarks["right_eyebrow"]
    eyes = landmarks["left_eye"] + landmarks["right_eye"]
    lips = landmarks["top_lip"] + landmarks["bottom_lip"]

    jawline_length = cv2.arcLength(np.array(jawline), True)
    nose_length = cv2.arcLength(np.array(nose), True)
    eyebrows_length = cv2.arcLength(np.array(eyebrows), True)
    eyes_length = cv2.arcLength(np.array(eyes), True)
    lips_length = cv2.arcLength(np.array(lips), True)

    face_shape_index = 0
    face_shape_distance = float("inf")
    for i in range(1, 6):
        distances = [
            jawline_length / nose_length,
            jawline_length / eyebrows_length,
            eyes_length / nose_length,
            lips_length / nose_length,
            lips_length / jawline_length,
        ]
        distance = sum([abs(distances[j] - face_ratios[i][j]) for j in range(len(distances))])
        if distance < face_shape_distance:
            face_shape_index = i
            face_shape_distance = distance

    return face_shapes[face_shape_index]

def app():
    # Set up the video capture
    cap = cv2.VideoCapture(0)
    st.title("Face Shape Detector")

    # Set the video capture properties to improve performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        if not ret:
            st.error("Failed to read frame from camera")
            break

        # Detect the face shape
        face_shape = detect_face_shape(frame)

        # Display the frame with the detected face shape if a face is detected
        if face_shape != "Undefined":
            cv2.putText(frame, face_shape, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            st.image(frame, channels="BGR")
            st.success(f"Face shape detected: {face_shape}")
            st.stop()

        # Display the live camera feed
        st.video(frame, format="BGR")

    # Close the webcam
    cap.release()

# Run the Streamlit app
if __name__ == "__main__":
    try:
        app()
    except Exception as e:
        st.error(f"An error occurred: {e}")