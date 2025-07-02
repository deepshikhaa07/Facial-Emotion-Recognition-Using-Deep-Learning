import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
import cv2
import numpy as np
import keras
import tensorflow as tf
from PIL import Image

# Load model and labels
model = tf.keras.models.load_model("C:/Users/sanja/OneDrive/Desktop/emotion app/basic_cnn_model.keras")

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Load emojis from disk or use emoji characters as fallback
emoji_map = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😄",
    "sad": "😢",
    "surprise": "😲",
    "neutral": "😐"
}

def predict_emotion(face_img):
    face = cv2.resize(face_img, (48, 48))
    face = face.reshape(1, 48, 48, 1).astype("float32") / 255.0
    preds = model.predict(face)[0]
    return emotion_labels[np.argmax(preds)], np.max(preds)

def main():
    st.title("Real-Time Emotion Detection 😄😭😠")
    run = st.button("Start Webcam")

    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            emotion, confidence = predict_emotion(roi_gray)
            emoji = emoji_map.get(emotion, "")
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            label = f"{emotion} {emoji} ({confidence:.2f})"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)


        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

    cap.release()

if __name__ == "__main__":
    main()
