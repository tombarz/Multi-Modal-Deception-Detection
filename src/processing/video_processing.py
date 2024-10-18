# src/data_processing/video_processing.py

import cv2
import dlib
import numpy as np
import sys
import os

# Adjust the import path if necessary
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import Config

# Initialize dlib's face detector and landmark predictor
predictor_path = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def extract_landmarks(frames, display=False):
    landmarks_seq = []
    for idx, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if len(faces) == 0:
            # Optionally, handle missing faces
            continue  # Skip frames where no face is detected
        face = faces[0]  # Assuming the first detected face
        shape = predictor(gray, face)
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])
        landmarks_seq.append(landmarks)

        if display:
            # Draw landmarks on the frame
            for (x, y) in landmarks:
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            # Show the frame
            cv2.imshow('Frame with Landmarks', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  # Exit if 'q' is pressed

    if display:
        cv2.destroyAllWindows()
    return landmarks_seq

def pad_or_truncate_sequence(sequence, max_len=Config.VIDEO_MAX_LEN):
    seq_len = len(sequence)
    if seq_len == max_len:
        return sequence
    elif seq_len < max_len:
        padding = [np.zeros((68, 2))] * (max_len - seq_len)
        return sequence + padding
    else:
        return sequence[:max_len]

def process_video(video_path, display=False):
    frames = extract_frames(video_path)
    landmarks_seq = extract_landmarks(frames, display=display)
    if len(landmarks_seq) == 0:
        return None
    sequence = pad_or_truncate_sequence(landmarks_seq)
    sequence = np.array(sequence).reshape(Config.SEQ_LEN, -1)  # Shape: (SEQ_LEN, 136)
    return sequence
