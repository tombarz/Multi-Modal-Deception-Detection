# data_processing.py (we'll organize this later)

import cv2
import dlib
import numpy as np

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

def extract_landmarks(frames):
    landmarks_seq = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if len(faces) == 0:
            # Optionally, you can handle missing faces here
            continue  # Skip frames where no face is detected
        face = faces[0]  # Assuming the first detected face
        shape = predictor(gray, face)
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])
        landmarks_seq.append(landmarks)
    return landmarks_seq

def pad_or_truncate_sequence(sequence, max_len=100):
    seq_len = len(sequence)
    if seq_len == max_len:
        return sequence
    elif seq_len < max_len:
        padding = [np.zeros((68, 2))] * (max_len - seq_len)
        return sequence + padding
    else:
        return sequence[:max_len]
