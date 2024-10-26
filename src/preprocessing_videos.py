import os
import numpy as np
from configs.config import Config
from models.PPR.video_processing import process_video

def preprocess_videos():
    # Paths to raw data directories
    deceptive_dir = 'C:/Users/TomBa/OneDrive/desktop/startUps/Veritas/code/Multi-Modal-Deception-Detection/data/trial_data/video/Deceptive'
    truthful_dir = 'C:/Users/TomBa/OneDrive/desktop/startUps/Veritas/code/Multi-Modal-Deception-Detection/data/trial_data/video/Truthful'

    # Paths to preprocessed data directories
    preprocessed_base_dir = os.path.join(Config.BASE_DIR, 'preprocessed_data')
    os.makedirs(preprocessed_base_dir, exist_ok=True)
    deceptive_output_dir = os.path.join(preprocessed_base_dir, 'Deceptive')
    truthful_output_dir = os.path.join(preprocessed_base_dir, 'Truthful')
    os.makedirs(deceptive_output_dir, exist_ok=True)
    os.makedirs(truthful_output_dir, exist_ok=True)

    # Process Deceptive videos
    for video_name in os.listdir(deceptive_dir):
        video_path = os.path.join(deceptive_dir, video_name)
        print(f"Processing {video_path}...")
        sequence = process_video(video_path, display=False)
        if sequence is None:
            print(f"Warning: No landmarks extracted for {video_name}. Skipping.")
            continue
        # Save the sequence
        output_path = os.path.join(deceptive_output_dir, f"{os.path.splitext(video_name)[0]}.npy")
        np.save(output_path, sequence)
        print(f"Saved preprocessed data to {output_path}")

    # Process Truthful videos
    for video_name in os.listdir(truthful_dir):
        video_path = os.path.join(truthful_dir, video_name)
        print(f"Processing {video_path}...")
        sequence = process_video(video_path, display=False)
        if sequence is None:
            print(f"Warning: No landmarks extracted for {video_name}. Skipping.")
            continue
        # Save the sequence
        output_path = os.path.join(truthful_output_dir, f"{os.path.splitext(video_name)[0]}.npy")
        np.save(output_path, sequence)
        print(f"Saved preprocessed data to {output_path}")

if __name__ == '__main__':
    preprocess_videos()
