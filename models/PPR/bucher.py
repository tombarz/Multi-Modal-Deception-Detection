import json
import subprocess
import os
import librosa
from video_processing import process_video as facial_processor
from configs.config import Config


def create_subclips_from_json(video_file: str, json_file: str,output_folder = "Output"):

    # Load JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)

    video_name = os.path.splitext(video_file)[0]

    os.makedirs(output_folder, exist_ok=True)
    json_data = {}
    for segment in data['segments']:

        start_time = segment['start']
        end_time = segment['end']
        segment_text = segment['text'].replace('\n', ' ')  # Flatten the text for file name

        print(f"video {start_time} and {end_time} is {start_time<end_time}")

        os.makedirs(f"{output_folder}/video", exist_ok=True)
        os.makedirs(f"{output_folder}/audio", exist_ok=True)

        video_subclip_name = f"{output_folder}/video/{video_name}_clip_{segment['id']}_{start_time:.2f}-{end_time:.2f}.mp4"
        audio_subclip_name = f"{output_folder}/audio/{video_name}_clip_{segment['id']}_{start_time:.2f}-{end_time:.2f}.mp3"

        video_cmd = [
            'ffmpeg',
            '-y',
            '-i', video_file,  # input file
            '-ss', str(start_time),  # start time
            '-to', str(end_time),  # duration
            '-pix_fmt', 'yuv420p',  # pixel format
            '-c:v', 'libx264',  # video codec
            '-an',  # no audio
            video_subclip_name  # output file
        ]

        audio_cmd = [
            'ffmpeg',
            '-y',  # Overwrite if file exists
            '-i', video_file,  # Input video file
            '-ss', str(start_time),  # Start time of the clip
            '-to', str(end_time),  # End time of the clip
            '-q:a', '0',  # Set audio quality to highest
            '-map', 'a',  # Select audio stream only
            audio_subclip_name  # Output audio file name
        ]

        subprocess.run(video_cmd)
        subprocess.run(audio_cmd)
        y, sr = librosa.load(audio_subclip_name, sr=Config.SAMPLE_RATE)
        id = str(segment['id'])
        json_data[str(segment['id'])] = {
            "Whisper data": segment,
            "Video path" : video_subclip_name,
            "Audio data" : audio_subclip_name,
            "Audio MFCCs" : librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=Config.N_MFCC,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            win_length=Config.WIN_LENGTH).tolist(),
            "Facial Process" : facial_processor(video_subclip_name).tolist(),
            "Lie Truth Indicator" : "TBD"
        }
        print(f"Created video subclip: {video_subclip_name} [{start_time} to {end_time}] (video only)")
        print(f"Created audio subclip: {audio_subclip_name} [{start_time} to {end_time}] (audio only)")

    with open(f'{output_folder}/{video_subclip_name.replace(".mp4","")}', 'w') as json_f:
        json.dump(json_data, json_f, indent=4)


def extract_audio_from_mp4(mp4_file: str, output_folder = "Output"):
    if not os.path.exists(mp4_file):
        print(f"Error: MP4 file {mp4_file} not found!")
        return
    os.makedirs(output_folder, exist_ok=True)


    base_name = os.path.splitext(mp4_file)[0]

    mp3_file = f"{output_folder}/{base_name}_audio.mp3"

    cmd = [
        'ffmpeg',
        '-y',  # Overwrite existing files
        '-i', mp4_file,  # Input video file
        '-q:a', '0',  # Set audio quality to the highest
        '-map', 'a',  # Select only the audio stream
        mp3_file  # Output MP3 file
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        print(f"Error extracting audio from {mp4_file}")
        print(result.stderr.decode())  # Print the FFmpeg error message
    else:
        print(f"Successfully extracted audio to {mp3_file}")
        return mp3_file



#extract_audio_from_mp4("video1.mp4")
