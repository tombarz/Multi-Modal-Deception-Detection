import whisper_timestamped as whisper
import json
import os
import subprocess


# Class is intended to allow us to create a single word subtitle file
from PIL import ImageFont
from moviepy.video.VideoClip import TextClip
from moviepy.config import change_settings
change_settings({"IMAGEMAGICK_BINARY": r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"})



def create_subs(file_name: str):

    os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ["PATH"]
    print("file name ",file_name)
    audio = whisper.load_audio(file_name)

    model = whisper.load_model("tiny", device="cpu")

    result = model.transcribe(audio, language="en", task="transcribe", verbose=True)

    # Adjusting the segments to ensure no more than 6 words per line
    for segment in result["segments"]:
        words = segment["text"].split()
        lines = [' '.join(words[i:i + 6]) for i in range(0, len(words), 6)]
        segment["text"] = '\n'.join(lines)

    subtitle_json_path = file_name.replace(".mp3", ".json")
    with open(subtitle_json_path, "w", encoding="utf-8") as outfile:
        json.dump(result, outfile, indent=2, ensure_ascii=False)
    combine_sentences(subtitle_json_path)
    return subtitle_json_path

def get_text_sections(data):
    texts = [segment["text"] for segment in data["segments"]]
    return texts

def srttime_to_seconds(srt_time):
    return srt_time.hours * 3600 + srt_time.minutes * 60 + srt_time.seconds + srt_time.milliseconds / 1000.0


def split_segments_by_newline(data):
    # Initialize variables for the new JSON
    new_segments = []

    for segment in data["segments"]:
        # Split the segment text by '\n'
        text_parts = segment["text"].split('\n')
        start_time = segment["start"]
        end_time = segment["end"]

        # Calculate the time duration for each part
        duration = (end_time - start_time) / len(text_parts)

        for i, text_part in enumerate(text_parts):
            # Calculate the start and end times for each part
            part_start_time = start_time + i * duration
            part_end_time = start_time + (i + 1) * duration

            # Create a new segment for each part
            new_segment = {
                "id": segment["id"] + i / 10,  # Adding a small fraction to ensure uniqueness
                "seek": segment["seek"],
                "start": part_start_time,
                "end": part_end_time,
                "text": text_part.strip(),
                "tokens": segment["tokens"],
                "temperature": segment["temperature"],
                "avg_logprob": segment["avg_logprob"],
                "compression_ratio": segment["compression_ratio"],
                "no_speech_prob": segment["no_speech_prob"],
                "confidence": segment["confidence"],
                "words": segment["words"]
            }
            new_segments.append(new_segment)

    # Update the segments in the data dictionary
    data["segments"] = new_segments

    return data  # Return the updated dictionary


    # Initialize variables for the new JSON
    new_segments = []

    for segment in data["segments"]:
        # Split the segment text by '\n'
        text_parts = segment["text"].split('\n')
        start_time = segment["start"]
        end_time = segment["end"]

        # Calculate the time duration for each part
        duration = (end_time - start_time) / len(text_parts)

        for i, text_part in enumerate(text_parts):
            # Calculate the start and end times for each part
            part_start_time = start_time + i * duration
            part_end_time = start_time + (i + 1) * duration

            # Create a new segment for each part
            new_segment = {
                "id": segment["id"] + i / 10,  # Adding a small fraction to ensure uniqueness
                "seek": segment["seek"],
                "start": part_start_time,
                "end": part_end_time,
                "text": text_part.strip(),
                "tokens": segment["tokens"],
                "temperature": segment["temperature"],
                "avg_logprob": segment["avg_logprob"],
                "compression_ratio": segment["compression_ratio"],
                "no_speech_prob": segment["no_speech_prob"],
                "confidence": segment["confidence"],
                "words": segment["words"]
            }
            new_segments.append(new_segment)

    # Update the segments in the data dictionary
    data["segments"] = new_segments

    # Convert the updated data back to JSON
    updated_json = json.dumps(data, indent=2)

    return updated_json


def generate_srt(data, srt_filename):
    data = split_segments_by_newline(data)
    subtitle_data = []
    print(data)
    for item in data['segments']:
        subtitle_data.append((item['start'], item['end'], item['words']))

    with open(srt_filename, 'w') as srt_file:
        count = 1
        for subtitle in subtitle_data:
            # Create a list to store each line of the subtitle
            lines = []
            current_line = []
            start = None
            # Create each subtitle line by grouping words together, up to 6 words per line
            for word_data in subtitle[2]:
                if(start == None):
                    start = word_data['start']
                current_line.append(word_data['text'])

                if len(current_line) == 1 or word_data['text'].endswith('.'):
                    lines.append((current_line, start, word_data['end']))
                    current_line = []
                    start = None

            # If there are any remaining words that didn't make up a full line, add them as a line
            if current_line:
                lines.append((current_line, start, subtitle[2][-1]['end']))

            # Write each line to the SRT file with its respective timing
            for line_data in lines:
                start_time = line_data[1]
                end_time = line_data[2]
                text = ' '.join(line_data[0])

                start_time_formatted = f"{int(start_time / 3600)}:{int((start_time % 3600) / 60)}:{int(start_time % 60)},{int((start_time % 1) * 1000)}"
                end_time_formatted = f"{int(end_time / 3600)}:{int((end_time % 3600) / 60)}:{int(end_time % 60)},{int((end_time % 1) * 1000)}"

                srt_file.write(f"{count}\n")
                srt_file.write(f"{start_time_formatted} --> {end_time_formatted}\n")
                srt_file.write(f"{text}\n\n")
                count += 1

    return srt_filename



def burn_subtitles_to_video(video_path, json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    srt_file = generate_srt(data, "subs.srt")

    output_path = video_path.replace(".mp4", "_subbed.mp4")
    print("---------------------------------_here____________________________________________________________-")
    print(output_path)

    command = [
        'ffmpeg',
        '-i', video_path,
        '-vf', f"subtitles={srt_file}:force_style='Alignment=2,x=(w-text_w)/2,y=h/2-text_h/2'",
        '-c:a', 'copy',
        output_path
    ]

    subprocess.run(command)

    return output_path




def create_subtitle_clips(subtitles, videosize, fontsize=70, font='THEBOLDFONT.ttf'):
    subtitle_clips = []
    supported_subtitle_colors = ["White", "Grey", "Gold","White", "Grey","White","White","White","White","White"]
    max_width = videosize[0] * 3 / 4

    for subtitle in subtitles:
        start_time = srttime_to_seconds(subtitle.start)
        end_time = srttime_to_seconds(subtitle.end)
        duration = end_time - start_time
        video_width, video_height = videosize
        text = subtitle.text.replace(".", "").replace(",", "")

        adjusted_font_size = get_font_size(text, max_width, font, fontsize)*1.5

        text_clip = TextClip(text, fontsize=adjusted_font_size, font=font,
                             color="White"(supported_subtitle_colors), bg_color='transparent',
                             size=(video_width * 3 / 4, None), method='caption', stroke_color='black').set_start(
            start_time).set_duration(
            duration)
        subtitle_x_position = 'center'
        subtitle_y_position = video_height * 1 / 2
        text_position = (subtitle_x_position, subtitle_y_position)
        subtitle_clips.append(text_clip.set_position(text_position))

    return subtitle_clips

def combine_sentences(json_file):
    with open(json_file, 'r') as f:
        json_data = json.load(f)
        def is_complete_sentence(text):
            return text.strip().endswith(('.', '!', '?'))

    segments = json_data['segments']
    combined_segments = []
    temp_segment = {}

    for segment in segments:
        if not temp_segment:
            temp_segment = segment.copy()
        else:
            temp_segment['text'] += ' ' + segment['text']
            temp_segment['end'] = segment['end']

        # Check if the current segment ends with a complete sentence
        if is_complete_sentence(temp_segment['text']):
            combined_segments.append(temp_segment)
            temp_segment = {}

    # Add any remaining segment
    if temp_segment:
        combined_segments.append(temp_segment)

    # Replace the segments in the JSON
    json_data['segments'] = combined_segments
    return json_data


def get_font_size(text, max_width, font_path, initial_font_size=70):
    font_size = initial_font_size
    font = ImageFont.truetype(font_path, font_size)
    text_width = font.getbbox(text)[2]  # The third element in the tuple is the width of the text

    while text_width > max_width:
        font_size -= 1
        font = ImageFont.truetype(font_path, font_size)
        text_width = font.getbbox(text)[2]

    return font_size


if __name__ == '__main__':
    #ret = create_subs("video1.mp3")
    print("run")