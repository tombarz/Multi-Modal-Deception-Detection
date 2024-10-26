import os
from bucher import extract_audio_from_mp4, create_subclips_from_json
from whispers_subs import create_subs



def pre_proc(raw_footage: str):
    if not os.path.exists(raw_footage):
        print(f"Error: MP4 file {raw_footage} not found!")
        return

    audio_file = extract_audio_from_mp4(raw_footage)
    json_path = create_subs(audio_file)
    create_subclips_from_json(raw_footage,json_path)


if __name__ == '__main__':
    pre_proc("trial_lie_001.mp4")
