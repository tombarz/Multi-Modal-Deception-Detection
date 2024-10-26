import torch
from pathlib import Path
class Config:
    # General settings
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / 'data' / 'trial_data' / 'video'
    MODEL_SAVE_DIR = BASE_DIR / 'models'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Sequence length
    SEQ_LEN = 100  # Define a fixed sequence length

    # Video settings
    USE_VIDEO = True
    VIDEO_FEATURE_DIM = 136  # 68 landmarks * 2 coordinates
    VIDEO_MAX_LEN = SEQ_LEN

    # Disable audio and text modalities
    USE_AUDIO = False
    USE_TEXT = False

    # Model settings
    HIDDEN_SIZE = 128
    NUM_LAYERS = 3
    DROPOUT_RATE = 0.5

    # Training settings
    NUM_EPOCHS = 30
    BATCH_SIZE = 16
    LEARNING_RATE = 0.0001

    # Audio settings
    USE_AUDIO = True
    SAMPLE_RATE = 16000  # Sampling rate for audio files
    N_MFCC = 13  # Number of MFCC coefficients
    N_FFT = 512  # FFT window size
    HOP_LENGTH = 160  # Hop length (in samples)
    WIN_LENGTH = 400  # Window length (in samples)
    AUDIO_FEATURE_DIM = N_MFCC  # Since we use N_MFCC coefficients
