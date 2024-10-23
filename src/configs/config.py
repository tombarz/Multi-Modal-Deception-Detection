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

    # Audio settings
    USE_AUDIO = False

    # Text settings
    USE_TEXT = True
    LANGUAGE_MODEL_NAME = 'bert-base-multilingual-cased'  # Change this to the desired model
    TEXT_FEATURE_DIM = 768  # Adjust based on the model's hidden size
    TEXT_MAX_LEN = 128  # Maximum number of tokens

    # Model settings
    HIDDEN_SIZE = 128
    NUM_LAYERS = 3
    DROPOUT_RATE = 0.5

    # Training settings
    NUM_EPOCHS = 30
    BATCH_SIZE = 16
    LEARNING_RATE = 0.0001
