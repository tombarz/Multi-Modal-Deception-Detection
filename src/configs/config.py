import torch
class Config:
    # General settings
    DATA_DIR = '../data/trial_data'
    MODEL_SAVE_DIR = '../models/'
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
