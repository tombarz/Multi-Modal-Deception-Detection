

class Config:
    # General settings
    DATA_DIR = '../data/trial_data'
    MODEL_SAVE_DIR = '../models/'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Sequence length
    SEQ_LEN = 100  # Define a fixed sequence length

    # Video settings
    USE_VIDEO = True
    VIDEO_FEATURE_DIM = 136  # Adjust if necessary

    # Audio settings
    USE_AUDIO = False
    AUDIO_N_MFCC = 40  # Number of MFCC coefficients
    AUDIO_FEATURE_DIM = AUDIO_N_MFCC

    # Text settings
    USE_TEXT = False
    TEXT_MAX_LEN = 100  # Max token length
    TEXT_FEATURE_DIM = 768  # BERT base hidden size

    # Model settings
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    DROPOUT_RATE = 0.5

    # Training settings
    NUM_EPOCHS = 30
    BATCH_SIZE = 16
    LEARNING_RATE = 0.0001
