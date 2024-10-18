import torch
from torch.utils.data import Dataset
from configs.config import Config
import numpy as np

class MultimodalDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list  # List of dictionaries with 'video', 'audio', 'text', 'label'

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        features = []

        # Video features
        if Config.USE_VIDEO:
            video_feat = sample['video']  # Shape: (seq_len, video_feature_dim)
            features.append(video_feat)

        # Audio features
        if Config.USE_AUDIO:
            audio_feat = sample['audio']  # Shape: (seq_len, audio_feature_dim)
            features.append(audio_feat)

        # Text features
        if Config.USE_TEXT:
            text_feat = sample['text']  # Shape: (seq_len, text_feature_dim)
            features.append(text_feat)

        # Concatenate features along the feature dimension
        # First, ensure all modalities have the same sequence length
        seq_len = Config.SEQ_LEN  # Define this in your config
        for i in range(len(features)):
            if features[i].shape[0] < seq_len:
                # Pad sequences
                padding = np.zeros((seq_len - features[i].shape[0], features[i].shape[1]))
                features[i] = np.vstack((features[i], padding))
            else:
                features[i] = features[i][:seq_len]

        concatenated_features = np.concatenate(features, axis=1)  # Shape: (seq_len, total_feature_dim)
        concatenated_features = torch.tensor(concatenated_features, dtype=torch.float32)

        label = torch.tensor(sample['label'], dtype=torch.long)
        return concatenated_features, label
