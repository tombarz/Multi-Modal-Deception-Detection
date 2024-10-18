# src/models/multimodal_model.py

import torch.nn as nn
from configs.config import Config

class MultimodalLSTMModel(nn.Module):
    def __init__(self):
        super(MultimodalLSTMModel, self).__init__()

        # Calculate total input size
        total_input_size = 0
        if Config.USE_VIDEO:
            total_input_size += Config.VIDEO_FEATURE_DIM
        if Config.USE_AUDIO:
            total_input_size += Config.AUDIO_FEATURE_DIM
        if Config.USE_TEXT:
            total_input_size += Config.TEXT_FEATURE_DIM

        self.hidden_size = Config.HIDDEN_SIZE
        self.num_layers = Config.NUM_LAYERS

        self.lstm = nn.LSTM(
            total_input_size,
            self.hidden_size,
            self.num_layers,
            batch_first=True,
            dropout=Config.DROPOUT_RATE
        )

        self.fc = nn.Linear(self.hidden_size, 2)

    def forward(self, x):
        # x shape: (batch_size, seq_len, total_input_size)
        h0 = x.new_zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = x.new_zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Take the output from the last time step
        out = self.fc(out)
        return out
