# src/train.py

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from configs.config import Config
from datasets.multimodal_dataset import MultimodalDataset
from models.multimodal_model import MultimodalLSTMModel
import torch.nn as nn

def load_data():
    data_list = []
    labels = []

    # Paths to data directories
    deceptive_dir = 'C:/Users/TomBa/OneDrive/desktop/startUps/Veritas/code/Multi-Modal-Deception-Detection/src/preprocessed_data/Deceptive'
    truthful_dir = 'C:/Users/TomBa/OneDrive/desktop/startUps/Veritas/code/Multi-Modal-Deception-Detection/src/preprocessed_data/Truthful'

    # Process deceptive videos
    for file_name in os.listdir(deceptive_dir):
        file_path = os.path.join(deceptive_dir, file_name)
        sequence = np.load(file_path)
        sample_data = {'video': sequence, 'label': 1}
        data_list.append(sample_data)

    # Process truthful videos
    for file_name in os.listdir(truthful_dir):
        file_path = os.path.join(truthful_dir, file_name)
        sequence = np.load(file_path)
        if sequence is None:
            continue  # Skip if processing failed
        sample_data = {'video': sequence, 'label': 0}
        data_list.append(sample_data)

    return data_list

def train_model():
    data_list = load_data()
    # Extract labels for stratification
    labels = [sample['label'] for sample in data_list]

    # Split data into training and testing with stratification
    train_data_list, test_data_list = train_test_split(
        data_list,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
    # Create datasets and dataloaders
    train_dataset = MultimodalDataset(train_data_list)
    test_dataset = MultimodalDataset(test_data_list)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Initialize model
    model = MultimodalLSTMModel()
    model.to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Training loop
    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            features = features.to(Config.DEVICE)  # Shape: (batch_size, seq_len, total_input_size)
            labels = labels.to(Config.DEVICE)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * features.size(0)

        epoch_loss = running_loss / len(train_dataset)
        print(f'Epoch [{epoch+1}/{Config.NUM_EPOCHS}], Loss: {epoch_loss:.4f}')

        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(Config.DEVICE)
                labels = labels.to(Config.DEVICE)
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')

    # Save the trained model
    os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(Config.MODEL_SAVE_DIR, 'multimodal_model.pth'))

if __name__ == '__main__':
    train_model()
