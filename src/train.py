# src/train.py

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from configs.config import Config
from processing.video_processing import process_video
from processing.text_processing import extract_text_features
from datasets.multimodal_dataset import MultimodalDataset
from models.multimodal_model import MultimodalLSTMModel
import torch.nn as nn

def load_data():
    data_list = []

    # Load Deceptive data
    deceptive_data = load_data_from_directory('Deceptive', label=1)
    data_list.extend(deceptive_data)

    # Load Truthful data
    truthful_data = load_data_from_directory('Truthful', label=0)
    data_list.extend(truthful_data)

    return data_list

def load_data_from_directory(subdir, label):
    data_list = []

    # Paths to data directories
    preprocessed_dir = os.path.join(Config.PREPROCESSED_DATA_DIR, subdir)
    text_dir = os.path.join(Config.TEXT_DATA_DIR, subdir)

    for file_name in os.listdir(preprocessed_dir):
        file_base_name = os.path.splitext(file_name)[0]
        sample_data = {}

        # Load video features if using video modality
        if Config.USE_VIDEO:
            video_file_path = os.path.join(preprocessed_dir, file_name)
            sequence = np.load(video_file_path)
            sample_data['video'] = sequence

        # Load text features if using text modality
        if Config.USE_TEXT:
            text_file_path = os.path.join(text_dir, f"{file_base_name}.txt")
            if not os.path.isfile(text_file_path):
                print(f"Warning: Text file {text_file_path} not found. Skipping sample.")
                continue
            with open(text_file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            text_embedding = extract_text_features(text_content)
            sample_data['text'] = text_embedding

        sample_data['label'] = label
        data_list.append(sample_data)

    return data_list

def train_one_epoch(model, criterion, optimizer, train_loader):
    model.train()
    running_loss = 0.0

    for features, labels in train_loader:
        features = features.to(Config.DEVICE)
        labels = labels.to(Config.DEVICE)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * features.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Compute confusion matrix components
            TP += ((predicted == 1) & (labels == 1)).sum().item()
            TN += ((predicted == 0) & (labels == 0)).sum().item()
            FP += ((predicted == 1) & (labels == 0)).sum().item()
            FN += ((predicted == 0) & (labels == 1)).sum().item()

    accuracy = 100 * correct / total
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, TP, TN, FP, FN, precision, recall, f1_score

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
        epoch_loss = train_one_epoch(model, criterion, optimizer, train_loader)
        print(f'Epoch [{epoch+1}/{Config.NUM_EPOCHS}], Loss: {epoch_loss:.4f}')

        # Evaluate on test set
        accuracy, TP, TN, FP, FN, precision, recall, f1_score = evaluate_model(model, test_loader)
        print(f'Test Accuracy: {accuracy:.2f}%')
        print(f'TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}')
        print(f'Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1_score:.2f}')

    # Save the trained model
    os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(Config.MODEL_SAVE_DIR, 'multimodal_model.pth'))

if __name__ == '__main__':
    train_model()
