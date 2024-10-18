# src/inference/inference.py
from configs.config import Config
from preprocessing.video_preprocessing import process_video
from src.models.multi

def multimodal_inference(input_paths):
    features = []

    # Video
    if Config.USE_VIDEO:
        video_sequence = process_video(input_paths['video'])
        if video_sequence is None:
            print("No faces detected in the video.")
            return None
        features.append(video_sequence)

    # Audio
    if Config.USE_AUDIO:
        audio_features = extract_audio_features(input_paths['audio'])
        features.append(audio_features)

    # Text
    if Config.USE_TEXT:
        text = input_paths['text']
        text_features = extract_text_features(text)
        features.append(text_features)

    # Ensure all features have the same sequence length
    for i in range(len(features)):
        if features[i].shape[0] < Config.SEQ_LEN:
            padding = np.zeros((Config.SEQ_LEN - features[i].shape[0], features[i].shape[1]))
            features[i] = np.vstack((features[i], padding))
        else:
            features[i] = features[i][:Config.SEQ_LEN]

    concatenated_features = np.concatenate(features, axis=1)  # Shape: (seq_len, total_feature_dim)
    concatenated_features = torch.tensor(concatenated_features, dtype=torch.float32).unsqueeze(0).to(Config.DEVICE)

    # Load the model
    model = MultimodalLSTMModel()
    model.load_state_dict(torch.load(os.path.join(Config.MODEL_SAVE_DIR, 'multimodal_model.pth')))
    model.to(Config.DEVICE)
    model.eval()

    # Inference
    with torch.no_grad():
        output = model(concatenated_features)
        _, predicted = torch.max(output.data, 1)
        label = predicted.item()
        if label == 1:
            print("The input is predicted to be deceptive.")
        else:
            print("The input is predicted to be truthful.")
        return label
