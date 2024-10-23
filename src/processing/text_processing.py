# src/data_processing/text_processing.py

from transformers import AutoTokenizer, AutoModel
import torch
from configs.config import Config

# Initialize tokenizer and model using Auto classes
tokenizer = AutoTokenizer.from_pretrained(Config.LANGUAGE_MODEL_NAME)
model = AutoModel.from_pretrained(Config.LANGUAGE_MODEL_NAME)
model.eval()
model.to(Config.DEVICE)

def extract_text_features(text):
    # Tokenize and encode the text
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding='max_length',
        max_length=Config.TEXT_MAX_LEN
    )
    inputs = {key: value.to(Config.DEVICE) for key, value in inputs.items()}

    # Get embeddings from the model
    with torch.no_grad():
        outputs = model(**inputs)
        # Use the [CLS] token representation for classification tasks
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # Shape: (1, hidden_size)
        # Convert to numpy array
        embedding = cls_embedding.cpu().numpy().squeeze(0)  # Shape: (hidden_size,)
    return embedding  # Return as a numpy array
