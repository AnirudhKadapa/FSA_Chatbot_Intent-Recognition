import argparse
import joblib
import torch
import numpy as np

def load_logreg_model():
    model = joblib.load("checkpoints/logreg/best_model.pkl")
    vectorizer = joblib.load("checkpoints/logreg/vectorizer.pkl")
    label_encoder = joblib.load("checkpoints/logreg/label_encoder.pkl")
    return lambda x: label_encoder.inverse_transform(model.predict(vectorizer.transform([x])))[0]

def load_svm_model():
    model = joblib.load("checkpoints/svm/best_model.pkl")
    vectorizer = joblib.load("checkpoints/svm/vectorizer.pkl")
    label_encoder = joblib.load("checkpoints/svm/label_encoder.pkl")
    return lambda x: label_encoder.inverse_transform(model.predict(vectorizer.transform([x])))[0]

def load_lstm_model():
    from torchtext.data.utils import get_tokenizer
    from torch.nn.utils.rnn import pad_sequence
    from torchtext.vocab import build_vocab_from_iterator
    import torch.nn as nn
    import pandas as pd

    class LSTMClassifier(nn.Module):
        def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)
        def forward(self, x):
            x = self.embedding(x)
            _, (hn, _) = self.lstm(x)
            return self.fc(hn[-1])

    tokenizer = get_tokenizer("basic_english")
    df = pd.read_csv("intent_dataset.csv")
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.fit(df['intent'])

    def yield_tokens(data_iter):
        for text in data_iter:
            yield tokenizer(text)

    vocab = build_vocab_from_iterator(yield_tokens(df['text']), specials=["<pad>"])
    vocab.set_default_index(vocab["<pad>"])

    model = LSTMClassifier(len(vocab), 64, 64, len(label_encoder.classes_))
    model.load_state_dict(torch.load("checkpoints/lstm/best_model.pt", map_location=torch.device("cpu")))
    model.eval()

    def predict(text):
        tokens = torch.tensor(vocab(tokenizer(text)), dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            output = model(tokens)
            pred = torch.argmax(output, dim=1).item()
        return label_encoder.inverse_transform([pred])[0]

    return predict

def load_bert_model():
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
    import torch

    tokenizer = DistilBertTokenizerFast.from_pretrained("checkpoints/bert")
    model = DistilBertForSequenceClassification.from_pretrained("checkpoints/bert")
    label_encoder = joblib.load("checkpoints/bert/label_encoder.pkl")

    def predict(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=1).item()
        return label_encoder.inverse_transform([pred])[0]

    return predict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intent Prediction Script")
    parser.add_argument("--model", type=str, choices=["logreg", "svm", "lstm", "bert"], required=True, help="Model to use")
    parser.add_argument("--text", type=str, required=True, help="Input text to classify")
    args = parser.parse_args()

    model_loaders = {
        "logreg": load_logreg_model,
        "svm": load_svm_model,
        "lstm": load_lstm_model,
        "bert": load_bert_model
    }

    predictor = model_loaders[args.model]()
    prediction = predictor(args.text)
    print(f"Predicted Intent: {prediction}")
