# -*- coding: utf-8 -*-
"""Untitled3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/16QLCBDad7zDeoI32Vx6qH1AphvFeCSwF
"""

import pandas as pd
import json
file_path = 'trainin_stage.jsonl'

data = list(map(json.loads, open(file_path) ))
print(len(data))

df = pd.DataFrame(data)

df.shape

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)
MAX_LEN = 512
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 2e-5
MODEL_NAME = 'bert-base-uncased'

# Dataset Class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float)
        }

# Model Architecture
class BertClassifier(nn.Module):
    def __init__(self, n_classes):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

# Training Function
def train_model(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in tqdm(train_loader):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['label'].to(DEVICE)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Accumulate loss
        batch_size = input_ids.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    # Return average loss for the epoch
    return total_loss / total_samples

# Evaluation Function
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            outputs = model(input_ids, attention_mask)
            probs = torch.sigmoid(outputs)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(torch.argmax(labels, dim=1).cpu().numpy())

    return precision_recall_fscore_support(all_labels, all_preds, average='weighted')

# Main Pipeline
def main():
    # Load and prepare data
    # Replace with your data
    texts = df['input'].values
    labels = df['label'].values.reshape(-1, 1)

    # One-hot encode labels
    ohe = OneHotEncoder(sparse_output=False)
    encoded_labels = ohe.fit_transform(labels)
    n_classes = encoded_labels.shape[1]

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # Cross-validation setup
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(texts, encoded_labels.argmax(1))):
        print(f'\nFold {fold + 1}/5')
        print('-' * 50)

        # Split data
        train_texts, test_texts = texts[train_idx], texts[test_idx]
        train_labels, test_labels = encoded_labels[train_idx], encoded_labels[test_idx]

        # Calculate pos_weight for imbalanced classes
        pos_weight = []
        for i in range(n_classes):
            n_pos = train_labels[:, i].sum()
            if n_pos == 0:
                pos_weight.append(1.0)
            else:
                pos_weight.append((len(train_labels) - n_pos) / n_pos)
        pos_weight = torch.tensor(pos_weight, dtype=torch.float).to(DEVICE)

        # Create DataLoaders
        train_dataset = TextDataset(train_texts, train_labels, tokenizer, MAX_LEN)
        test_dataset = TextDataset(test_texts, test_labels, tokenizer, MAX_LEN)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Initialize model and optimizer
        model = BertClassifier(n_classes).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Training
        for epoch in range(1, EPOCHS + 1):
            print(f'Epoch {epoch}/{EPOCHS}')
            train_loss = train_model(model, train_loader, criterion, optimizer)
            print("Train Loss ", train_loss)

        # Evaluation
        precision, recall, f1, _ = evaluate_model(model, test_loader)
        fold_metrics.append({
            'fold': fold + 1,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

        print(f'\nFold {fold + 1} Metrics:')
        print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

    # Final results
    print('\nFinal Cross-Validation Results:')
    for metrics in fold_metrics:
        print(f"Fold {metrics['fold']}: "
              f"Precision: {metrics['precision']:.4f}, "
              f"Recall: {metrics['recall']:.4f}, "
              f"F1: {metrics['f1']:.4f}")

    avg_metrics = {
        'precision': np.mean([m['precision'] for m in fold_metrics]),
        'recall': np.mean([m['recall'] for m in fold_metrics]),
        'f1': np.mean([m['f1'] for m in fold_metrics])
    }
    print(f'\nAverage Metrics: Precision: {avg_metrics["precision"]:.4f}, '
          f'Recall: {avg_metrics["recall"]:.4f}, '
          f'F1: {avg_metrics["f1"]:.4f}')

if __name__ == '__main__':
    main()

import pandas as pd
import numpy as np
import torch
import os
import joblib
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
from pathlib import Path

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 2e-5
MODEL_NAME = 'bert-base-uncased'
SAVE_DIR = 'best_models'
os.makedirs(SAVE_DIR, exist_ok=True)

# Dataset Class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float)
        }

# Model Architecture
class BertClassifier(nn.Module):
    def __init__(self, n_classes):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

# Training Function
def train_model(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['label'].to(DEVICE)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        batch_size = input_ids.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    torch.cuda.empty_cache()
    return total_loss / total_samples

# Enhanced Evaluation Function
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_ids = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            outputs = model(input_ids, attention_mask)
            probs = torch.sigmoid(outputs)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(torch.argmax(labels, dim=1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    acc = accuracy_score(all_labels, all_preds)
    return precision, recall, f1, acc, all_preds, all_labels, all_probs

# Main Pipeline
def main(df):
    # Load data
    texts = df['input'].values
    labels = df['label'].values

    # Prepare labels
    ohe = OneHotEncoder(sparse_output=False)
    encoded_labels = ohe.fit_transform(labels.reshape(-1, 1))  # Reshape here
    n_classes = encoded_labels.shape[1]
    joblib.dump(ohe, os.path.join(SAVE_DIR, 'label_encoder.pkl'))

    # Initialize components
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_predictions = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(texts, encoded_labels.argmax(1))):
        print(f'\n{"="*40}')
        print(f'Fold {fold + 1}/5')
        print(f'{"="*40}')

        # Split data
        train_texts, test_texts = texts[train_idx], texts[test_idx]
        train_labels, test_labels = encoded_labels[train_idx], encoded_labels[test_idx]
        original_test_labels = labels[test_idx].flatten()

        # Class weights
        pos_weight = []
        for i in range(n_classes):
            n_pos = train_labels[:, i].sum()
            pos_weight.append((len(train_labels) - n_pos) / n_pos if n_pos else 1.0)
        pos_weight = torch.tensor(pos_weight, dtype=torch.float).to(DEVICE)

        # DataLoaders
        train_dataset = TextDataset(train_texts, train_labels, tokenizer, MAX_LEN)
        test_dataset = TextDataset(test_texts, test_labels, tokenizer, MAX_LEN)
        train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False)

        # Model setup
        model = BertClassifier(n_classes).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_f1 = 0.0
        history = []

        # Training loop
        for epoch in range(1, EPOCHS + 1):
            print(f'\nEpoch {epoch}/{EPOCHS}')
            train_loss = train_model(model, train_loader, criterion, optimizer)

            # Validation
            val_precision, val_recall, val_f1, val_acc, _, _, _ = evaluate_model(model, test_loader)
            print(f"Train Loss: {train_loss:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}")

            # Track metrics
            history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_f1': val_f1,
                'val_acc': val_acc
            })

            # Save best model
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_f1': val_f1,
                    'val_acc': val_acc,
                }, os.path.join(SAVE_DIR, f'best_model_fold_{fold+1}.pt'))

        # Load best model for predictions
        checkpoint = torch.load(os.path.join(SAVE_DIR, f'best_model_fold_{fold+1}.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])

        # Get final predictions
        _, _, _, _, preds, true_labels, probs = evaluate_model(model, test_loader)

        # Create prediction DataFrame
        fold_df = pd.DataFrame({
        'text': test_texts,
        'true_label': labels[test_idx],  # Original labels
        'predicted_label': ohe.categories_[0][preds],  # Direct category mapping
        'fold': fold+1
    })

        # Add probabilities
        for i, class_name in enumerate(ohe.categories_[0]):
            fold_df[f'prob_{class_name}'] = np.array(probs)[:, i]

        all_predictions.append(fold_df)

        # Store fold results
        best_epoch = max(history, key=lambda x: x['val_f1'])
        print(f'\nBest Epoch {best_epoch["epoch"]}:')
        print(f"F1: {best_epoch['val_f1']:.4f} | Acc: {best_epoch['val_acc']:.4f}")

    # Save all predictions
    final_predictions = pd.concat(all_predictions, ignore_index=True)
    final_predictions.to_csv('cross_validation_predictions.csv', index=False)
    print("\nSaved all predictions to cross_validation_predictions.csv")

    # Final report
    print('\nFinal Cross-Validation Results:')
    fold_results = []
    for fold, df in enumerate(all_predictions):
        precision, recall, f1, _ = precision_recall_fscore_support(
            df['true_label'], df['predicted_label'], average='weighted')
        acc = accuracy_score(df['true_label'], df['predicted_label'])
        fold_results.append({
            'Fold': fold+1,
            'F1': f1,
            'Accuracy': acc,
            'Precision': precision,
            'Recall': recall
        })

    results_df = pd.DataFrame(fold_results)
    print(results_df)
    print(f"\nAverage Metrics:\n{results_df.mean()['F1', 'Accuracy', 'Precision', 'Recall']}")

if __name__ == '__main__':
    main(df)

!pip install -U transformers



import pandas as pd

df = pd.read_json("data.jsonl", lines=True)

df.head()

def predict(texts, fold=2):
    # Load components
    n_classes = 5
    model = BertClassifier(n_classes).to(DEVICE)
    checkpoint = torch.load(f'best_models/best_model_fold_{fold}.pt', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    ohe = joblib.load('best_models/label_encoder.pkl')
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # Preprocess
    dataset = TextDataset(texts, np.zeros((len(texts), len(ohe.categories_[0]))), tokenizer, MAX_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Predict
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in loader:
            inputs = batch['input_ids'].to(DEVICE)
            masks = batch['attention_mask'].to(DEVICE)
            outputs = model(inputs, masks)
            probs = torch.sigmoid(outputs)
            predictions.extend(probs.cpu().numpy())
    print(ohe.categories_[0][np.argmax(predictions, axis=1)][0])
    return ohe.categories_[0][np.argmax(predictions, axis=1)][0]
    return pd.DataFrame({
        'text': texts,
        'prediction': ohe.categories_[0][np.argmax(predictions, axis=1)],
        **{f'prob_{cls}': np.array(predictions)[:, i] for i, cls in enumerate(ohe.categories_[0])}
    })

data = '''
Write a python function grad that calculate Gradient of tensor u with respect to a tuple of tensors xs. Given :math:`u` and :math:`x_1`, ..., :math:`x_n`, the function returns :math:`\frac{\partial u}{\partial x_1}`, ..., :math:`\frac{\partial u}{\partial x_n}` :param u: The :math:`u` described above. :type u: `torch.Tensor` :param *xs: The sequence of :math:`x_i` described above. :type xs: `torch.Tensor` :return: A tuple of :math:`\frac{\partial u}{\partial x_1}`, ..., :math:`\frac{\partial u}{\partial x_n}` :rtype: List[`torch.Tensor`]	'''

df['result'] = df['prompt'].apply(predict)

import torch
import joblib
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from concurrent.futures import ThreadPoolExecutor

# Assume these are defined elsewhere in your code
# from your_module import BertClassifier, TextDataset, DEVICE, MODEL_NAME, MAX_LEN, BATCH_SIZE

# Global variables for model components
model = None
ohe = None
tokenizer = None

def load_components(fold=2):
    global model, ohe, tokenizer
    n_classes = 5
    # Initialize and load the model only once
    model = BertClassifier(n_classes).to(DEVICE)
    checkpoint = torch.load(f'best_models/best_model_fold_{fold}.pt', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load other components
    ohe = joblib.load('best_models/label_encoder.pkl')
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# Call this once before making predictions
load_components(fold=2)

def predict_text(text):
    """
    Predicts the category for a single text.
    """
    # Create a dataset with one text
    dataset = TextDataset([text],
                          np.zeros((1, len(ohe.categories_[0]))),
                          tokenizer,
                          MAX_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in loader:
            inputs = batch['input_ids'].to(DEVICE)
            masks = batch['attention_mask'].to(DEVICE)
            outputs = model(inputs, masks)
            probs = torch.sigmoid(outputs)
            predictions.extend(probs.cpu().numpy())

    # Return the predicted category for this single text
    pred_category = ohe.categories_[0][np.argmax(predictions, axis=1)][0]
    print(pred_category)
    return pred_category

# Assuming df is your DataFrame with a 'prompt' column containing texts to predict
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(predict_text, df['input']))

df['result'] = results



df.to_csv("aicoder.csv")

df[df["result"] == "infer"].shape

