import numpy as np
import matplotlib.pyplot as plt
import re
import time 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
from tqdm import tqdm
import os
os.chdir("/Users/charliestamp/Documents/IRDM/CW2")
import gensim
import gensim.downloader as api
import nltk
from nltk.tokenize import word_tokenize
nltk.download("punkt")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, accuracy_score

def mock_text_to_embedding(text, embedding_dim=300):
    """
    Generate a random 300-d embedding for each input text.
    """
    return np.random.rand(embedding_dim)

class QueryPassageDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        X = np.hstack([df["query_embedding"].tolist(), df["passage_embedding"].tolist()])
        y = df["relevancy"].values  # Labels

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # Ensure shape (N,1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

def fit(model, train_loader, val_loader, num_epochs=2, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()

    for epoch in range(num_epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1} - Training Loss: {loss.item():.4f}")


def evaluate(model, data_loader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            outputs = model(batch_x).cpu().numpy()
            predictions = (outputs >= 0.5).astype(int)

            y_true.extend(batch_y.cpu().numpy().flatten())
            y_pred.extend(predictions.flatten())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

def compute_lr_scores(model, queries_df, candidates_df):
    rankings = []
    for _, query_row in queries_df.iterrows():
        qid = query_row["qid"]
        query_embedding = torch.tensor(query_row["query_embedding"], dtype=torch.float32)

        query_passages = candidates_df[candidates_df["qid"] == qid].copy()
        X_test = np.stack(query_passages["passage_embedding"].values)
        X_test = np.hstack((np.tile(query_embedding.numpy(), (X_test.shape[0], 1)), X_test))
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

        with torch.no_grad():
            scores = model(X_test_tensor).numpy().flatten()

        query_passages["score"] = scores
        query_passages = query_passages.sort_values(by="score", ascending=False).reset_index(drop=True)
        query_passages["rank"] = query_passages.index + 1
        query_passages["algoname"] = "LR"

        rankings.extend(query_passages[["qid", "pid", "rank", "score", "algoname", "relevancy"]].values.tolist())

    rankings_df = pd.DataFrame(rankings, columns=["qid", "pid", "rank", "score", "algoname", "relevancy"])
    return rankings_df

def average_precision(relevant_labels):
    num_relevant = sum(relevant_labels)
    if num_relevant == 0:
        return 0

    precision_at_k = []
    num = 0
    for i, relev in enumerate(relevant_labels, start=1):
        if relev == 1:
            num += 1
            precision_at_k.append(num / i)

    return sum(precision_at_k) / num_relevant

def ndcg_at_k(relevance_scores, k=10):
    relevance_scores = relevance_scores[:k]
    dcg = sum(relevance_scores[i] / np.log2(i + 2) for i in range(len(relevance_scores)))
    idcg = sum(sorted(relevance_scores, reverse=True)[i] / np.log2(i + 2) for i in range(len(relevance_scores)))
    return dcg / idcg if idcg > 0 else 0

def compute_mAP_and_mNDCG(model, queries_df, passages_df):
    rankings = compute_lr_scores(model, queries_df, passages_df)
    query_metrics = []

    for qid, group in rankings.groupby("qid"):
        sorted_group = group.sort_values(by="score", ascending=False)
        relevancy_list = sorted_group["relevancy"].tolist()

        ap = average_precision(relevancy_list)
        ndcg = ndcg_at_k(relevancy_list, k=10)

        query_metrics.append({'AP': ap, 'NDCG': ndcg})

    results_df = pd.DataFrame(query_metrics)
    mAP = results_df['AP'].mean()
    mNDCG = results_df['NDCG'].mean()

    return {"mAP": mAP, "mNDCG": mNDCG}


# Mock dataset
mock_train_data = pd.DataFrame({
    "qid": ["q1", "q1", "q2", "q2", "q3", "q3"],
    "pid": ["p1", "p2", "p3", "p4", "p5", "p6"],
    "query": ["what is AI?", "what is AI?", "deep learning?", "deep learning?", "NLP techniques", "NLP techniques"],
    "passage": ["AI is intelligence", "AI is a field", "DL is a subset of AI", "DL uses neural networks", 
                "NLP involves text processing", "NLP helps understand language"],
    "relevancy": [1, 0, 1, 0, 1, 0]  # Binary labels (1 = relevant, 0 = not relevant)
})


mock_val_data = mock_train_data.copy()  # Use same for validation for simplicity
mock_test_queries = mock_train_data[['qid', 'query']].drop_duplicates()
mock_candidates = mock_train_data[['qid', 'pid', 'query', 'passage', 'relevancy']]

# Apply fake embeddings
mock_train_data["query_embedding"] = mock_train_data["query"].apply(lambda x: mock_text_to_embedding(x))
mock_train_data["passage_embedding"] = mock_train_data["passage"].apply(lambda x: mock_text_to_embedding(x))

mock_val_data["query_embedding"] = mock_val_data["query"].apply(lambda x: mock_text_to_embedding(x))
mock_val_data["passage_embedding"] = mock_val_data["passage"].apply(lambda x: mock_text_to_embedding(x))

mock_test_queries["query_embedding"] = mock_test_queries["query"].apply(lambda x: mock_text_to_embedding(x))
mock_candidates["passage_embedding"] = mock_candidates["passage"].apply(lambda x: mock_text_to_embedding(x))


# Create dataset and DataLoader
train_dataset = QueryPassageDataset(mock_train_data)
val_dataset = QueryPassageDataset(mock_val_data)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2)

# Train the model
input_dim = train_dataset.X.shape[1]
model = LogisticRegression(input_dim)
fit(model, train_loader, val_loader)

evaluate(model, val_loader)

rankings = compute_lr_scores(model, mock_train_data, mock_train_data)
print(rankings.head())

metrics = compute_mAP_and_mNDCG(model, mock_test_queries, mock_candidates)
print(metrics)