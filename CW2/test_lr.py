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

from lr import load_glove_embeddings, text_to_embedding_glove, QueryPassageDataset, LogisticRegression, fit, evaluate, compute_lr_scores, save_rankings
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, accuracy_score

# TESTING LOGISTIC REGRESSION MODEL:

# ---------------------------------------------------------
# 🔹 Test Code: Generate Synthetic Data
# ---------------------------------------------------------
np.random.seed(42)
torch.manual_seed(42)

# Simulate 1000 query-passage pairs with 300-dimensional embeddings
num_samples = 1000
embedding_dim = 300

X_train = np.random.rand(num_samples, embedding_dim * 2)  # Simulated query + passage embeddings
y_train = np.random.randint(0, 2, size=num_samples)  # Binary relevance labels (0 or 1)

X_val = np.random.rand(200, embedding_dim * 2)  # Simulated validation set
y_val = np.random.randint(0, 2, size=200)

# Create dataset and DataLoader
train_dataset = QueryPassageDataset(X_train, y_train)
val_dataset = QueryPassageDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# ---------------------------------------------------------
# 🔹 Test Logistic Regression Model
# ---------------------------------------------------------
input_dim = embedding_dim * 2  # Since we concatenate query and passage embeddings
model = LogisticRegression(input_dim)

# Run a forward pass with dummy data
test_input = torch.randn(1, input_dim)  # Single sample
output = model(test_input)
print(f"🔹 Model Test Output: {output.item():.4f}")  # Should be between 0 and 1

# ---------------------------------------------------------
# 🔹 Train Logistic Regression Model
# ---------------------------------------------------------
train_losses, val_losses = fit(model, train_loader, val_loader, num_epochs=5, lr=0.001, device="cpu")
test_metrics = evaluate(model, val_loader, device= 'cpu')

print("✅ Model training complete! 🎉")

# TESTING WHOLE PIPELINE:

# Mock Training Data
train_data = {
    "qid": ["q1", "q1", "q2", "q2", "q3", "q3"],
    "pid": ["p1", "p2", "p3", "p4", "p5", "p6"],
    "query": ["What is AI?", "What is AI?", "Best sci-fi movies?", "Best sci-fi movies?", "Quantum computing?", "Quantum computing?"],
    "passage": ["AI is the field of intelligence", "Sports are fun", 
                "Sci-fi movies are futuristic", "I love to cook",
                "Quantum computing is the future", "Dogs are friendly"],
    "relevancy": [1, 0, 1, 0, 1, 0]  # Labels: 1 = relevant, 0 = not relevant
}
train_df = pd.DataFrame(train_data)

# Mock Validation Data
val_data = {
    "qid": ["q1", "q2", "q3"],
    "pid": ["p7", "p8", "p9"],
    "query": ["What is AI?", "Best sci-fi movies?", "Quantum computing?"],
    "passage": ["AI can beat humans in chess", "The Matrix is a great sci-fi movie", "Quantum bits exist in superposition"],
    "relevancy": [1, 1, 1]
}
val_df = pd.DataFrame(val_data)

glove_path = "glove.6B.300d.txt"
glove_model = load_glove_embeddings(glove_path)

# Apply mock embedding function
tqdm.pandas(desc="Processing Queries")
train_df["query_embedding"] = train_df["query"].progress_apply(lambda q: text_to_embedding_glove(q, glove_model, embedding_dim=5))

tqdm.pandas(desc="Processing Passages")
train_df["passage_embedding"] = train_df["passage"].progress_apply(lambda p: text_to_embedding_glove(p, glove_model, embedding_dim=5))

tqdm.pandas(desc="Processing Validation Queries")
val_df["query_embedding"] = val_df["query"].progress_apply(lambda q: text_to_embedding_glove(q, glove_model, embedding_dim=5))

tqdm.pandas(desc="Processing Validation Passages")
val_df["passage_embedding"] = val_df["passage"].progress_apply(lambda p: text_to_embedding_glove(p, glove_model, embedding_dim=5))

# Combine query and passage embeddings
X_train = np.hstack([np.vstack(train_df["query_embedding"]), np.vstack(train_df["passage_embedding"])])
y_train = np.array(train_df["relevancy"])

X_val = np.hstack([np.vstack(val_df["query_embedding"]), np.vstack(val_df["passage_embedding"])])
y_val = np.array(val_df["relevancy"])

# Create DataLoaders
train_dataset = QueryPassageDataset(X_train, y_train)
val_dataset = QueryPassageDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2)

# Train the model
input_dim = X_train.shape[1]
model = LogisticRegression(input_dim)
fit(model, train_loader, val_loader, num_epochs=5, lr=0.01, device="cpu")

# Evaluate on validation set
evaluate(model, val_loader)

# Test Query Ranking
test_queries = pd.DataFrame({"qid": ["q1", "q2"], "query": ["What is AI?", "Best sci-fi movies?"]})
test_candidates = pd.DataFrame({
    "qid": ["q1", "q1", "q2", "q2"],
    "pid": ["p10", "p11", "p12", "p13"],
    "query": ["What is AI?", "What is AI?", "Best sci-fi movies?", "Best sci-fi movies?"],
    "passage": ["AI technology is growing fast", "AI can solve problems", 
                "Sci-fi movies predict the future", "Some movies are not sci-fi"]
})

# Generate test embeddings
test_queries["query_embedding"] = test_queries["query"].apply(lambda q: text_to_embedding_glove(q, glove_model, embedding_dim=5))
test_candidates["passage_embedding"] = test_candidates["passage"].apply(lambda p: text_to_embedding_glove(p, glove_model, embedding_dim=5))

# Compute rankings
test_rankings = compute_lr_scores(model, test_queries, test_candidates)
save_rankings(test_rankings, "mock_LR.txt")

print("\nMock Pipeline complete! 🎉")

