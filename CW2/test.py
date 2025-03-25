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

from task2_CW2 import load_glove_embeddings, text_to_embedding_glove, QueryPassageDataset, LogisticRegression, fit, evaluate
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, accuracy_score


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

