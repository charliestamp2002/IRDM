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


def load_glove_embeddings(glove_file):
    """
    Load GloVe embeddings from a text file.
    
    Parameters:
        glove_file (str): Path to GloVe embeddings (e.g., "glove.6B.300d.txt").
    
    Returns:
        gensim.models.KeyedVectors: Word embeddings model.
    """
    embeddings = {}
    
    with open(glove_file, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]  # First value is the word
            vector = np.array(values[1:], dtype=np.float32)  # Remaining values are the embedding
            embeddings[word] = vector
    
    return embeddings

def text_to_embedding_glove(text, glove_model, embedding_dim=300):
    """
    Convert text into an embedding vector by averaging word embeddings.
    
    Parameters:
        text (str): Input query or passage.
        glove_model (dict): Loaded GloVe embeddings as a dictionary.
        embedding_dim (int): Dimension of embeddings.
    
    Returns:
        np.array: Averaged embedding vector.
    """
    word_embeddings = []
    tokens = word_tokenize(text.lower())  # Tokenize and lowercase (This part may be changed for a better tokenizer...)
    for word in tokens: 
        if word in glove_model:
            word_embeddings.append(glove_model[word])

    if len(word_embeddings) == 0:  # Handle case where no words are found
        return np.zeros(embedding_dim)
    
    return np.mean(word_embeddings, axis=0)  # Compute average embedding

class QueryPassageDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # Convert to tensor
        self.y = torch.tensor(y, dtype=torch.float32)  # Convert to tensor

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LogisticRegression(nn.Module): 

    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x): 
        return torch.sigmoid(self.linear(x))

def fit(model, train_loader, val_loader, num_epochs=10, lr=1e-3, device = 'cpu'):

    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses = [], []

    for epoch in range(num_epochs): 
        model.train()

        running_loss = 0.0
        # Progress bar for training batches
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for batch_x, batch_y in progress_bar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)  
            batch_y = batch_y.view(-1, 1)  

            optimizer.zero_grad()  
            outputs = model(batch_x)  
            loss = criterion(outputs, batch_y)  
            loss.backward()  
            optimizer.step()  
            
            running_loss += loss.item()
            progress_bar.set_postfix({"Train Loss": loss.item()})  

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation Step
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0

        with torch.no_grad():  
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                batch_y = batch_y.view(-1, 1)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    return train_losses, val_losses

def evaluate(model, data_loader, device = 'cpu'): 

    model.eval()
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad(): # we are evaluating (no gradients needed)
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.cpu().numpy()
            outputs = model(batch_x).cpu().numpy()
            predictions = (outputs >= 0.5).astype(int)  # Convert probs to binary predictions
            probabilities = outputs.flatten()

            y_true.extend(batch_y)
            y_pred.extend(predictions.flatten())
            y_probs.extend(probabilities)

    # Compute Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division = 0)
    recall = recall_score(y_true, y_pred, zero_division = 0)
    f1 = f1_score(y_true, y_pred, zero_division = 0)
    auc_roc = roc_auc_score(y_true, y_probs)  # AUC-ROC
    auc_pr = average_precision_score(y_true, y_probs)  # AUC-PR

    print(f" Model Evaluation Results:")
    print(f" Accuracy: {accuracy:.4f}")
    print(f" Precision: {precision:.4f}")
    print(f" Recall: {recall:.4f}")
    print(f" F1 Score: {f1:.4f}")
    print(f" AUC-ROC: {auc_roc:.4f}")
    print(f" AUC-PR: {auc_pr:.4f}")


if __name__ == '__main__':

    # Download and load pretrained Word2Vec (GoogleNews 300-dimensional vectors)
    word2vec_model = api.load("word2vec-google-news-300")

    glove_path = "glove.6B.300d.txt"
    glove_model = load_glove_embeddings(glove_path)

    # Test: Retrieve embedding for a word
    print("Embedding for 'computer':", glove_model["computer"])

    # Load training data
    train_file = "train_data.tsv"
    train_df = pd.read_csv(train_file, sep = "\t", names=["qid", "pid", "query", "passage", "relevancy"])

    #Load Validation data
    val_file = "validation_data.tsv"
    val_df = pd.read_csv(val_file, sep="\t", names=["qid", "pid", "query", "passage", "relevancy"])

    # Example usage
    query = "What is machine learning?"
    query_embedding = text_to_embedding_glove(query, glove_model)
    print("Query embedding shape:", query_embedding.shape)

    tqdm.pandas(desc="Processing Queries")
    train_df["query_embedding"] = train_df["query"].progress_apply(lambda q: text_to_embedding_glove(q, glove_model))

    tqdm.pandas(desc="Processing Passages")
    train_df["passage_embedding"] = train_df["passage"].progress_apply(lambda p: text_to_embedding_glove(p, glove_model))

    tqdm.pandas(desc="Processing Validation Queries")
    val_df["query_embedding"] = val_df["query"].progress_apply(lambda q: text_to_embedding_glove(q, glove_model))

    tqdm.pandas(desc="Processing Validation Passages")
    val_df["passage_embedding"] = val_df["passage"].progress_apply(lambda p: text_to_embedding_glove(p, glove_model))

    X_train = np.hstack([train_df["query_embedding"].tolist(), train_df["passage_embedding"].tolist()])
    y_train = train_df["relevancy"].values  # Binary labels (0 or 1)


    X_val = np.hstack([val_df["query_embedding"].tolist(), val_df["passage_embedding"].tolist()])
    y_val = val_df["relevancy"].values  # Binary labels (0 or 1)    

    train_dataset = QueryPassageDataset(X_train, y_train)
    val_dataset = QueryPassageDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    #val_loader = DataLoader(val_dataset, batch_size=32)

    #val_dataset = QueryPassageDataset(X_val, y_val)

    input_dim = X_train.shape[1]
    model = LogisticRegression(input_dim)
    train_losses, val_losses = fit(model, train_loader, val_loader, num_epochs=10, lr=0.001, device="cpu")

    #evaluate model
    evaluate(model, val_loader)





    
    




