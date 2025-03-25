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
nltk.download("wordnet")

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

from nltk.stem import WordNetLemmatizer # Use lemmatisation for now (this is optional and can be changed)
lemmatizer = WordNetLemmatizer()

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

    lemmatizer = WordNetLemmatizer()
    word_embeddings = []
    
    tokens = word_tokenize(text.lower())  # Tokenize and lowercase (This part may be changed for a better tokenizer...)
    # Optional: Lemmatize words (subject to change)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]    
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
    #criterion = nn.BCELoss()
    #Use weighted BCE loss
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1].to(device))
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


def evaluate_classification(rankings_df):
    """
    Evaluate the binary classification predictions.
    
    Parameters:
        rankings_df (DataFrame): DataFrame containing ["qid", "pid", "rank", "score", "prediction", "relevancy"].
    
    Returns:
        dict: Dictionary of evaluation metrics.
    """
    y_true = rankings_df["relevancy"].values  # Ground truth labels
    y_pred = rankings_df["prediction"].values  # Model's binary predictions

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc_roc = roc_auc_score(y_true, rankings_df["score"].values)  # AUC uses continuous scores
    auc_pr = average_precision_score(y_true, rankings_df["score"].values)  # PR AUC uses continuous scores

    print(f"Evaluation Results:")
    print(f" Accuracy: {accuracy:.4f}")
    print(f" Precision: {precision:.4f}")
    print(f" Recall: {recall:.4f}")
    print(f" F1 Score: {f1:.4f}")
    print(f" AUC-ROC: {auc_roc:.4f}")
    print(f" AUC-PR: {auc_pr:.4f}")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "AUC-ROC": auc_roc, "AUC-PR": auc_pr}

def compute_lr_scores(model, queries_df, candidates_df, include_relevancy=False):
    """
    Compute and rank passage relevance scores using the trained Logistic Regression model.
    
    Parameters:
        model: Trained logistic regression model.
        queries_df: DataFrame with query embeddings.
        candidates_df: DataFrame with passage embeddings.
    
    Returns:
        DataFrame: Ranked passages with columns [qid, pid, rank, score, algoname]
    """
        
    rankings = []
    device = 'cpu'

    for _, query_row in tqdm(queries_df.iterrows(), total=len(queries_df), desc=f"Scoring Passages with Logistic Regression"):
        qid = query_row["qid"]
        query_embedding = torch.tensor(query_row['query_embedding'], dtype = torch.float32).to(device)

        query_passages = candidates_df[candidates_df["qid"] == qid].copy()

        # Prepare input features (concatenation of query & passage embeddings)
        X_test = np.stack(query_passages["passage_embedding"].values)
        X_test = np.hstack((np.tile(query_embedding.cpu().numpy(), (X_test.shape[0], 1)), X_test))
        # converting to tensor...
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

        # Computing relevance scores
        with torch.no_grad():
            scores = model(X_test_tensor).cpu().numpy().flatten()

        query_passages["score"] = scores
        query_passages = query_passages.sort_values(by="score", ascending=False).reset_index(drop=True)

        query_passages["rank"] = query_passages.index + 1
        query_passages["algoname"] = "LR"
        query_passages["qid"] = query_passages["qid"].astype(str)
        query_passages["pid"] = query_passages["pid"].astype(str)
        #rankings.extend(query_passages[["qid", "pid", "rank", "score", "algoname"]].values.tolist())

        # Select appropriate columns
        columns_to_include = ["qid", "pid", "rank", "score", "algoname"]
        if include_relevancy and "relevancy" in query_passages.columns:
            columns_to_include.append("relevancy")

        rankings.extend(query_passages[columns_to_include].values.tolist())

    #rankings_df = pd.DataFrame(rankings, columns=["qid", "pid", "rank", "score", "algoname", "relevancy"])

    return pd.DataFrame(rankings, columns=columns_to_include)

# def compute_lr_scores(model, queries_df, candidates_df, threshold=0.5, include_relevancy=False):
#     """
#     Compute and rank passage relevance scores using the trained Logistic Regression model.
    
#     Parameters:
#         model: Trained logistic regression model.
#         queries_df: DataFrame with query embeddings.
#         candidates_df: DataFrame with passage embeddings.
#         threshold (float): Decision threshold for classification.
#         include_relevancy (bool): Whether to include the relevancy column in the rankings.

#     Returns:
#         DataFrame: Ranked passages with columns [qid, pid, rank, score, prediction, algoname]
#     """
        
#     rankings = []
#     device = 'cpu'

#     for _, query_row in tqdm(queries_df.iterrows(), total=len(queries_df), desc="Scoring Passages with Logistic Regression"):
#         qid = query_row["qid"]
#         query_embedding = torch.tensor(query_row['query_embedding'], dtype=torch.float32).to(device)

#         query_passages = candidates_df[candidates_df["qid"] == qid].copy()

#         # Prepare input features (concatenation of query & passage embeddings)
#         X_test = np.stack(query_passages["passage_embedding"].values)
#         X_test = np.hstack((np.tile(query_embedding.cpu().numpy(), (X_test.shape[0], 1)), X_test))
#         X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

#         # Compute relevance scores (continuous values from Logistic Regression)
#         with torch.no_grad():
#             scores = model(X_test_tensor).cpu().numpy().flatten()

#         query_passages["score"] = scores  # Keep original scores for ranking
#         query_passages["prediction"] = (scores >= threshold).astype(int)  # Binary classification

#         # Sort by score (ranking still based on continuous values)
#         query_passages = query_passages.sort_values(by="score", ascending=False).reset_index(drop=True)

#         query_passages["rank"] = query_passages.index + 1
#         query_passages["algoname"] = "LR"
#         query_passages["qid"] = query_passages["qid"].astype(str)
#         query_passages["pid"] = query_passages["pid"].astype(str)

#         # Select appropriate columns
#         columns_to_include = ["qid", "pid", "rank", "score", "prediction", "algoname"]
#         if include_relevancy and "relevancy" in query_passages.columns:
#             columns_to_include.append("relevancy")

#         rankings.extend(query_passages[columns_to_include].values.tolist())

#     return pd.DataFrame(rankings, columns=columns_to_include)

def average_precision(relevant_labels): 

    """
    Compute Average Precision (AP).
    
    Parameters:
        relevant_labels (list): Binary list where 1 = relevant, 0 = not relevant.
    
    Returns:
        float: Average Precision (AP) score.
    """

    num_relevant = sum(relevant_labels)
    if num_relevant == 0: 
        return 0
    
    precision_at_k = []
    num = 0

    for i, relev in enumerate(relevant_labels, start = 1): 
        if relev == 1: 
            num += 1 
            precision_at_k.append(num / i)

    return sum(precision_at_k) / num_relevant


def ndcg_at_k(relevance_scores, k=10):
    """
    Compute NDCG@k.
    
    Parameters:
        relevance_scores (list): List of relevance scores sorted by model ranking.
        k (int): Rank cutoff.
    
    Returns:
        float: NDCG@k score.
    """
     
    relevance_scores = relevance_scores[:k] # limit to k scores

    dcg = sum(relevance_scores[i] / np.log2(i + 2) for i in range(len(relevance_scores)))
    idcg = sum(sorted(relevance_scores, reverse=True)[i] / np.log2(i + 2) for i in range(len(relevance_scores)))

    return dcg / idcg if idcg > 0 else 0 # avoid div by 0


# def compute_mAP_and_mNDCG(model, val_queries_df, val_passages_df): 

#     rankings = compute_lr_scores(model, val_queries_df, val_passages_df)
#     rankings = rankings.sort_values(by = ['qid', 'score'], ascending = [True, False])

#     query_metrics = []

#     for qid, group in tqdm(rankings.groupby("qid"), desc="Computing mAP and mNDCG"):
#         sorted_group = group.sort_values(by = "score", ascending = False)
#         relevancy_list = sorted_group["relevancy"].tolist()

#         ap = average_precision(relevancy_list)
#         ndcg = ndcg_at_k(relevancy_list, k = 10)

#         query_metrics.append({'AP': ap, 'NDCG':ndcg})

#     results_df = pd.DataFrame(query_metrics)
#     mAP = results_df['AP'].mean() 
#     mNDCG = results_df['NDCG'].mean()

#     return {"mAP": mAP, "mNDCG": mNDCG}


def compute_mAP_and_mNDCG(model, queries_df, passages_df, include_relevancy=True):
    """
    Compute Mean Average Precision (mAP) and Mean Normalized Discounted Cumulative Gain (mNDCG).

    Parameters:
        model: Trained logistic regression model.
        queries_df: DataFrame with query embeddings.
        passages_df: DataFrame with passage embeddings.
        include_relevancy (bool): Whether to include the relevancy column in the rankings.

    Returns:
        dict: Dictionary containing 'mAP' and 'mNDCG' scores.
    """

    # Compute rankings with or without relevancy
    rankings = compute_lr_scores(model, queries_df, passages_df, include_relevancy=include_relevancy)

    # Ensure relevancy is present if required
    if include_relevancy and "relevancy" not in rankings.columns:
        raise ValueError("Error: 'relevancy' column is missing in computed rankings!")

    # Sort rankings correctly
    rankings = rankings.sort_values(by=['qid', 'score'], ascending=[True, False])

    query_metrics = []

    for qid, group in tqdm(rankings.groupby("qid"), desc="Computing mAP and mNDCG"):
        sorted_group = group.sort_values(by="score", ascending=False)
        
        if include_relevancy:
            relevancy_list = sorted_group["relevancy"].tolist()
        else:
            raise ValueError("Cannot compute mAP and mNDCG without relevancy information!")

        #print(f"Relevancy list at {qid}: {relevancy_list}")

        ap = average_precision(relevancy_list)
        ndcg = ndcg_at_k(relevancy_list, k=10)

        query_metrics.append({'AP': ap, 'NDCG': ndcg})

    results_df = pd.DataFrame(query_metrics)
    mAP = results_df['AP'].mean()
    mNDCG = results_df['NDCG'].mean()

    return {"mAP": mAP, "mNDCG": mNDCG}


def save_rankings(rankings_df, output_file):
    """
    Save rankings in the required six-column format.
    
    Format: <qid> A2 <pid> <rank> <score> <algoname>
    """
    rankings_df["A2"] = "A2"  # Static column
    rankings_df = rankings_df[["qid", "A2", "pid", "rank", "score", "algoname"]]
    
    rankings_df.to_csv(output_file, sep=" ", index=False, header=False)
    print(f"Logistic Regression rankings saved to {output_file}")

def plot_lr(train_loader, val_loader, input_dim, learning_rates=[0.0001, 0.001, 0.01, 0.1], num_epochs=200, device="cpu", include_val_loss = False):
    """
    Train logistic regression models with different learning rates and plot loss vs. epochs.

    Args:
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        input_dim (int): Number of input features.
        learning_rates (list): List of learning rates to test.
        num_epochs (int): Number of epochs for training.
        device (str): Device to run training on ("cpu" or "cuda").
    
    Returns:
        None (Plots Loss vs. Epoch for each learning rate)
    """
    plt.figure(figsize=(10, 6))

    for lr in learning_rates:
        model = LogisticRegression(input_dim).to(device)
        train_losses, val_losses = fit(model, train_loader, val_loader, num_epochs=num_epochs, lr=lr, device=device)
        
        if include_val_loss:
            plt.plot(range(1, num_epochs+1), train_losses, label=f"Train Loss (LR={lr})", linestyle='solid')
            plt.plot(range(1, num_epochs+1), val_losses, label=f"Val Loss (LR={lr})", linestyle = 'dashed')
        else:
            plt.plot(range(1, num_epochs+1), train_losses, label=f"Train Loss (LR={lr})", linestyle='solid') 

    plt.xlabel("Epoch", fontsize = 16)
    plt.ylabel("Loss", fontsize = 16)
    plt.title("Loss vs. Epoch for Different Learning Rates", fontsize = 20)
    plt.legend(loc = 'upper right')
    plt.grid(False)
    plt.show()



# ---------------------------------------- # 
"""
Create Negative Sampling function to address poor initial performance on mAP and compute_mAP_and_mNDCG
"""

def negative_sampling(train_df, neg_ratio = 3, random_seed = 42):

    np.random.seed(random_seed)
    positive_samples = train_df[train_df['relevancy'] == 1]
    negative_samples = train_df[train_df['relevancy'] == 0]

    num_negatives_needed = min(len(negative_samples), len(positive_samples) * neg_ratio)
    sampled_negatives = negative_samples.sample(n=num_negatives_needed, random_state=random_seed)

    # Combine positive and sampled negative samples
    balanced_train_df = pd.concat([positive_samples, sampled_negatives]).sample(frac=1, random_state=random_seed).reset_index(drop=True)

    print(f"Original dataset: {len(train_df)} samples")
    print(f"Balanced dataset: {len(balanced_train_df)} samples")
    print(f"Positive samples: {len(positive_samples)}, Negative samples after sampling: {len(sampled_negatives)}")

    return balanced_train_df

if __name__ == '__main__':

    # Download and load pretrained Word2Vec (GoogleNews 300-dimensional vectors)
    #word2vec_model = api.load("word2vec-google-news-300")

    #Download glove model (will use glove for now as opposed to word2vec)
    glove_path = "glove.6B.300d.txt"
    glove_model = load_glove_embeddings(glove_path)

    dtype_mapping = {"qid": str, "pid": str, "query": str, "passage": str, "relevancy": int}

    # Load training data
    train_file = "train_data.tsv"
    train_df = pd.read_csv(train_file, sep = "\t", names=["qid", "pid", "query", "passage", "relevancy"], header = 0, dtype = dtype_mapping)
    print("Length of Train DF:", len(train_df))

    # Reduce to 10% of the data to keep within RAM restrictions (random sampling)
    #train_df = train_df.sample(frac=0.1, random_state=42).reset_index(drop=True) 

    # Now use negative sampling to address class imbalance: 
    train_df = negative_sampling(train_df, neg_ratio = 5)

    #Load Validation data
    val_file = "validation_data.tsv"
    val_df = pd.read_csv(val_file, sep="\t", names=["qid", "pid", "query", "passage", "relevancy"], header = 0, dtype = dtype_mapping)
    print("Length of Val DF:", len(val_df))

    val_df = negative_sampling(val_df, neg_ratio = 5)

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

    # For Weighted BCE
    num_positives = np.sum(y_train == 1)
    num_negatives = np.sum(y_train == 0)
    total_samples = len(y_train)

    w_1 = total_samples /  (num_positives)  # Weight for relevant samples
    w_0 = total_samples / (num_negatives)  # Weight for non-relevant samples

    class_weights = torch.tensor([w_0, w_1], dtype=torch.float32)

    learning_rates = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

    #plot_lr(train_loader, val_loader, input_dim, learning_rates = learning_rates, num_epochs=1000, device="cpu", include_val_loss = False)

    #print(f"Comparing Learning Rates Completed!")

    # Best Learning rate = [] now used.
    model = LogisticRegression(input_dim)
    train_losses, val_losses = fit(model, train_loader, val_loader, num_epochs=50, lr=0.001, device="cpu")

    #evaluate model
    evaluate(model, val_loader)

    # Now re-ranking the candidate_passages_top1000.tsv file with test_queries.tsv: 

    test_queries_file = "test-queries.tsv"
    candidates_file = "candidate_passages_top1000.tsv"

    test_queries_df = pd.read_csv(test_queries_file, sep="\t", names=["qid", "query"])
    candidates_df = pd.read_csv(candidates_file, sep="\t", names=["qid", "pid", "query", "passage"])

    tqdm.pandas(desc="Processing Queries")
    test_queries_df["query_embedding"] = test_queries_df["query"].progress_apply(lambda q: text_to_embedding_glove(q, glove_model))

    tqdm.pandas(desc="Processing Passages")
    candidates_df["passage_embedding"] = candidates_df["passage"].progress_apply(lambda p: text_to_embedding_glove(p, glove_model))

    #computing lr scores
    lr_rankings = compute_lr_scores(model, test_queries_df, candidates_df,  include_relevancy = False)
    save_rankings(lr_rankings, "LR.txt")
    #computing mAP and mNDCG scores: 

    # Split Validation Data for mAP & mNDCG correctly...
    val_queries_df = val_df[['qid', 'query']].drop_duplicates()

    # Recompute query embeddings
    tqdm.pandas(desc="Processing Validation Queries")
    val_queries_df["query_embedding"] = val_queries_df["query"].progress_apply(lambda q: text_to_embedding_glove(q, glove_model))

    print(val_queries_df.columns)  
    print(val_queries_df["query_embedding"].isna().sum())  # Ensure no NaNs

    val_passages_df = val_df[['qid', 'pid', 'passage', 'relevancy']]
    val_passages_df["passage_embedding"] = val_passages_df["passage"].progress_apply(lambda q: text_to_embedding_glove(q, glove_model))

    # Debugging: Check if all necessary columns exist
    print("Validation Queries Columns:", val_queries_df.columns)  
    print("Validation Passages Columns:", val_passages_df.columns)  

    # Ensure embeddings exist before ranking
    print("Missing query embeddings:", val_queries_df["query_embedding"].isna().sum())  
    print("Missing passage embeddings:", val_passages_df["passage_embedding"].isna().sum())  

    # Ensure relevancy column exists in val_passages_df
    if "relevancy" not in val_passages_df.columns:
        raise ValueError("Error: 'relevancy' column is missing in validation passages!")
    
    #Check this (??? )
    #lr_rankings_with_relevancy = compute_lr_scores(model, val_queries_df, val_passages_df, include_relevancy=True)


    # # Compute mAP and mNDCG only if embeddings are present
    # if val_queries_df["query_embedding"].isna().sum() == 0 and val_passages_df["passage_embedding"].isna().sum() == 0:
    #     metrics = compute_mAP_and_mNDCG(model, val_queries_df, lr_rankings_with_relevancy) # needs to be val_passages_df and include correct relevancy
    #     print(metrics)
    # else:
    #     print("Error: Missing embeddings, cannot compute mAP and mNDCG!")
        
    # metrics = compute_mAP_and_mNDCG(model, val_queries_df, val_passages_df)
    metrics = compute_mAP_and_mNDCG(model, val_queries_df, val_passages_df, include_relevancy=True)
    print(metrics)
    # print(metrics)

    print("Training, Evaluation, and Metrics Calculation Complete!")



 
   



    
    




