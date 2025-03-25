from lr import *
import torch.nn.functional as F


class TwoTowerNet(nn.Module):
    def __init__(self, embedding_dim, use_cosine = False):
        super(TwoTowerNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.use_cosine = use_cosine

        # Query encoding tower
        self.q_layers = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # Passage encoding tower
        self.p_layers = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # Final scoring layer using similarity features
        self.final_layers = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        if not self.use_cosine:
            self.final_layers = nn.Sequential(
                nn.Linear(3, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid())
            

    def forward(self, x):
        q_out = self.q_layers(x[:, :self.embedding_dim])  # Query encoding
        p_out = self.p_layers(x[:, self.embedding_dim:])  # Passage encoding

        if self.use_cosine:
            # Cosine similarity option
            score = F.cosine_similarity(q_out, p_out, dim=1).unsqueeze(1)  # Shape (batch_size, 1)
        else:
            # Compute similarity features
            dot_prod = torch.sum(q_out * p_out, dim=1, keepdim=True)  # Shape (batch_size, 1)
            q_out_norm = torch.norm(q_out, dim=1, keepdim=True)
            p_out_norm = torch.norm(p_out, dim=1, keepdim=True)

            # Final scoring
            score = self.final_layers(torch.hstack((dot_prod, q_out_norm, p_out_norm)))

        return score
    

class TwoTowerDataset(Dataset):
    def __init__(self, queries_df, passages_df):
        """
        Dataset for TwoTowerNet, combining query and passage embeddings.
        
        queries_df: DataFrame with query embeddings (qid, query_embedding)
        passages_df: DataFrame with passage embeddings (qid, passage_embedding, relevancy)
        """
        self.queries = queries_df.set_index("qid")  # Index by qid for fast lookup
        self.passages = passages_df

    def __len__(self):
        return len(self.passages)

    def __getitem__(self, idx):
        row = self.passages.iloc[idx]
        qid = row["qid"]
        # passage_embedding = row["passage_embedding"]

        # # Retrieve corresponding query embedding
        # #query_embedding = self.queries.loc[qid]["query_embedding"]

        # query_embedding = self.queries.loc[qid]["query_embedding"]
        # if isinstance(query_embedding, list):  # If stored as a list
        #     query_embedding = np.array(query_embedding, dtype=np.float32)
        # elif not isinstance(query_embedding, np.ndarray):  # Handle unexpected cases
        #     raise TypeError(f"Unexpected type {type(query_embedding)} for query_embedding")
        
        # passage_embedding = np.array(row["passage_embedding"], dtype=np.float32)

        # Fix for Pandas Series issue
        query_embedding = self.queries.loc[qid, "query_embedding"]
        if isinstance(query_embedding, pd.Series):  # If multiple rows exist, take the first
            query_embedding = query_embedding.iloc[0]

        # Ensure query_embedding is a NumPy array
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding, dtype=np.float32)
        elif not isinstance(query_embedding, np.ndarray):
            raise TypeError(f"Unexpected type {type(query_embedding)} for query_embedding")

        # Ensure passage_embedding is a NumPy array
        passage_embedding = row["passage_embedding"]
        if isinstance(passage_embedding, list):
            passage_embedding = np.array(passage_embedding, dtype=np.float32)
        elif not isinstance(passage_embedding, np.ndarray):
            raise TypeError(f"Unexpected type {type(passage_embedding)} for passage_embedding")

        combined_embedding = np.concatenate((query_embedding, passage_embedding), axis=0)

        # Combine query & passage embeddings
        #combined_embedding = np.concatenate((query_embedding, passage_embedding))

        return torch.tensor(combined_embedding, dtype=torch.float32), torch.tensor(row["relevancy"], dtype=torch.float32)
    
def train_two_tower(model, train_loader, val_loader, num_epochs=10, lr=1e-3, device='cpu'):
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for batch_x, batch_y in progress_bar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).view(-1, 1)

            optimizer.zero_grad()
            scores = model(batch_x)
            loss = criterion(scores, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({"Train Loss": loss.item()})  

        avg_train_loss = running_loss / len(train_loader)

        # Validation 
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).view(-1, 1)

                scores = model(batch_x)
                loss = criterion(scores, batch_y)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    return model

"""
EVALUATION FUNCTIONS
"""


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

def evaluate_two_tower(model, data_loader, device='cpu'):
    model.eval()
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.cpu().numpy()

            scores = model(batch_x).cpu().numpy().flatten()
            predictions = (scores >= 0.5).astype(int)

            y_true.extend(batch_y)
            y_pred.extend(predictions)
            y_probs.extend(scores)

    # Compute Evaluation Metrics
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "AUC-ROC": roc_auc_score(y_true, y_probs),
        "AUC-PR": average_precision_score(y_true, y_probs),
    }

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    return metrics


# def compute_tt_scores(model, queries_df, candidates_df, neural_network_type, include_relevancy = False):
#     rankings = []
#     device = 'cpu'

#     queries_dict = queries_df.set_index("qid")["query_embedding"].to_dict()  # Fast lookup

#     for _, passage_row in tqdm(candidates_df.iterrows(), total=len(candidates_df), desc=f"Scoring Passages with {neural_network_type}"):
#         qid = passage_row["qid"]
#         query_embedding = queries_dict[qid]  
#         passage_embedding = passage_row["passage_embedding"]

#         # Concatenate embeddings
#         combined_embedding = np.concatenate((query_embedding, passage_embedding))
#         input_tensor = torch.tensor(combined_embedding, dtype=torch.float32).unsqueeze(0).to(device)

#         with torch.no_grad():
#             score = model(input_tensor).cpu().item()

#         rankings.append((qid, passage_row["pid"], score))

#     # Convert to DataFrame and sort
#     rankings_df = pd.DataFrame(rankings, columns=["qid", "pid", "score"])
#     rankings_df["rank"] = rankings_df.groupby("qid")["score"].rank(ascending=False, method="first")
#     rankings_df["algoname"] = neural_network_type

#     return rankings_df.sort_values(by=["qid", "rank"])


def compute_tt_scores(model, queries_df, candidates_df, neural_network_type, include_relevancy=False):
    """
    Compute and rank passage relevance scores using the trained TwoTowerNet model.

    Parameters:
        model: Trained TwoTowerNet model.
        queries_df: DataFrame with query embeddings.
        candidates_df: DataFrame with passage embeddings.
        neural_network_type (str): Name of the model.
        include_relevancy (bool): Whether to include the 'relevancy' column in rankings.

    Returns:
        DataFrame: Ranked passages with columns [qid, pid, rank, score, algoname] (and relevancy if applicable).
    """

    rankings = []
    device = 'cpu'

    for _, query_row in tqdm(queries_df.iterrows(), total=len(queries_df), desc=f"Scoring Passages with {neural_network_type}"):
        qid = query_row["qid"]
        query_embedding = torch.tensor(query_row['query_embedding'], dtype=torch.float32).to(device)

        query_passages = candidates_df[candidates_df["qid"] == qid].copy()

        # Prepare input features (concatenation of query & passage embeddings)
        X_test = np.stack(query_passages["passage_embedding"].values)
        X_test = np.hstack((np.tile(query_embedding.cpu().numpy(), (X_test.shape[0], 1)), X_test))

        # Convert to tensor
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

        # Compute relevance scores
        with torch.no_grad():
            scores = model(X_test_tensor).cpu().numpy().flatten()

        query_passages["score"] = scores
        query_passages = query_passages.sort_values(by="score", ascending=False).reset_index(drop=True)

        query_passages["rank"] = query_passages.index + 1
        query_passages["algoname"] = neural_network_type
        query_passages["qid"] = query_passages["qid"].astype(str)
        query_passages["pid"] = query_passages["pid"].astype(str)

        # Define columns to include
        columns_to_include = ["qid", "pid", "rank", "score", "algoname"]
        if include_relevancy and "relevancy" in query_passages.columns:
            columns_to_include.append("relevancy")

        rankings.extend(query_passages[columns_to_include].values.tolist())

    return pd.DataFrame(rankings, columns=columns_to_include)

def compute_mAP_and_mNDCG(model, queries_df, passages_df, neural_network_type, include_relevancy=True):
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
    rankings = compute_tt_scores(model, queries_df, passages_df, neural_network_type = neural_network_type, include_relevancy=include_relevancy)

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

        ap = average_precision(relevancy_list)
        ndcg = ndcg_at_k(relevancy_list, k=10)

        query_metrics.append({'AP': ap, 'NDCG': ndcg})

    results_df = pd.DataFrame(query_metrics)
    mAP = results_df['AP'].mean()
    mNDCG = results_df['NDCG'].mean()

    return {"mAP": mAP, "mNDCG": mNDCG}

# Negative Sampling

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


if __name__ == "__main__": 

    glove_path = "glove.6B.300d.txt"
    embedding_dim = 300
    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load GloVe embeddings
    print("Loading GloVe embeddings...")
    glove_model = load_glove_embeddings(glove_path)

    # Load training and validation datasets
    dtype_mapping = {"qid": str, "pid": str, "query": str, "passage": str, "relevancy": int}

    print("Loading training data...")
    train_df = pd.read_csv("train_data.tsv", sep="\t", names=["qid", "pid", "query", "passage", "relevancy"], header=0, dtype=dtype_mapping)
    train_df = negative_sampling(train_df, neg_ratio = 3)

    print("Loading validation data...")
    val_df = pd.read_csv("validation_data.tsv", sep="\t", names=["qid", "pid", "query", "passage", "relevancy"], header=0, dtype=dtype_mapping)
    val_df = negative_sampling(val_df, neg_ratio = 3)

    # Process text embeddings
    print("Generating query embeddings...")
    tqdm.pandas(desc="Processing query embeddings")
    train_df["query_embedding"] = train_df["query"].progress_apply(lambda q: text_to_embedding_glove(q, glove_model))
    val_df["query_embedding"] = val_df["query"].progress_apply(lambda q: text_to_embedding_glove(q, glove_model))

    print("Generating passage embeddings...")
    tqdm.pandas(desc="Processing passage embeddings")
    train_df["passage_embedding"] = train_df["passage"].progress_apply(lambda p: text_to_embedding_glove(p, glove_model))
    val_df["passage_embedding"] = val_df["passage"].progress_apply(lambda p: text_to_embedding_glove(p, glove_model))

    # Prepare datasets and data loaders
    train_dataset = TwoTowerDataset(train_df, train_df)
    val_dataset = TwoTowerDataset(val_df, val_df)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize and train the model
    print("Initializing TwoTowerNet model...")
    model = TwoTowerNet(embedding_dim=embedding_dim).to(device)

    print("Training model...")
    model = train_two_tower(model, train_loader, val_loader, num_epochs=num_epochs, lr=learning_rate, device=device)

    # Evaluate the model
    print("Evaluating model...")
    evaluate_two_tower(model, val_loader, device=device)

    # Compute mAP and mNDCG on validation set
    print("Computing mAP and mNDCG...")
    val_queries_df = val_df[['qid', 'query']].drop_duplicates()
    val_passages_df = val_df[['qid', 'pid', 'passage', 'relevancy']]

    tqdm.pandas(desc="Processing Validation Queries")
    val_queries_df["query_embedding"] = val_queries_df["query"].progress_apply(lambda q: text_to_embedding_glove(q, glove_model))

    tqdm.pandas(desc="Processing Validation Passages")
    val_passages_df["passage_embedding"] = val_passages_df["passage"].progress_apply(lambda p: text_to_embedding_glove(p, glove_model))

    metrics = compute_mAP_and_mNDCG(model, val_queries_df, val_passages_df, neural_network_type="TwoTowerNet", include_relevancy=True)
    print(f"Validation Results - mAP: {metrics['mAP']:.4f}, mNDCG: {metrics['mNDCG']:.4f}")

    # Load test queries and candidate passages for ranking
    print("Loading test queries and candidate passages...")
    test_queries_df = pd.read_csv("test-queries.tsv", sep="\t", names=["qid", "query"])
    candidates_df = pd.read_csv("candidate_passages_top1000.tsv", sep="\t", names=["qid", "pid", "query", "passage"])

    print("Processing test query embeddings...")
    tqdm.pandas(desc="Processing...")
    test_queries_df["query_embedding"] = test_queries_df["query"].progress_apply(lambda q: text_to_embedding_glove(q, glove_model))

    print("Processing candidate passage embeddings...")
    tqdm.pandas(desc="Processing...")
    candidates_df["passage_embedding"] = candidates_df["passage"].progress_apply(lambda p: text_to_embedding_glove(p, glove_model))

    # Compute ranking scores
    print("Computing passage rankings...")
    rankings_df = compute_tt_scores(model, test_queries_df, candidates_df, neural_network_type="TwoTowerNet")

    # Save rankings
    output_file = "TwoTowerNet_Rankings.txt"
    rankings_df.to_csv(output_file, sep=" ", index=False, header=False)
    print(f"Rankings saved to {output_file}")