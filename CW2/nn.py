from lr import *
import torch.nn.functional as F


"""
DEFINING INPUT PROCESSING FUNCTIONS:
"""

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
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())  # Tokenize and lowercase (This part may be changed for a better tokenizer...)
    for word in tokens: 
        if word in glove_model:
            word_embeddings.append(glove_model[word])

    if len(word_embeddings) == 0:  # Handle case where no words are found
        return np.zeros(embedding_dim)
    
    return np.mean(word_embeddings, axis=0)  # Compute average embedding


"""

DEFINING NEURAL NETWORK ARCHITECTURES: 

CLASSIC FEED-FORWARD: 

"""



class QueryPassageDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # Convert to tensor
        self.y = torch.tensor(y, dtype=torch.float32)  # Convert to tensor

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

class FeedForward(nn.Module):
    def __init__(self, input_dim): 
        super(FeedForward, self).__init__()

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): 
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x 
    
def fit_feedforward(model, train_loader, val_loader, num_epochs = 10, lr = 1e-3, device = 'cpu'): 
    
    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr) # optimiser will be kept at Adam optimizer for Logistic Reg, and other Neural Network models for consistency
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

def evaluate_feedforward(model, data_loader, device = 'cpu'): 

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

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "AUC-ROC": auc_roc,
        "AUC-PR": auc_pr}



def compute_nn_scores(model, queries_df, candidates_df, neural_network_type, include_relevancy=False):
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

    for _, query_row in tqdm(queries_df.iterrows(), total=len(queries_df), desc=f"Scoring Passages with {neural_network_type}"):
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
            if isinstance(model, TwoTowerNN):
                scores = model(query_embedding.unsqueeze(0).expand(X_test_tensor.shape[0], -1), X_test_tensor).cpu().numpy().flatten()
            else:   
                scores = model(X_test_tensor).cpu().numpy().flatten()

        query_passages["score"] = scores
        query_passages = query_passages.sort_values(by="score", ascending=False).reset_index(drop=True)

        query_passages["rank"] = query_passages.index + 1
        query_passages["algoname"] = "NN"
        query_passages["qid"] = query_passages["qid"].astype(str)
        query_passages["pid"] = query_passages["pid"].astype(str)
        #rankings.extend(query_passages[["qid", "pid", "rank", "score", "algoname", "relevancy"]].values.tolist())

               # Select appropriate columns
        columns_to_include = ["qid", "pid", "rank", "score", "algoname"]
        if include_relevancy and "relevancy" in query_passages.columns:
            columns_to_include.append("relevancy")

        rankings.extend(query_passages[columns_to_include].values.tolist())

    #rankings_df = pd.DataFrame(rankings, columns=["qid", "pid", "rank", "score", "algoname", "relevancy"])

    return pd.DataFrame(rankings, columns=columns_to_include)


def compute_tt_scores(model, queries_df, candidates_df, neural_network_type, include_relevancy=False):
    rankings = []
    device = 'cpu'

    for _, query_row in tqdm(queries_df.iterrows(), total=len(queries_df), desc=f"Scoring Passages with {neural_network_type}"):
        qid = query_row["qid"]
        query_embedding = torch.tensor(query_row['query_embedding'], dtype=torch.float32).to(device)

        query_passages = candidates_df[candidates_df["qid"] == qid].copy()
        X_test = np.stack(query_passages["passage_embedding"].values)

        # ✅ Fix: Separate query and passage embeddings
        X_query = np.tile(query_embedding.cpu().numpy(), (X_test.shape[0], 1))  # Expand query embeddings
        X_passage = X_test  # Keep passage embeddings unchanged

        # Convert to tensors
        X_query_tensor = torch.tensor(X_query, dtype=torch.float32).to(device)
        X_passage_tensor = torch.tensor(X_passage, dtype=torch.float32).to(device)

        with torch.no_grad():
            scores = model(X_query_tensor, X_passage_tensor).cpu().numpy().flatten()

        query_passages["score"] = scores
        query_passages = query_passages.sort_values(by="score", ascending=False).reset_index(drop=True)

        query_passages["rank"] = query_passages.index + 1
        query_passages["algoname"] = neural_network_type
        query_passages["qid"] = query_passages["qid"].astype(str)
        query_passages["pid"] = query_passages["pid"].astype(str)

        columns_to_include = ["qid", "pid", "rank", "score", "algoname"]
        if include_relevancy and "relevancy" in query_passages.columns:
            columns_to_include.append("relevancy")

        rankings.extend(query_passages[columns_to_include].values.tolist())

    return pd.DataFrame(rankings, columns=columns_to_include)

"""
TWO TOWER FEEDFORWARD:
"""

class TwoTowerNN(nn.Module):
    def __init__(self, embedding_dim=300, hidden_dim=128, use_cosine=False):
        super(TwoTowerNN, self).__init__()
        self.use_cosine = use_cosine

        # Define Query Tower: 

        self.query_tower = nn.Sequential(nn.Linear(embedding_dim, hidden_dim),
                                         nn.ReLU(),
                                         nn.Linear(hidden_dim, hidden_dim),
                                         nn.ReLU())
        
        self.passage_tower = nn.Sequential(nn.Linear(embedding_dim, hidden_dim),
                                         nn.ReLU(),
                                         nn.Linear(hidden_dim, hidden_dim),
                                         nn.ReLU())
        
        if not self.use_cosine: 
            self.scoring_layer = nn.Linear(hidden_dim * 2, 1)

    def forward(self, query_embedding, passage_embedding): 

        query_vec = self.query_tower(query_embedding)
        passage_vec = self.passage_tower(passage_embedding)

        if self.use_cosine: 
            score = F.cosine_similarity(query_vec, passage_vec, dim = 1).unsqueeze(1) # should be shape (batch_size, 1)

        else: 
            comb = torch.cat((query_vec, passage_vec), dim = 1)
            score = torch.sigmoid(self.scoring_layer(comb))

        return score

class TwoTowerDataset(Dataset):
    def __init__(self, queries_df, passages_df):
        """
        Initializes a dataset for TwoTowerNN.
        
        Parameters:
            queries_df: DataFrame with query embeddings.
            passages_df: DataFrame with passage embeddings and labels.
        """
        self.queries = queries_df
        self.passages = passages_df

        # Ensure embeddings are numpy arrays
        self.queries["query_embedding"] = self.queries["query_embedding"].apply(lambda x: np.array(x))
        self.passages["passage_embedding"] = self.passages["passage_embedding"].apply(lambda x: np.array(x))

    def __len__(self):
        return len(self.passages)

    def __getitem__(self, idx):
        # query_embedding = self.queries.iloc[idx]["query_embedding"]
        # passage_embedding = self.passages.iloc[idx]["passage_embedding"]
        qid = self.passages.iloc[idx]["qid"]
        query_embedding = self.queries[self.queries["qid"] == qid]["query_embedding"].values[0]
        passage_embedding = self.passages.iloc[idx]["passage_embedding"]

        label = self.passages.iloc[idx]["relevancy"]

        return (
            torch.tensor(query_embedding, dtype=torch.float32),
            torch.tensor(passage_embedding, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )


def train_two_tower(model, train_loader, val_loader, num_epochs=10, lr=1e-3, device='cpu'):
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader: 
            query_embed, passage_embed, label = batch

        #for query_embed, passage_embed, label in train_loader:
            #query_embed, passage_embed, label = query_embed.to(device), passage_embed.to(device), label.to(device).view(-1, 1)

            optimizer.zero_grad()
            scores = model(query_embed, passage_embed)
            loss = criterion(scores, label.view(-1, 1)) # reshapes label to batch size (32 ,1)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # Validation 
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            #for query_embed, passage_embed, label in val_loader:
            for batch in val_loader:
                query_embed, passage_embed, label = batch

                #query_embed, passage_embed, label = query_embed.to(device), passage_embed.to(device), label.to(device).view(-1, 1)
                scores = model(query_embed, passage_embed)
                loss = criterion(scores, label.view(-1,1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    return model

def evaluate_two_tower(model, data_loader, device='cpu'):
    model.eval()
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for query_embed, passage_embed, label in data_loader:
            query_embed, passage_embed = query_embed.to(device), passage_embed.to(device)

            label = label.cpu().numpy()
            scores = model(query_embed, passage_embed).cpu().numpy().flatten()

            predictions = (scores >= 0.5).astype(int)

            y_true.extend(label)
            y_pred.extend(predictions.flatten())
            y_probs.extend(scores)

    # Compute Evaluation Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc_roc = roc_auc_score(y_true, y_probs)  # AUC-ROC
    auc_pr = average_precision_score(y_true, y_probs)  # AUC-PR

    # Print Results
    print(f" Model Evaluation Results:")
    print(f" Accuracy: {accuracy:.4f}")
    print(f" Precision: {precision:.4f}")
    print(f" Recall: {recall:.4f}")
    print(f" F1 Score: {f1:.4f}")
    print(f" AUC-ROC: {auc_roc:.4f}")
    print(f" AUC-PR: {auc_pr:.4f}")

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "AUC-ROC": auc_roc,
        "AUC-PR": auc_pr}



# --------------------------------------------- # 

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

#     rankings = compute_nn_scores(model, val_queries_df, val_passages_df)
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
    rankings = compute_nn_scores(model, queries_df, passages_df, neural_network_type = neural_network_type, include_relevancy=include_relevancy)

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


def save_rankings(rankings_df, output_file):
    """
    Save rankings in the required six-column format.
    
    Format: <qid> A2 <pid> <rank> <score> <algoname>
    """
    rankings_df["A2"] = "A2"  # Static column
    rankings_df = rankings_df[["qid", "A2", "pid", "rank", "score", "algoname"]]
    
    rankings_df.to_csv(output_file, sep=" ", index=False, header=False)
    print(f"Logistic Regression rankings saved to {output_file}")

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


if __name__ == "__main__": 
    
    glove_path = "glove.6B.300d.txt"
    glove_model = load_glove_embeddings(glove_path)

    dtype_mapping = {"qid": str, "pid": str, "query": str, "passage": str, "relevancy": int}

    # Load training data
    train_file = "train_data.tsv"
    train_df = pd.read_csv(train_file, sep = "\t", names=["qid", "pid", "query", "passage", "relevancy"], header = 0, dtype = dtype_mapping)


    # Once up and running need to make sure the train_df is the same for all models.
    # Reduce to 10% of the data to keep within RAM restrictions (random sampling)
    # train_df = train_df.sample(frac=0.1, random_state=42).reset_index(drop=True)
    # Use negative sampling instead as in Logistic Regression
    train_df = negative_sampling(train_df, neg_ratio = 3)

    val_file = "validation_data.tsv"
    val_df = pd.read_csv(val_file, sep="\t", names=["qid", "pid", "query", "passage", "relevancy"], header = 0, dtype = dtype_mapping)
    val_df = negative_sampling(val_df, neg_ratio = 3)

    tqdm.pandas(desc="Processing Queries")
    train_df["query_embedding"] = train_df["query"].progress_apply(lambda q: text_to_embedding_glove(q, glove_model))

    tqdm.pandas(desc="Processing Passages")
    train_df["passage_embedding"] = train_df["passage"].progress_apply(lambda p: text_to_embedding_glove(p, glove_model))

    tqdm.pandas(desc="Processing Validation Queries")
    val_df["query_embedding"] = val_df["query"].progress_apply(lambda q: text_to_embedding_glove(q, glove_model))

    tqdm.pandas(desc="Processing Validation Passages")
    val_df["passage_embedding"] = val_df["passage"].progress_apply(lambda p: text_to_embedding_glove(p, glove_model))

    model_type = "TwoTowerNN"

    if model_type == "FeedForward":
        X_train = np.hstack([train_df["query_embedding"].tolist(), train_df["passage_embedding"].tolist()])
        y_train = train_df["relevancy"].values  # Binary labels

        X_val = np.hstack([val_df["query_embedding"].tolist(), val_df["passage_embedding"].tolist()])
        y_val = val_df["relevancy"].values  # Binary labels

        train_dataset = QueryPassageDataset(X_train, y_train)
        val_dataset = QueryPassageDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)

        input_dim = X_train.shape[1]
        model = FeedForward(input_dim)

        train_losses, val_losses = fit_feedforward(model, train_loader, val_loader, num_epochs=50, lr=0.001, device="cpu")

        # Evaluate model
        evaluate_feedforward(model, val_loader)
    
    elif model_type == "TwoTowerNN": 

        train_dataset = TwoTowerDataset(train_df, train_df)
        val_dataset = TwoTowerDataset(val_df, val_df)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)

        model = TwoTowerNN()

        model = train_two_tower(model, train_loader, val_loader, num_epochs=3, lr=0.001, device="cpu")

        # Evaluate model
        evaluate_two_tower(model, val_loader)

    # # Train Feedforward model: 
    # input_dim = 2 * 300
    # model = FeedForward(input_dim)

    # X_train = np.hstack([train_df["query_embedding"].tolist(), train_df["passage_embedding"].tolist()])
    # y_train = train_df["relevancy"].values  # Binary labels (0 or 1)

    # X_val = np.hstack([val_df["query_embedding"].tolist(), val_df["passage_embedding"].tolist()])
    # y_val = val_df["relevancy"].values  # Binary labels (0 or 1)    

    # train_dataset = QueryPassageDataset(X_train, y_train)
    # val_dataset = QueryPassageDataset(X_val, y_val)

    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=32)

    # fit_feedforward(model, train_loader, val_loader)
    # evaluate_feedforward(model, val_loader)

    ### Load Test Data for Ranking ###
    test_queries_file = "test-queries.tsv"
    candidates_file = "candidate_passages_top1000.tsv"

    test_queries_df = pd.read_csv(test_queries_file, sep="\t", names=["qid", "query"])
    candidates_df = pd.read_csv(candidates_file, sep="\t", names=["qid", "pid", "query", "passage"])

    tqdm.pandas(desc="Processing Test Queries")
    test_queries_df["query_embedding"] = test_queries_df["query"].progress_apply(lambda q: text_to_embedding_glove(q, glove_model))

    tqdm.pandas(desc="Processing Candidate Passages")
    candidates_df["passage_embedding"] = candidates_df["passage"].progress_apply(lambda p: text_to_embedding_glove(p, glove_model))

    # Compute rankings using the chosen model
    if model_type == "FeedForward":
        nn_rankings = compute_nn_scores(model, test_queries_df, candidates_df, neural_network_type=model_type, include_relevancy=False)
        save_rankings(nn_rankings, f"{model_type}_Rankings.txt")

    elif model_type == "TwoTower": 
        nn_rankings = compute_tt_scores(model, test_queries_df, candidates_df, neural_network_type=model_type, include_relevancy=False)
        save_rankings(nn_rankings, f"{model_type}_Rankings.txt")


    ### Compute mAP & mNDCG ###
    val_queries_df = val_df[['qid', 'query']].drop_duplicates()

    tqdm.pandas(desc="Processing Validation Queries")
    val_queries_df["query_embedding"] = val_queries_df["query"].progress_apply(lambda q: text_to_embedding_glove(q, glove_model))

    val_passages_df = val_df[['qid', 'pid', 'passage', 'relevancy']]
    val_passages_df["passage_embedding"] = val_passages_df["passage"].progress_apply(lambda p: text_to_embedding_glove(p, glove_model))

    print("Validation Queries Columns:", val_queries_df.columns)  
    print("Validation Passages Columns:", val_passages_df.columns)  

    print("Missing query embeddings:", val_queries_df["query_embedding"].isna().sum())  
    print("Missing passage embeddings:", val_passages_df["passage_embedding"].isna().sum())  

    if "relevancy" not in val_passages_df.columns:
        raise ValueError("Error: 'relevancy' column is missing in validation passages!")

    # Compute mAP and mNDCG
    metrics = compute_mAP_and_mNDCG(model, val_queries_df, val_passages_df, neural_network_type = model_type, include_relevancy=True)
    print(metrics)

    print("Training, Evaluation, and Metrics Calculation Complete!")












