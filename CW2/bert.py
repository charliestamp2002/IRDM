import torch
import torch.nn as nn
from transformers import DistilBertModel
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer
from lr import average_precision, ndcg_at_k, save_rankings, negative_sampling
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, accuracy_score
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.optim as optim

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

class PairwiseBERT(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased'):
        super(PairwiseBERT, self).__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return self.classifier(cls_output)
    
class PairwiseBERTDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        query = row['query']
        passage = row['passage']
        label = row['relevancy']

        inputs = tokenizer(
            query,
            passage,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float)
        }
    
# class SingleQueryBERTDataset(Dataset):
#     def __init__(self, query, passages_df, tokenizer):
#         self.query = query
#         self.passages = passages_df
#         self.tokenizer = tokenizer

#     def __len__(self):
#         return len(self.passages)

#     def __getitem__(self, idx):
#         passage = self.passages.iloc[idx]["passage"]
#         pid = self.passages.iloc[idx]["pid"]

#         inputs = self.tokenizer(
#             self.query,
#             passage,
#             padding='max_length',
#             truncation=True,
#             max_length=128,
#             return_tensors='pt'
#         )

#         return {
#             'input_ids': inputs['input_ids'].squeeze(0),
#             'attention_mask': inputs['attention_mask'].squeeze(0),
#             'pid': pid  # keep track of passage ID
#         }

class SingleQueryBERTDataset(Dataset):
    def __init__(self, query, passages_df):
        self.query = query
        self.passages = passages_df

    def __len__(self):
        return len(self.passages)

    def __getitem__(self, idx):
        passage = self.passages.iloc[idx]["passage"]
        pid = self.passages.iloc[idx]["pid"]

        inputs = tokenizer(
            self.query,
            passage,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'pid': pid  # keep track of passage ID
        }
    
    
def train(model, dataloader, optimizer, criterion, device, epoch = None, total_epochs = None):
    model.train()
    total_loss = 0

    desc = f"Training Epoch {epoch+1}/{total_epochs}" if epoch is not None else "Training"
    progress_bar = tqdm(dataloader, desc=desc, leave=False)

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device).view(-1, 1)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    y_true, y_pred, y_probs = [], [], []
    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].numpy()
            probs = torch.sigmoid(model(input_ids, mask)).squeeze().cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            y_true.extend(labels)
            y_pred.extend(preds)
            y_probs.extend(probs)

    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
        'AUC-ROC': roc_auc_score(y_true, y_probs),
        'AUC-PR': average_precision_score(y_true, y_probs)
    }

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    return metrics

# def compute_bert_scores(model, tokenizer, queries_df, candidates_df, include_relevancy=False, batch_size=32):
#     rankings = []
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     model.eval()

#     for _, query_row in tqdm(queries_df.iterrows(), total=len(queries_df), desc="Scoring Passages with Pairwise BERT"):
#         qid = query_row["qid"]
#         query = query_row["query"]
#         query_passages = candidates_df[candidates_df["qid"] == qid].copy().reset_index(drop=True)

#         # dataset = PairwiseBERTDataset(query, query_passages, tokenizer)
#         dataset = SingleQueryBERTDataset(query, query_passages, tokenizer)
#         dataloader = DataLoader(dataset, batch_size=batch_size)

#         scores = []
#         pids = []

#         with torch.no_grad():
#             for batch in dataloader:
#                 input_ids = batch['input_ids'].to(device)
#                 attention_mask = batch['attention_mask'].to(device)

#                 outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#                 #logits = outputs.logits.squeeze(-1).cpu().numpy()
#                 logits = outputs.squeeze(-1).cpu().numpy()

#                 scores.extend(logits.tolist())
#                 pids.extend(batch['pid'])

#         query_passages["score"] = scores
#         query_passages["pid"] = pids
#         query_passages = query_passages.sort_values(by="score", ascending=False).reset_index(drop=True)
#         query_passages["rank"] = query_passages.index + 1
#         query_passages["algoname"] = "NN"
#         query_passages["qid"] = query_passages["qid"].astype(str)
#         query_passages["pid"] = query_passages["pid"].astype(str)

#         columns_to_include = ["qid", "pid", "rank", "score", "algoname"]
#         if include_relevancy and "relevancy" in query_passages.columns:
#             columns_to_include.append("relevancy")

#         rankings.extend(query_passages[columns_to_include].values.tolist())

#     return pd.DataFrame(rankings, columns=columns_to_include)


def compute_bert_scores(model, queries_df, candidates_df, include_relevancy=False, batch_size=32):
    rankings = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    for _, query_row in tqdm(queries_df.iterrows(), total=len(queries_df), desc="Scoring Passages with Pairwise BERT"):
        qid = query_row["qid"]
        query = query_row["query"]
        query_passages = candidates_df[candidates_df["qid"] == qid].copy().reset_index(drop=True)

        # dataset = PairwiseBERTDataset(query, query_passages, tokenizer)
        dataset = SingleQueryBERTDataset(query, query_passages)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        scores = []
        pids = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                #logits = outputs.logits.squeeze(-1).cpu().numpy()
                logits = outputs.squeeze(-1).cpu().numpy()

                scores.extend(logits.tolist())
                pids.extend(batch['pid'])

        query_passages["score"] = scores
        query_passages["pid"] = pids
        query_passages = query_passages.sort_values(by="score", ascending=False).reset_index(drop=True)
        query_passages["rank"] = query_passages.index + 1
        query_passages["algoname"] = "NN"
        # query_passages["qid"] = query_passages["qid"].astype(str)
        # query_passages["pid"] = query_passages["pid"].astype(str)

        columns_to_include = ["qid", "pid", "rank", "score", "algoname"]
        if include_relevancy and "relevancy" in query_passages.columns:
            columns_to_include.append("relevancy")

        rankings.extend(query_passages[columns_to_include].values.tolist())

    return pd.DataFrame(rankings, columns=columns_to_include)

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
    rankings = compute_bert_scores(model, queries_df, passages_df, include_relevancy=include_relevancy)

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
    print(f"NN rankings saved to {output_file}")

if __name__ == "__main__": 

    dtype_mapping = {"qid": str, "pid": str, "query": str, "passage": str, "relevancy": int}

    train_df = pd.read_csv("train_data.tsv", sep="\t", names=["qid", "pid", "query", "passage", "relevancy"], header=0, dtype=dtype_mapping)
    train_df = negative_sampling(train_df, neg_ratio=5)

    val_df = pd.read_csv("validation_data.tsv", sep="\t", names=["qid", "pid", "query", "passage", "relevancy"], header=0, dtype=dtype_mapping)
    val_df = negative_sampling(val_df, neg_ratio=5)

    train_dataset = PairwiseBERTDataset(train_df)
    val_dataset = PairwiseBERTDataset(val_df)
        
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    device = torch.device("cpu")

    model = PairwiseBERT().to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.BCEWithLogitsLoss()

    print("Training Pairwise BERT...")
    for epoch in range(1):
        loss = train(model, train_loader, optimizer, criterion, device, epoch = epoch, total_epochs = 1)
        print(f"Epoch {epoch+1} - Loss: {loss:.4f}")
        evaluate(model, val_loader, device)

    # Compute rankings for test set
    test_queries_df = pd.read_csv("test-queries.tsv", sep="\t", names=["qid", "query"])
    candidates_df = pd.read_csv("candidate_passages_top1000.tsv", sep="\t", names=["qid", "pid", "query", "passage"])

    bert_rankings = compute_bert_scores(model, test_queries_df, candidates_df)
    save_rankings(bert_rankings, "BERT.txt")

    val_queries_df = val_df[["qid", "query"]].drop_duplicates()
    val_passages_df = val_df[["qid", "pid", "passage", "relevancy"]]

    metrics = compute_mAP_and_mNDCG(model, val_queries_df, val_passages_df, include_relevancy=True)
    print(f"Validation Results - mAP: {metrics['mAP']:.4f}, mNDCG: {metrics['mNDCG']:.4f}")




        