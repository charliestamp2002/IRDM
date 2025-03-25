import numpy as np
import matplotlib.pyplot as plt
import re
import time 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import pandas as pd
from tqdm import tqdm
import os
os.chdir("/Users/charliestamp/Documents/IRDM/CW2")
from nltk.tokenize import word_tokenize
nltk.download("punkt")
nltk.download("wordnet")
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

from nltk.stem import WordNetLemmatizer # Use lemmatisation for now (this is optional and can be changed)
lemmatizer = WordNetLemmatizer()

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def tokenize_and_lemmatize(text):
    """Tokenizes, lowercases, removes stopwords, and lemmatizes the input text."""
    tokens = word_tokenize(text.lower())  # Tokenization + lowercasing
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]  
    return tokens
    
class InvertedIndex:
    def __init__(self, vocabulary, stop_words=None):
        self.vocabulary = vocabulary  # Vocabulary from Task 1
        self.inverted_index = {}  # The inverted index structure
        self.passage_lengths = {}  # Map each passage ID (pid) to its length
        #self.stop_words = set(stop_words) if stop_words else set()  # Optional stop words set

    def build_index(self, df):
        """Builds the inverted index from a DataFrame with 'pid' and 'passage' columns."""

        start_time = time.time()

        for _, row in df.iterrows():
            pid = row['pid']
            passage = row['passage']
            tokens = tokenize_and_lemmatize(passage)  # Preprocess and tokenize the passage

            # Store the passage length (number of tokens)
            self.passage_lengths[pid] = len(tokens)

            # Calculate term frequencies for this passage
            term_frequencies = {}
            for token in tokens:
                if token in term_frequencies:
                    term_frequencies[token] += 1
                else:
                    term_frequencies[token] = 1

            # Update the inverted index
            for token, tf in term_frequencies.items():
                if token not in self.inverted_index:
                    self.inverted_index[token] = {}  # Use a dictionary to store {pid: tf}
                self.inverted_index[token][pid] = tf

        end_time = time.time()
        print(f"Building the inverted index took {end_time - start_time:.2f} seconds")


def compute_bm25_idf(inverted_index, total_documents):
    """Compute BM25 IDF values for each term."""
    idf = {}
    
    for term, posting_list in inverted_index.items():
        doc_freq = len(posting_list)  # Number of documents containing term
        idf[term] = np.log((total_documents - doc_freq + 0.5) / (doc_freq + 0.5) + 1)

    return idf

def compute_bm25_score(query_terms, passage, passage_length, idf, avg_passage_length, k1=1.2, k2=100, b=0.75):
    """Compute the BM25 score between a query and a passage"""
    score = 0

    for term in query_terms:
        if term in idf:  # Ignore terms not found in passages
            tf = passage.get(term, 0)  # Get term frequency in passage
            qtf = query_terms.count(term)  # Query term frequency (important for k2)
            term_idf = idf[term]

            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (passage_length / avg_passage_length))
            passage_component = term_idf * (numerator / denominator)
            query_component = (qtf * (k2 + 1)) / (qtf + k2)

            # Final BM25 score for this term
            score += passage_component * query_component

    return score

def extract_vocabulary(validation_df, stop_words):
    """Extract vocabulary from the passages in validation_data.tsv."""
    # processor = TextProcessor(None, stop_words=stop_words)
    vocabulary = set()

    for passage in tqdm(validation_df['passage'], desc="Extracting Vocabulary"):
        # processor.raw_text = passage
        # processor.to_lowercase()
        # processor.remove_punctuation()
        # processor.tokenize()
        # processor.stem_tokens()
        # processor.remove_stopwords()

        passage = tokenize_and_lemmatize(passage)
        vocabulary.update(passage)  # Add words to vocabulary

    return vocabulary

def rank_passages_bm25(queries_df, validation_df, inverted_index, idf, passage_lengths, avg_passage_length):
    """Rank passages using BM25 scoring."""
    results = []

    #processor = TextProcessor(None, stop_words=stop_words)

    for _, query_row in tqdm(queries_df.iterrows(), desc="Processing Queries"):
        qid = query_row['qid']
        query_text = query_row['queries']

        # # Preprocess query
        # processor.raw_text = query_text
        # processor.to_lowercase()
        # processor.remove_punctuation()
        # processor.tokenize()
        # processor.stem_tokens()
        # processor.remove_stopwords()
        query_terms = tokenize_and_lemmatize(query_text)
        query_terms = [word for word in query_terms if word in idf]

        # Get candidate passages for this query
        candidate_passages = validation_df[validation_df['qid'] == qid]

        similarities = []
        for _, passage_row in candidate_passages.iterrows():
            pid = passage_row['pid']
            passage_tf = {term: inverted_index[term].get(pid, 0) for term in query_terms}
            passage_length = passage_lengths.get(pid, 0)

            # Compute BM25 score
            bm25_score = compute_bm25_score(query_terms, passage_tf, passage_length, idf, avg_passage_length)
            similarities.append((qid, pid, bm25_score))

        # Sort and keep top 100
        similarities = sorted(similarities, key=lambda x: x[2], reverse=True)
        results.extend(similarities)

    return results

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


def evaluate_bm25(validation_file, bm25_results_file):

    validation_df = pd.read_csv(
    validation_file, 
    sep='\t', 
    header=0,  # This tells pandas the first row contains column names
    names=['qid', 'pid', 'queries', 'passage', 'relevancy'])

    # Convert relevancy column to integer (0 or 1)
    validation_df['relevancy'] = validation_df['relevancy'].astype(int)

    bm25_df = pd.read_csv(bm25_results_file, header=None, names=['qid', 'pid', 'score'])

    # Debugging: Ensure BM25 ranking is not empty
    print(f"BM25 Results Shape: {bm25_df.shape}")
    if bm25_df.shape[0] == 0:
        print("Warning: BM25 ranking returned no results.")
        return {'MAP': 0.0, 'NDCG@10': 0.0}

    # Print number of relevant documents for debugging
    print("Relevance Score Distribution:\n", validation_df['relevancy'].value_counts())

    # Check if there are relevant documents
    if validation_df['relevancy'].sum() == 0:
        print("Warning: No relevant documents found in validation set. MAP and NDCG will be 0.")
        return {'MAP': 0.0, 'NDCG@10': 0.0}


    # Merge BM25 rankings with relevance labels
    merged_df = bm25_df.merge(validation_df[['qid', 'pid', 'relevancy']], on=['qid', 'pid'], how='left').fillna(0)

    # Debugging: Ensure merged_df has relevancy
    print("Merged Data Sample:\n", merged_df.head())
    print("Merged Data Columns:", merged_df.columns)

    # Compute evaluation metrics
    query_metrics = []

    for qid, group in tqdm(merged_df.groupby('qid'), total=len(merged_df['qid'].unique()), desc="Computing AP & NDCG"):        # Sort group by BM25 score
        sorted_group = group.sort_values(by='score', ascending=False)
        relevance_labels = sorted_group['relevancy'].tolist()

        # Compute AP and NDCG
        ap = average_precision(relevance_labels)
        ndcg = ndcg_at_k(relevance_labels, k=50)

        query_metrics.append({'qid': qid, 'AP': ap, 'NDCG@k': ndcg})

    # Debugging: Ensure query_metrics is not empty
    print(f"Number of queries processed: {len(query_metrics)}")
    if len(query_metrics) == 0:
        print("Warning: No queries processed. Check input files.")
        return {'MAP': 0.0, 'NDCG@10': 0.0}

    # Convert to DataFrame and compute mean metrics
    results_df = pd.DataFrame(query_metrics)
    mean_ap = results_df['AP'].mean()
    mean_ndcg = results_df['NDCG@k'].mean()

    print(f"Mean Average Precision (MAP): {mean_ap:.4f}")
    print(f"Mean NDCG@k: {mean_ndcg:.4f}")

    return {'MAP': mean_ap, 'NDCG@k': mean_ndcg}


def save_results_to_csv(results, output_file):
    """Save BM25 ranking results."""
    results_df = pd.DataFrame(results, columns=['qid', 'pid', 'score'])
    results_df.to_csv(output_file, index=False, header=False)


if __name__ == "__main__":

    print(os.listdir("."))
    print(os.getcwd()) 

    print("Starting Information Retrieval Pipeline...")
    start_time = time.time()

    # Load validation dataset for proper ranking
    print("Loading validation dataset...")
    validation_file = "validation_data.tsv"
    validation_df = pd.read_csv(validation_file, sep='\t', header=0, names=['qid', 'pid', 'queries', 'passage', 'relevancy'])

    #Excluding stop words from analysis as in cw1: 
    # print("Building vocabulary from passage collection...")
    # stop_words = set(stopwords.words('english'))  # Define stop words
    # processor = TextProcessor('passage_collection_new.txt', stop_words=stop_words)
    # processor.process()
    # vocabulary = processor.get_vocabulary()

    print("Extracting vocabulary from validation_data.tsv...")
    vocabulary = extract_vocabulary(validation_df, stop_words)

    # Build the inverted index only for validation passages
    # print("Building inverted index for validation passages...")
    # index = InvertedIndex(vocabulary=vocabulary, stop_words=stop_words)
    # index.build_index(validation_df[['pid', 'passage']])
    print("Building inverted index from validation_data.tsv...")
    index = InvertedIndex(vocabulary=set(), stop_words=stop_words)
    index.build_index(validation_df[['pid', 'passage']])

    #quantities needed for BM25
    print("Computing BM25 IDF values...")
    total_documents = len(validation_df['pid'].unique())
    avg_passage_length = np.mean(list(index.passage_lengths.values()))
    bm25_idf = compute_bm25_idf(index.inverted_index, total_documents=total_documents)

    queries_df = validation_df[['qid', 'queries']].drop_duplicates()

    #Rank passages using BM25
    print("Running BM25 ranking...")
    results_bm25 = rank_passages_bm25(queries_df, validation_df, index.inverted_index, bm25_idf, index.passage_lengths, avg_passage_length)
    save_results_to_csv(results_bm25, "bm25_rankings.csv")

    # Evaluate BM25 metrics
    print("Evaluating BM25 performance...")
    metrics = evaluate_bm25("validation_data.tsv", "bm25_rankings.csv")
    print(metrics)

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")











    

