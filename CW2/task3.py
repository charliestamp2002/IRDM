import numpy as np  
import matplotlib.pyplot as plt
import pandas as pd
from task1 import TextProcessor  
import nltk
from nltk.corpus import stopwords
from task2 import InvertedIndex
from tqdm import tqdm
import time

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def compute_tf(tokens):

    tf = {}

    for token in tokens:
        tf[token] = tf.get(token, 0) + 1
        # Normalize TF by dividing by total number of tokens
    total_tokens = len(tokens)
    tf = {term: freq / total_tokens for term, freq in tf.items()}

    return tf

# Calculate IDF (Inverse Document Frequency): The logarithm of the total number of documents divided by the number of documents containing the term (plus 1 to avoid division by zero).

def compute_idf(inverted_index, total_documents):

    idf = {}

    for term, posting_list in inverted_index.items():
        doc_freq = len(posting_list)
        idf[term] = np.log(total_documents / doc_freq + 1)

    return idf


# Compute TF-IDF for Passages

def compute_tfidf_for_passages(inverted_index, idf, passage_lengths):
    
    tfidf_vecs = {}

    for term, posting_list in inverted_index.items():
        for pid, tf in posting_list.items():
            if pid not in tfidf_vecs:
                tfidf_vecs[pid] = {}

            normalised_tf = tf / passage_lengths[pid]

            tfidf_vecs[pid][term] = normalised_tf * idf[term]

    return tfidf_vecs

def compute_tfidf_for_query(query_terms, inverted_index, idf):
    """Compute TF-IDF vector for a single query (no preprocessing here)."""
    
    tf = {}
    for token in query_terms:
        tf[token] = tf.get(token, 0) + 1

    # Normalize TF
    total_tokens = sum(tf.values())
    tf = {term: freq / total_tokens for term, freq in tf.items()}

    # Compute TF-IDF
    tfidf = {term: tf.get(term, 0) * idf.get(term, 0) for term in idf}
    
    return tfidf


def cosine_similarity(tfidf_q, tfidf_p):
    """Compute cosine similarity between two TF-IDF vectors."""

    # Dot product
    numerator = sum(tfidf_q.get(term, 0) * tfidf_p.get(term, 0) for term in tfidf_q)

    # Magnitudes
    magnitude_q = np.sqrt(sum(val**2 for val in tfidf_q.values()))
    magnitude_p = np.sqrt(sum(val**2 for val in tfidf_p.values()))

    # Avoid division by zero
    if magnitude_q == 0 or magnitude_p == 0:
        return 0.0
    return numerator / (magnitude_q * magnitude_p)


def rank_passages(queries_df, passages_df, tfidf_passages, inverted_index, idf, vocabulary, stop_words):
    """Rank passages using TF-IDF with cosine similarity."""
    
    results = []
    for _, query_row in tqdm(queries_df.iterrows(), desc="Processing Queries"):
        qid = query_row['qid']
        query_text = query_row['query']

        # Preprocess query text in rank function 
        processor = TextProcessor(None, stop_words=stop_words)
        processor.raw_text = query_text
        processor.to_lowercase()
        processor.remove_punctuation()
        processor.tokenize()
        processor.stem_tokens()
        processor.remove_stopwords()
        query_terms = [word for word in processor.tokens if word in vocabulary]

        # Compute TF-IDF vector for the query
        query_tfidf = compute_tfidf_for_query(query_terms, inverted_index, idf)

        # Get candidate passages for this query
        candidate_passages = passages_df[passages_df['qid'] == qid].head(100)

        similarities = []
        for _, passage_row in candidate_passages.iterrows():
            pid = passage_row['pid']
            passage_tfidf = tfidf_passages.get(pid, {})
            similarity = cosine_similarity(query_tfidf, passage_tfidf)
            similarities.append((qid, pid, similarity))

        # Sort and take the top 100
        similarities = sorted(similarities, key=lambda x: x[2], reverse=True)[:100]
        results.extend(similarities)

    return results


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

def rank_passages_bm25(queries_df, passages_df, inverted_index, idf, passage_lengths, avg_passage_length):
    """Rank passages using BM25 scoring."""
    results = []

    for _, query_row in tqdm(queries_df.iterrows(), desc="Processing Queries"):
        qid = query_row['qid']
        query_text = query_row['query']

        # Preprocess query
        processor = TextProcessor(None, stop_words=stop_words)
        processor.raw_text = query_text
        processor.to_lowercase()
        processor.remove_punctuation()
        processor.tokenize()
        processor.stem_tokens()
        processor.remove_stopwords()
        query_terms = [word for word in processor.tokens if word in idf]

        # Get candidate passages for this query
        candidate_passages = passages_df[passages_df['qid'] == qid].head(100)

        similarities = []
        for _, passage_row in candidate_passages.iterrows():
            pid = passage_row['pid']
            passage = {term: inverted_index[term].get(pid, 0) for term in query_terms}
            passage_length = passage_lengths.get(pid, 0)

            # Compute BM25 score
            bm25_score = compute_bm25_score(query_terms, passage, passage_length, idf, avg_passage_length)
            similarities.append((qid, pid, bm25_score))

        # Sort and keep top 100
        similarities = sorted(similarities, key=lambda x: x[2], reverse=True)[:100]
        results.extend(similarities)

    return results

def save_results_to_csv(results, output_file):
    """Save the ranking results to tfidf.csv."""
    results_df = pd.DataFrame(results, columns=['qid', 'pid', 'score'])
    results_df.to_csv(output_file, index=False, header=False)


if __name__ == "__main__":

    start_time = time.time()
    # Step 1: Initialize and build the inverted index
    file_path = 'candidate-passages-top1000.tsv'
    passages_df = pd.read_csv(file_path, sep='\t', header=None, names=['qid', 'pid', 'query', 'passage'])
    processor = TextProcessor('passage-collection.txt', stop_words=stop_words)
    processor.process()
    vocabulary = processor.get_vocabulary()
    stop_words = set(stopwords.words('english'))  # Define stop words
    index = InvertedIndex(vocabulary=vocabulary, stop_words=stop_words)
    index.build_index(passages_df[['pid', 'passage']])
    
    #quantities needed for BM25
    total_documents = len(passages_df['pid'].unique())
    avg_passage_length = np.mean(list(index.passage_lengths.values()))
    # Step 2: Compute IDF values
    idf = compute_idf(index.inverted_index, total_documents=len(passages_df['pid'].unique()))
    

    # Step 3: Compute TF-IDF for passages
    tfidf_passages = compute_tfidf_for_passages(index.inverted_index, idf, index.passage_lengths)
    
    # Step 4: Load queries
    queries_file = 'test-queries.tsv'
    queries_df = pd.read_csv(queries_file, sep='\t', header=None, names=['qid', 'query'])

    # Step 5: Rank passages using TextProcessor for queries
    results = rank_passages(
        queries_df, passages_df, tfidf_passages, index.inverted_index, idf, vocabulary, stop_words
    )

    #Step 5.5 Rank passages using BM25
    results_bm25 = rank_passages_bm25(queries_df, passages_df, index.inverted_index, idf, index.passage_lengths, avg_passage_length)
    
    # Step 6: Save results to tfidf.csv & bm25.csv
    save_results_to_csv(results, 'tfidf.csv')
    save_results_to_csv(results_bm25, 'bm25.csv')

    print("results saved to tfidf.csv")
    print("results saved to bm25.csv")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")






