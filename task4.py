import numpy as np  
import matplotlib.pyplot as plt
import pandas as pd
from task1 import TextProcessor  
import nltk
from nltk.corpus import stopwords
from task2 import InvertedIndex
from tqdm import tqdm

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def compute_collection_statistics(inverted_index, passage_lengths): 
    """Compute collection-level statistics for Dirichlet smoothing."""
    collection_frequency = {}
    total_tokens = 0

    for term, posting_list in inverted_index.items():
        collection_frequency[term] = sum(posting_list.values())
        total_tokens += collection_frequency[term]

    return collection_frequency, total_tokens

def laplace_smoothing(query_terms, passage, passage_length, vocabulary_size): 
    score = 0 

    for term in query_terms: 

        term_count = passage.get(term, 0)
        smooth_prob = (term_count + 1) / (passage_length + vocabulary_size)

        score += np.log(smooth_prob)

    return score

def lidstone_smoothing(query_terms, passage, passage_length, vocabulary_size, epsilon=0.1):
    
    score = 0
    for term in query_terms:
        term_count = passage.get(term, 0)
        smoothed_prob = (term_count + epsilon) / (passage_length + epsilon * vocabulary_size)
        score += np.log(smoothed_prob)
    return score

def dirichlet_smoothing(query_terms, passage, passage_length, collection_frequency, total_tokens, mu=50):
    score = 0
    for term in query_terms:
        term_count = passage.get(term, 0)
        collection_prob = collection_frequency.get(term, 0) / total_tokens
        smoothed_prob = (term_count + mu * collection_prob) / (passage_length + mu)
        score += np.log(smoothed_prob)
    return score

def rank_passages(queries_df, inverted_index, passage_lengths, collection_frequency, total_tokens, vocabulary_size, smoothing_method, **kwargs):
    results = []

    for _, query_row in tqdm(queries_df.iterrows(), desc="Processing Queries", total=len(queries_df)):
        qid = query_row['qid']
        query_text = query_row['query']

        # Preprocess the query
        processor = TextProcessor(None, stop_words=stop_words)
        processor.raw_text = query_text

        processor.to_lowercase()
        processor.remove_punctuation()
        processor.tokenize()
        processor.stem_tokens()
        processor.remove_stopwords()
        query_terms = [term for term in processor.tokens if term in vocabulary]

        # Get candidate passages for this query
        candidate_passages = passages_df[passages_df['qid'] == qid]


        similarities = []
        for _, passage_row in candidate_passages.iterrows():
            pid = passage_row['pid']

            # Build the passage term frequency dictionary from the inverted index
            passage = {term: inverted_index[term].get(pid, 0) for term in query_terms}
            passage_length = passage_lengths.get(pid, 0)

            # Calculate the query likelihood score using the selected smoothing method
            if smoothing_method == dirichlet_smoothing:
                score = smoothing_method(query_terms, passage, passage_length, collection_frequency, total_tokens, **kwargs)
            else:
                score = smoothing_method(query_terms, passage, passage_length, vocabulary_size, **kwargs)

            similarities.append((qid, pid, score))

        # Sort and keep the top 100
        similarities = sorted(similarities, key=lambda x: x[2], reverse=True)[:100]
        results.extend(similarities)
    
    return results


# Save Results to CSV
def save_results_to_csv(results, output_file):
    """Save the ranking results to a CSV file."""
    results_df = pd.DataFrame(results, columns=['qid', 'pid', 'log-score'])
    results_df.to_csv(output_file, index=False, header=False)


# Function to extract top passages for a specific query ID across smoothing methods
def compare_smoothing_methods(qid, results_files):
    """Compare the top 5 passages for a specific query across different smoothing methods."""
    
    print(f"\n=== Query ID: {qid} - {queries_df[queries_df['qid'] == qid]['query'].values[0]} ===\n")

    for method_name, file_path in results_files.items():
        print(f"\n--- Top 5 Passages for {method_name} Smoothing ---\n")
        
        # Load  results file
        results_df = pd.read_csv(file_path, names=['qid', 'pid', 'log-score'])

        # Filter results for the specific query ID and sort by highest log-score
        top_results = results_df[results_df['qid'] == qid].sort_values(by='log-score', ascending=False).head(5)

        # Print each passage with its score
        for _, row in top_results.iterrows():
            pid = row['pid']
            score = row['log-score']

            # Find the passage text
            passage_text = passages_df[passages_df['pid'] == pid]['passage'].values[0]

            print(f"Passage (PID: {pid}): {passage_text}")
            print(f"Score: {score}")
            print("-" * 80)

def preprocess_query(query_id, queries_df, stop_words):
    """
    Preprocesses a query given its ID using the TextProcessor class.

    Parameters:
        query_id (int): The ID of the query to preprocess.
        queries_df (DataFrame): The DataFrame containing queries with columns ['qid', 'query'].
        stop_words (set): The set of stop words to remove.

    Returns:
        list: The processed tokens of the query.
    """
    # Retrieve the query text based on query ID
    query_row = queries_df[queries_df['qid'] == query_id]
    
    if query_row.empty:
        print(f"Query ID {query_id} not found.")
        return None

    query_text = query_row['query'].values[0]

    # Initialize the text processor
    processor = TextProcessor(None, stop_words=stop_words)
    processor.raw_text = query_text

    # Perform all preprocessing steps
    processor.to_lowercase()
    processor.remove_punctuation()
    processor.tokenize()
    processor.stem_tokens()
    processor.remove_stopwords()

    return processor.tokens  # Return processed query tokens


  
if __name__ == "__main__":
    # Load passages and queries
    passages_file = 'candidate-passages-top1000.tsv'
    passages_df = pd.read_csv(passages_file, sep='\t', header=None, names=['qid', 'pid', 'query', 'passage'])
    queries_file = 'test-queries.tsv'
    queries_df = pd.read_csv(queries_file, sep='\t', header=None, names=['qid', 'query'])

    # Build inverted index and compute collection statistics
    processor = TextProcessor('passage-collection.txt', stop_words=stop_words)
    processor.process()
    vocabulary = processor.get_vocabulary()

    index = InvertedIndex(vocabulary=vocabulary, stop_words=stop_words)
    index.build_index(passages_df[['pid', 'passage']])

    collection_frequency, total_tokens = compute_collection_statistics(index.inverted_index, index.passage_lengths)
    vocabulary_size = len(vocabulary)

   
    # Laplace_smoothing(query_terms, passage, passage_length, vocabulary_size):
    
    laplace_results = rank_passages(
        queries_df=queries_df, 
        inverted_index=index.inverted_index, 
        passage_lengths=index.passage_lengths, 
        collection_frequency=None,  # Not used for Laplace Smoothing
        total_tokens=None,  # Not used for Laplace Smoothing
        vocabulary_size=vocabulary_size, 
        smoothing_method=laplace_smoothing
    )
    save_results_to_csv(laplace_results, 'laplace.csv')

    # Lidstone Smoothing
    lidstone_results = rank_passages(
        queries_df=queries_df, 
        inverted_index=index.inverted_index, 
        passage_lengths=index.passage_lengths, 
        collection_frequency=None,  # Not used for Lidstone Smoothing
        total_tokens=None,  # Not used for Lidstone Smoothing
        vocabulary_size=vocabulary_size, 
        smoothing_method=lidstone_smoothing, epsilon=0.1
    )
    save_results_to_csv(lidstone_results, 'lidstone.csv')

    # Dirichlet Smoothing
    dirichlet_results = rank_passages(
        queries_df=queries_df, 
        inverted_index=index.inverted_index, 
        passage_lengths=index.passage_lengths, 
        collection_frequency=collection_frequency,  # Used for Dirichlet Smoothing
        total_tokens=total_tokens,  # Used for Dirichlet Smoothing
        vocabulary_size=vocabulary_size, 
        smoothing_method=dirichlet_smoothing, mu=50
    )
    save_results_to_csv(dirichlet_results, 'dirichlet.csv')

    print("Results saved: laplace.csv, lidstone.csv, dirichlet.csv")

    # Load test queries and passages
    queries_file = 'test-queries.tsv'
    passages_file = 'candidate-passages-top1000.tsv'

    queries_df = pd.read_csv(queries_file, sep='\t', header=None, names=['qid', 'query'])
    passages_df = pd.read_csv(passages_file, sep='\t', header=None, names=['qid', 'pid', 'query', 'passage'])

    smoothing_methods = {
    "Laplace": "laplace.csv",
    "Lidstone": "lidstone.csv",
    "Dirichlet": "dirichlet.csv"
    }


    # Specify the Query ID you want to compare

    # Table 1 Query ID
    query_id_to_compare_table1 = 1134787

    # Table 2 Query ID
    query_id_to_compare_table2 = 1107988

    # TABLE 1:
    processed_query_table1 = preprocess_query(query_id_to_compare_table1, queries_df, stop_words)
    print(f"Processed query tokens for ID {query_id_to_compare_table1}: {processed_query_table1}")

    # Run the comparison
    compare_smoothing_methods(query_id_to_compare_table1, smoothing_methods)

    # TABLE 2: 
    processed_query_table2 = preprocess_query(query_id_to_compare_table2, queries_df, stop_words)
    print(f"Processed query tokens for ID {query_id_to_compare_table2}: {processed_query_table2}")

    # Run the comparison
    compare_smoothing_methods(query_id_to_compare_table2, smoothing_methods)



