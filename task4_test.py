import pandas as pd

import numpy as np  
import matplotlib.pyplot as plt
import pandas as pd
from task1 import TextProcessor  # Import from task1.py
import nltk
from nltk.corpus import stopwords
from task2 import InvertedIndex
from tqdm import tqdm

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load test queries and passages
queries_file = 'test-queries.tsv'
passages_file = 'candidate-passages-top1000.tsv'

queries_df = pd.read_csv(queries_file, sep='\t', header=None, names=['qid', 'query'])
passages_df = pd.read_csv(passages_file, sep='\t', header=None, names=['qid', 'pid', 'query', 'passage'])

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

# Dictionary of smoothing method names and corresponding result file paths
smoothing_methods = {
    "Laplace": "laplace.csv",
    "Lidstone": "lidstone.csv",
    "Dirichlet": "dirichlet.csv"
}

# Specify the Query ID you want to compare
#138632	definition of tangent
#494835	sensibilities, definition
#190044	foods to detox liver naturally
#1127622 meaning of heat capacity
# 792752	what is ruclip
#168216	does legionella pneumophila cause pneumonia
# 1134787 function of malt
#1063750 why did the us volunterilay enter ww1
#421756	is prorate the same as daily rate
#1126814	noct temperature
query_id_to_compare = 1107988  


processed_query = preprocess_query(query_id_to_compare, queries_df, stop_words)
print(f"Processed query tokens for ID {query_id_to_compare}: {processed_query}")

# Run the comparison
compare_smoothing_methods(query_id_to_compare, smoothing_methods)