# import pandas as pd

# # Load results from CSV files
# laplace_df = pd.read_csv("laplace.csv", names=["qid", "pid", "score"])
# lidstone_df = pd.read_csv("lidstone.csv", names=["qid", "pid", "score"])
# dirichlet_df = pd.read_csv("dirichlet.csv", names=["qid", "pid", "score"])

# # Choose a sample query (replace '123' with an actual query ID from test-queries.tsv)
# sample_qid = 1108939  

# # Extract top 5 passages for this query under each model
# top_laplace = laplace_df[laplace_df["qid"] == sample_qid].head(5)
# top_lidstone = lidstone_df[lidstone_df["qid"] == sample_qid].head(5)
# top_dirichlet = dirichlet_df[dirichlet_df["qid"] == sample_qid].head(5)

# # Display the results side by side
# print(f"Top 5 passages for query ID {sample_qid} using Laplace Smoothing:")
# print(top_laplace.to_string(index=False))
# print("\n")

# print(f"Top 5 passages for query ID {sample_qid} using Lidstone Smoothing:")
# print(top_lidstone.to_string(index=False))
# print("\n")

# print(f"Top 5 passages for query ID {sample_qid} using Dirichlet Smoothing:")
# print(top_dirichlet.to_string(index=False))
# print("\n")

import pandas as pd


# Load queries and passages
queries_df = pd.read_csv("test-queries.tsv", sep="\t", header=None, names=["qid", "query"])
passages_df = pd.read_csv("candidate-passages-top1000.tsv", sep="\t", header=None, names=["qid", "pid", "query_text", "passage"])

# Load ranking results
laplace_df = pd.read_csv("laplace.csv", names=["qid", "pid", "score"])
lidstone_df = pd.read_csv("lidstone.csv", names=["qid", "pid", "score"])
dirichlet_df = pd.read_csv("dirichlet.csv", names=["qid", "pid", "score"])

# Choose a sample query (Replace 123 with an actual qid)
sample_qid = 138632

#138632	definition of tangent

# Extract the query text for the chosen qid
query_text = queries_df[queries_df["qid"] == sample_qid]["query"].values[0]
print(f"\nQuery ID: {sample_qid}")
print(f"Query Text: {query_text}\n")

# Extract top 5 passages for this query under each model
top_laplace = laplace_df[laplace_df["qid"] == sample_qid].head(1)
top_lidstone = lidstone_df[lidstone_df["qid"] == sample_qid].head(1)
top_dirichlet = dirichlet_df[dirichlet_df["qid"] == sample_qid].head(1)

# Function to get passage text for the top-ranked passages
def get_passage_text(df):
    return df.merge(passages_df[["pid", "passage"]], on="pid", how="left")

# Merge passage text
top_laplace = get_passage_text(top_laplace)
top_lidstone = get_passage_text(top_lidstone)
top_dirichlet = get_passage_text(top_dirichlet)

# Display results
print(f"Top 5 passages for query ID {sample_qid} using Laplace Smoothing:")
print(top_laplace.to_string(index=False))
print("\n")

print(f"Top 5 passages for query ID {sample_qid} using Lidstone Smoothing:")
print(top_lidstone.to_string(index=False))
print("\n")

print(f"Top 5 passages for query ID {sample_qid} using Dirichlet Smoothing:")
print(top_dirichlet.to_string(index=False))
print("\n")




# Compute TF-IDF for Queries

# def compute_tfidf_for_query(query, inverted_index, idf, vocabulary, stop_words):
#     """Compute the TF-IDF vector for a single query using the TextProcessor."""
#     # Use TextProcessor from task1.py to preprocess the query
#     processor = TextProcessor(None, stop_words=stop_words)  # Initialize processor with stop words
#     processor.raw_text = query  # Set the raw text as the query
#     processor.to_lowercase()
#     processor.remove_punctuation()
#     processor.tokenize()
#     processor.stem_tokens()  # Apply stemming
#     processor.remove_stopwords()  # Optional stop word removal


#     # Extract tokens and calculate TF
#     query_tokens = [word for word in processor.tokens if word in vocabulary]
#     tf = {}

#     for token in query_tokens: 
#         tf[token] = tf.get(token, 0) + 1

#     # Normalize TF
#     total_tokens = sum(tf.values())
#     tf = {term: freq / total_tokens for term, freq in tf.items()}

#     # Compute TF-IDF using the inverted index's IDF
#     tfidf = {term: tf.get(term, 0) * idf.get(term, 0) for term in idf}
#     return tfidf

# def rank_passages(queries_df, passages_df, tfidf_passages, inverted_index, idf, vocabulary, stop_words):

#     results = []
#     for _, query_row in tqdm(queries_df.iterrows(), desc = "Processing Queries"):
#         qid = query_row['qid']
#         query_text = query_row['query']

#         #Preprocess query
#         processor = TextProcessor(None, stop_words=stop_words)
#         processor.raw_text = query_text
#         processor.to_lowercase()
#         processor.remove_punctuation()
#         processor.tokenize()
#         processor.stem_tokens()
#         processor.remove_stopwords()
#         query_terms = [word for word in processor.tokens if word in vocabulary]

#         # Compute TF-IDF vector for the query
#         query_tfidf = compute_tfidf_for_query(query_text, inverted_index, idf, vocabulary, stop_words)
        
#         # Get candidate passages for this query
#         candidate_passages = passages_df[passages_df['qid'] == qid].head(100)
        
#         similarities = []
#         for _, passage_row in candidate_passages.iterrows():
#             pid = passage_row['pid']
#             passage_tfidf = tfidf_passages.get(pid, {})
#             similarity = cosine_similarity(query_tfidf, passage_tfidf)
#             similarities.append((qid, pid, similarity))
#             # Sort by similarity score and take the top 100
            
#         similarities = sorted(similarities, key=lambda x: x[2], reverse=True)[:100]
#         results.extend(similarities)

#     return results

# def test_preprocessing_consistency(queries_df, passages_df, inverted_index, passage_lengths, idf):
#     """Test whether queries and passages go through the same preprocessing steps."""
    
#     # Pick the first query for testing
#     sample_query = queries_df.iloc[0]
#     qid = sample_query['qid']
#     query_text = sample_query['query']

#     # Preprocess query
#     query_processor = TextProcessor(None, stop_words=stop_words)
#     query_processor.raw_text = query_text
#     query_processor.to_lowercase()
#     query_processor.remove_punctuation()
#     query_processor.tokenize()
#     query_processor.stem_tokens()
#     query_processor.remove_stopwords()
#     query_terms = query_processor.tokens

#     print("\n🔹 Preprocessed Query Terms:")
#     print(query_terms)

#     # Get candidate passages for this query
#     candidate_passages = passages_df[passages_df['qid'] == qid].head(1)  # Take first passage for testing

#     for _, passage_row in candidate_passages.iterrows():
#         pid = passage_row['pid']
#         raw_passage_text = passage_row['passage']

#         # Preprocess passage manually (the same way it's done in the Inverted Index)
#         passage_processor = TextProcessor(None, stop_words=stop_words)
#         passage_processor.raw_text = raw_passage_text
#         passage_processor.to_lowercase()
#         passage_processor.remove_punctuation()
#         passage_processor.tokenize()
#         passage_processor.stem_tokens()
#         passage_processor.remove_stopwords()
#         passage_terms = passage_processor.tokens

#         print("\n🔹 Preprocessed Passage Terms:")
#         print(passage_terms)

#         # Compare query and passage terms
#         common_terms = set(query_terms) & set(passage_terms)
#         print("\n✅ Common Terms (Should Exist If Preprocessing is Correct):")
#         print(common_terms)

#         missing_query_terms = set(query_terms) - set(passage_terms)
#         missing_passage_terms = set(passage_terms) - set(query_terms)

#         print("\n⚠️ Query Terms NOT Found in Passage (Potential Preprocessing Issue!):")
#         print(missing_query_terms)

#         print("\n⚠️ Passage Terms NOT Found in Query (Expected if query is small, but should be checked!):")
#         print(missing_passage_terms)

#     print("\n🔹 If common terms exist and missing terms are minimal, preprocessing is consistent! ✅")




def compute_bm25_idf(inverted_index, total_documents):
    """Compute BM25 IDF values for each term."""
    idf = {}
    
    for term, posting_list in inverted_index.items():
        doc_freq = len(posting_list)  # Number of documents containing term
        idf[term] = np.log((total_documents - doc_freq + 0.5) / (doc_freq + 0.5) + 1)

    return idf

def compute_bm25_score(query_terms, passage, passage_length, idf, avg_passage_length, k1=1.2, b=0.75):
    """Compute the BM25 score between a query and a passage."""
    score = 0

    for term in query_terms:
        if term in idf:  # Ignore terms not found in passages
            tf = passage.get(term, 0)  # Get term frequency in passage
            term_idf = idf[term]

            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (passage_length / avg_passage_length))
            score += term_idf * (numerator / denominator)

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
     
   



