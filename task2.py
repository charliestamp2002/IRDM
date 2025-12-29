import numpy as np  
import matplotlib.pyplot as plt
import pandas as pd
from task1 import TextProcessor 
import nltk
import time
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


class InvertedIndex:
    def __init__(self, vocabulary, stop_words=None):
        self.vocabulary = vocabulary  # Vocabulary from Task 1
        self.inverted_index = {}  # The inverted index structure
        self.passage_lengths = {}  # Map each passage ID (pid) to its length
        self.stop_words = set(stop_words) if stop_words else set()  # Optional stop words set

    def preprocess(self, text):
        """Preprocesses the text using the Task 1 TextProcessor."""
        processor = TextProcessor(None)  # Initialize TextProcessor without a file
        processor.raw_text = text  # Directly assign the raw text
        processor.to_lowercase()
        processor.remove_punctuation()
        processor.tokenize()
        processor.stem_tokens()
        processor.remove_stopwords()  # Optional based on decision in Task 1
        
        return [word for word in processor.tokens if word in self.vocabulary]

    def build_index(self, df):
        """Builds the inverted index from a DataFrame with 'pid' and 'passage' columns."""

        start_time = time.time()

        for _, row in df.iterrows():
            pid = row['pid']
            passage = row['passage']
            tokens = self.preprocess(passage)  # Preprocess and tokenize the passage

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

    def search(self, term):
        """Searches for a term in the inverted index and returns the posting list with term frequencies."""
        return self.inverted_index.get(term, {})

    def get_passage_length(self, pid):
        """Returns the length of a passage given its ID."""
        return self.passage_lengths.get(pid, 0)

    def display_index(self, top_n=10):
        """Displays the first N entries in the inverted index for preview."""
        count = 0
        for term, posting_list in self.inverted_index.items():
            print(f"{term}: {posting_list}")
            count += 1
            if count >= top_n:
                break

    def display_passage_lengths(self, top_n=10):
        """Displays the lengths of the first N passages for preview."""
        count = 0
        for pid, length in self.passage_lengths.items():
            print(f"Passage ID {pid}: Length = {length}")
            count += 1
            if count >= top_n:
                break


if __name__ == "__main__":

    start_time = time.time()

    file_path = 'passage-collection.txt'
    processor = TextProcessor(file_path, stop_words=stop_words)
    processor.process()

    # Extract vocabulary and stop words
    vocabulary = processor.get_vocabulary()

    file_path = 'candidate-passages-top1000.tsv'
    candidate_passages_df = pd.read_csv(file_path, sep='\t', header=None, names=['qid', 'pid', 'query', 'passage'])

    # Build the inverted index
    index = InvertedIndex(vocabulary=vocabulary, stop_words=stop_words)
    index.build_index(candidate_passages_df)

    # Display a preview of the inverted index
    index.display_index(top_n=5)

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")



