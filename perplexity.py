import numpy as np  
import matplotlib.pyplot as plt
import pandas as pd
from task1 import TextProcessor  
import nltk
from nltk.corpus import stopwords
from task2 import InvertedIndex
from tqdm import tqdm
from scipy.special import digamma
from scipy.optimize import minimize
from scipy.stats import gamma

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

# Laplace Smoothing
def laplace_smoothing(query_terms, passage, passage_length, vocabulary_size): 
    score = 0 
    for term in query_terms: 
        term_count = passage.get(term, 0)
        smooth_prob = (term_count + 1) / (passage_length + vocabulary_size)
        score += np.log(smooth_prob)
    return score

# Lidstone Smoothing
def lidstone_smoothing(query_terms, passage, passage_length, vocabulary_size, epsilon=0.1):
    score = 0
    for term in query_terms:
        term_count = passage.get(term, 0)
        smoothed_prob = (term_count + epsilon) / (passage_length + epsilon * vocabulary_size)
        score += np.log(smoothed_prob)
    return score

# Dirichlet Smoothing
def dirichlet_smoothing(query_terms, passage, passage_length, collection_frequency, total_tokens, mu=50):
    score = 0
    for term in query_terms:
        term_count = passage.get(term, 0)
        collection_prob = collection_frequency.get(term, 0) / total_tokens
        smoothed_prob = (term_count + mu * collection_prob) / (passage_length + mu)
        score += np.log(smoothed_prob)
    return score

### NEW FUNCTION: Compute Perplexity ###
def compute_perplexity(query_terms, passage, passage_length, vocabulary_size, smoothing_method, **kwargs):
    """Computes perplexity of a query under a given smoothing model."""
    log_prob_sum = 0
    for term in query_terms:
        term_count = passage.get(term, 0)
        if smoothing_method == dirichlet_smoothing:
            smoothed_prob = smoothing_method([term], passage, passage_length, kwargs['collection_frequency'], kwargs['total_tokens'], kwargs['mu'])
        else:
            smoothed_prob = smoothing_method([term], passage, passage_length, vocabulary_size, **kwargs)
        
        log_prob_sum += smoothed_prob

    avg_log_prob = log_prob_sum / len(query_terms)
    perplexity = np.exp(-avg_log_prob)
    return perplexity

### NEW FUNCTION: Bayesian Estimation of ε ###
def bayesian_estimate_epsilon(term_frequencies, alpha=2.0, beta=5.0):
    """
    Bayesian estimation of ε using a Gamma prior.
    
    Prior: Gamma(alpha, beta)
    Likelihood: Based on observed term frequencies
    
    Returns the estimated ε that maximizes the posterior.
    """
    term_counts = np.array(list(term_frequencies.values()))
    
    def neg_log_posterior(epsilon):
        likelihood = np.sum(np.log(term_counts + epsilon)) - len(term_counts) * np.log(epsilon * len(term_counts) + np.sum(term_counts))
        prior = (alpha - 1) * np.log(epsilon) - beta * epsilon  # Log of Gamma prior
        return -(likelihood + prior)
    
    result = minimize(neg_log_posterior, x0=0.1, bounds=[(1e-5, 1.0)], method='L-BFGS-B')
    return result.x[0] if result.success else 0.1

# Rank Passages Function
def rank_passages(queries_df, inverted_index, passage_lengths, collection_frequency, total_tokens, vocabulary_size, smoothing_method, **kwargs):
    results = []
    perplexities = []

    for _, query_row in tqdm(queries_df.iterrows(), desc="Processing Queries", total=len(queries_df)):
        qid = query_row['qid']
        query_text = query_row['query']

        processor = TextProcessor(None, stop_words=stop_words)
        processor.raw_text = query_text
        processor.to_lowercase()
        processor.remove_punctuation()
        processor.tokenize()
        processor.stem_tokens()
        processor.remove_stopwords()
        query_terms = [term for term in processor.tokens if term in vocabulary]

        candidate_passages = passages_df[passages_df['qid'] == qid]

        similarities = []
        for _, passage_row in candidate_passages.iterrows():
            pid = passage_row['pid']
            passage = {term: inverted_index[term].get(pid, 0) for term in query_terms}
            passage_length = passage_lengths.get(pid, 0)

            if smoothing_method == dirichlet_smoothing:
                score = smoothing_method(query_terms, passage, passage_length, collection_frequency, total_tokens, **kwargs)
            else:
                score = smoothing_method(query_terms, passage, passage_length, vocabulary_size, **kwargs)

            similarities.append((qid, pid, score))

        similarities = sorted(similarities, key=lambda x: x[2], reverse=True)[:100]
        results.extend(similarities)

        # Compute Perplexity for the first ranked passage
        if similarities:
            top_pid = similarities[0][1]
            top_passage = {term: inverted_index[term].get(top_pid, 0) for term in query_terms}
            top_length = passage_lengths.get(top_pid, 0)

            perplexity = compute_perplexity(query_terms, top_passage, top_length, vocabulary_size, smoothing_method, **kwargs)
            perplexities.append((qid, top_pid, perplexity))

    return results, perplexities

# Save Results to CSV
def save_results_to_csv(results, output_file):
    """Save ranking results to CSV."""
    results_df = pd.DataFrame(results, columns=['qid', 'pid', 'log-score'])
    results_df.to_csv(output_file, index=False, header=False)

if __name__ == "__main__":
    passages_file = 'candidate-passages-top1000.tsv'
    passages_df = pd.read_csv(passages_file, sep='\t', header=None, names=['qid', 'pid', 'query', 'passage'])
    queries_file = 'test-queries.tsv'
    queries_df = pd.read_csv(queries_file, sep='\t', header=None, names=['qid', 'query'])

    processor = TextProcessor('passage-collection.txt', stop_words=stop_words)
    processor.process()
    vocabulary = processor.get_vocabulary()

    index = InvertedIndex(vocabulary=vocabulary, stop_words=stop_words)
    index.build_index(passages_df[['pid', 'passage']])

    collection_frequency, total_tokens = compute_collection_statistics(index.inverted_index, index.passage_lengths)
    vocabulary_size = len(vocabulary)

    # Bayesian Estimation of ε
    estimated_epsilon = bayesian_estimate_epsilon(collection_frequency)
    print(f"Estimated ε: {estimated_epsilon:.5f}")

    # Run Lidstone Smoothing with Bayesian ε
    lidstone_results, lidstone_perplexities = rank_passages(
        queries_df=queries_df,
        inverted_index=index.inverted_index,
        passage_lengths=index.passage_lengths,
        collection_frequency=None,
        total_tokens=None,
        vocabulary_size=vocabulary_size,
        smoothing_method=lidstone_smoothing,
        epsilon=estimated_epsilon
    )
    save_results_to_csv(lidstone_results, 'lidstone.csv')

    # Save Perplexity Results
    perplexity_df = pd.DataFrame(lidstone_perplexities, columns=['qid', 'pid', 'perplexity'])
    perplexity_df.to_csv('perplexity_lidstone.csv', index=False)

    print("Results saved: lidstone.csv, perplexity_lidstone.csv")



# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import nltk
# import re
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
# from scipy.stats import linregress
# nltk.download('stopwords')

# class TextProcessor: 

#     def __init__(self, file_path, stop_words=None):
#         self.file_path = file_path
#         self.raw_text = ""
#         self. cleaned_text = "" 
#         self.tokens = []
#         self.vocabulary = set()
#         self.term_frequencies = {}
#         self.stop_words = stop_words if stop_words else set(stopwords.words('english'))
#         self.stemmer = PorterStemmer()  # Initialize the stemmer

#     def read_file(self):
#         """Reads the file and stores the raw text."""
#         with open(self.file_path, 'r') as file:
#             self.raw_text = file.read()
#         return self.raw_text
    
#     def to_lowercase(self): 
#         """Converts raw text to lowercase."""
#         result = ""
#         for char in self.raw_text:
#             if 'A' <= char <= 'Z':
#                 result += char.lower()
#             else: 
#                 result += char
        
#         self.cleaned_text = result
        
    
#     def remove_punctuation(self): 
#         """Removes punctuation from the text."""

#         """Removes all non-alphanumeric characters from the text."""
#         self.cleaned_text = re.sub(r'[^a-zA-Z0-9]', ' ', self.cleaned_text)
#         # punctuation = """!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""
#         # result = ""
#         # for char in self.cleaned_text: 
#         #     if char not in punctuation:
#         #         result += char

#         # self.cleaned_text = result

#     def tokenize(self):
#         """Tokenizes the text."""
#         tokens = []
#         word = ""

#         for char in self.cleaned_text:
#             if char == " " or char == "\n": # A space indicates the end of a word (i.e. where to tokenize). Also addresses potential new lines.
#                 if word:    
#                     # Split alphanumeric characters (e.g., al39 -> ['al', '39'])
#                     alnum_split = re.findall(r'[a-zA-Z]+|\d+', word)
#                     tokens.extend(alnum_split)
#                     word = ""
#             else:
#                 word += char

#         if word: #In the case that there is a last word, add this word to tokens
#             alnum_split = re.findall(r'[a-zA-Z]+|\d+', word)
#             tokens.extend(alnum_split)
        
#         self.tokens = tokens 

#     def stem_tokens(self):
#         """Applies stemming to the tokens."""
#         self.tokens = [self.stemmer.stem(word) for word in self.tokens]

#     def remove_stopwords(self):
#         """Removes stopwords from the tokens."""
#         self.tokens = [word for word in self.tokens if word not in self.stop_words]

#     def unique_vocabulary(self):
#         """Returns the unique vocabulary of the text."""
#         self.vocabulary = set(self.tokens)
#         #return self.vocabulary

#     def calculate_term_frequencies(self): 

#         for token in self.tokens: 
#             if token in self.term_frequencies:
#                 self.term_frequencies[token] += 1
#             else: 
#                 self.term_frequencies[token] = 1
 
#     def process(self): 

#         self.read_file()
#         self.to_lowercase()
#         self.remove_punctuation()
#         self.tokenize()
#         self.stem_tokens() # Apply stemming after tokenization
#         self.remove_stopwords() # Remove stopwords after tokenization (optional)
#         self.unique_vocabulary() # Generate necessary vocabulary
#         self.calculate_term_frequencies() 

#     def get_tokens(self):
#         return self.tokens
    
#     def get_vocabulary(self):
#         return self.vocabulary
    
#     def get_term_frequencies(self):
#         return self.term_frequencies    


# if __name__ == "__main__":

#     file_path = 'passage-collection.txt'
#     processor = TextProcessor(file_path)
#     processor.process()

#     # Output: 



#     tokens = processor.get_tokens()
#     vocabulary = processor.get_vocabulary()

#     print("length of Vocabulary: ", len(vocabulary))

#     term_frequencies = processor.get_term_frequencies()

#     top_ten_terms = sorted(term_frequencies.items(), key=lambda x: x[1], reverse=True)[:10]

#     for term, freq in top_ten_terms:
#         print(f"{term}: {freq}")

#     def plot_zipf_law(term_frequencies):

#         # sort the terms by frequency
#         sorted_terms = sorted(term_frequencies.items(), key=lambda x: x[1], reverse=True)

#         ranks = np.arange(1, len(sorted_terms) + 1)
#         print(ranks)
#         frequencies = np.array([freq for _, freq in sorted_terms], dtype = np.float64)  
#         total_terms = np.sum(frequencies)
#         probabilities = frequencies / total_terms

#         harmonic_sum = np.sum(1 / ranks)
#         C_expected = 1 / harmonic_sum
#         zipf_distribution = C_expected / ranks


#         # Log-log transformation for fitting
#         log_ranks = np.log(ranks)
#         log_probs = np.log(probabilities)

#         # Perform linear regression
#         slope, intercept, r_value, p_value, std_err_slope = linregress(log_ranks, log_probs)
#         fitted_line = intercept + slope * log_ranks # Best-fit line

#         # Compute standard error of the intercept
#         n = len(log_ranks)
#         std_err_intercept = std_err_slope * np.sqrt(np.sum(log_ranks**2) / n)

#         # Compute fitted C from the intercept
#         C_fitted = np.exp(intercept)
#         C_uncertainty = np.exp(intercept) * std_err_intercept  # Uncertainty propagation

#         # Print Expected C
#         print(f"Expected C (from harmonic sum): {C_expected:.4f}")

#         # Print estimated values & uncertainties
#         print(f"Fitted C (from log-log regression): {C_fitted:.4f} ± {C_uncertainty:.4f}")
#         print(f"Estimated exponent (s) from log-log fit: {-slope:.4f} ± {std_err_slope:.4f}")

#         # Create log-log plot
#         plt.figure(figsize=(10, 6))
#         plt.plot(ranks, probabilities, marker=".", linestyle="none", label = "Empirical Data")
#         plt.plot(ranks, zipf_distribution, linestyle="--", color="red", label="Theoretical Zipf's Law")
#         plt.title("Zipf's Law: Probability vs. Rank")
#         plt.xlabel("Rank")
#         plt.ylabel("Normalised Frequency")
#         plt.grid(False)
#         plt.legend()
#         plt.savefig("zipf_plot_nonlog.pdf", format="pdf", bbox_inches="tight")
#         plt.show()

#         # Create log-log plot
#         plt.figure(figsize=(10, 6))
#         plt.plot(log_ranks, log_probs, marker=".", linestyle="none", label = "Empirical Data")
#         plt.plot(log_ranks, fitted_line, linestyle="--", color="red", label="Best-fit Line")
#         plt.plot(log_ranks, np.log(zipf_distribution), linestyle="--", color="green", label="Theoretical Zipf's Law")
#         # plt.loglog(ranks, probabilities, marker=".", linestyle="none")
#         # plt.loglog(ranks, zipf_distribution, linestyle="--", color="red", label="Theoretical Zipf's Law")
#         plt.title("Zipf's Law: Probability vs. Rank")
#         plt.xlabel("Rank (log scale)")
#         plt.ylabel("Normalised Frequency (log scale)")
#         plt.grid(False)
#         plt.legend()
#         # Save as a PDF (vector format)
#         #plt.savefig("zipf_plot_SW_NOTincluded.pdf", format="pdf", bbox_inches="tight")
#         plt.show()

#     plot_zipf_law(term_frequencies)

# #BOTH PLOT VERSION: 

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import nltk
# import re
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
# from scipy.stats import linregress

# nltk.download('stopwords')


# class TextProcessor:
#     def __init__(self, file_path, remove_stopwords=True):
#         self.file_path = file_path
#         self.raw_text = ""
#         self.cleaned_text = ""
#         self.tokens = []
#         self.vocabulary = set()
#         self.term_frequencies = {}
#         self.remove_stopwords = remove_stopwords
#         self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
#         self.stemmer = PorterStemmer()

#     def read_file(self):
#         """Reads the file and stores the raw text."""
#         with open(self.file_path, 'r') as file:
#             self.raw_text = file.read()

#     def preprocess(self):
#         """Preprocesses the text: lowercasing, punctuation removal, tokenization, and optional stopword removal."""
#         self.cleaned_text = re.sub(r'[^a-zA-Z0-9]', ' ', self.raw_text.lower())
#         self.tokens = re.findall(r'\b\w+\b', self.cleaned_text)
#         if self.remove_stopwords:
#             self.tokens = [word for word in self.tokens if word not in self.stop_words]
#         self.vocabulary = set(self.tokens)
#         self.calculate_term_frequencies()

#     def calculate_term_frequencies(self):
#         """Computes term frequencies."""
#         self.term_frequencies = {}
#         for token in self.tokens:
#             self.term_frequencies[token] = self.term_frequencies.get(token, 0) + 1

#     def get_term_frequencies(self):
#         return self.term_frequencies


# def plot_zipf_law(term_frequencies_with_stopwords, term_frequencies_without_stopwords):
#     """Plots Zipf's Law comparison with theoretical Zipf distribution and linear regression."""

#     def compute_zipf_data(term_frequencies):
#         """Sorts terms by frequency and computes Zipf distribution."""
#         sorted_terms = sorted(term_frequencies.items(), key=lambda x: x[1], reverse=True)
#         ranks = np.arange(1, len(sorted_terms) + 1)
#         frequencies = np.array([freq for _, freq in sorted_terms], dtype=np.float64)
#         total_terms = np.sum(frequencies)
#         probabilities = frequencies / total_terms
#         return ranks, probabilities

#     def perform_linear_regression(ranks, probabilities):
#         """Performs log-log linear regression and returns the fitted line."""
#         log_ranks = np.log(ranks)
#         log_probs = np.log(probabilities)
#         slope, intercept, r_value, p_value, std_err_slope = linregress(log_ranks, log_probs)
#         fitted_line = intercept + slope * log_ranks
#         return log_ranks, log_probs, fitted_line, slope, intercept, std_err_slope

#     def compute_theoretical_zipf(ranks):
#         """Computes the expected Zipf distribution."""
#         harmonic_sum = np.sum(1 / ranks)
#         C_expected = 1 / harmonic_sum
#         return C_expected / ranks

#     # Compute Zipf data for both cases
#     ranks_with, probs_with = compute_zipf_data(term_frequencies_with_stopwords)
#     ranks_without, probs_without = compute_zipf_data(term_frequencies_without_stopwords)

#     # Perform regression
#     log_ranks_with, log_probs_with, fitted_with, slope_with, intercept_with, std_err_slope_with = perform_linear_regression(ranks_with, probs_with)
#     log_ranks_without, log_probs_without, fitted_without, slope_without, intercept_without, std_err_slope_without = perform_linear_regression(ranks_without, probs_without)

#     # Compute theoretical Zipf distribution
#     theoretical_zipf_with = compute_theoretical_zipf(ranks_with)
#     theoretical_zipf_without = compute_theoretical_zipf(ranks_without)

#     # Print estimated parameters
#     print(f"\nZipf's Law Analysis (With Stopwords):")
#     print(f"  Slope (Exponent): {-slope_with:.4f} ± {std_err_slope_with:.4f}")
#     print(f"  Intercept: {intercept_with:.4f}")
    
#     print(f"\nZipf's Law Analysis (Without Stopwords):")
#     print(f"  Slope (Exponent): {-slope_without:.4f} ± {std_err_slope_without:.4f}")
#     print(f"  Intercept: {intercept_without:.4f}")

#     # Plot both empirical and theoretical Zipf's Law
#     plt.figure(figsize=(10, 6))
    
#     # Scatter plots for empirical data
#     plt.scatter(log_ranks_with, log_probs_with, marker=".", alpha=0.5, label="Empirical (With Stopwords)", color="blue", s = 10)
#     plt.scatter(log_ranks_without, log_probs_without, marker=".", alpha=0.5, label="Empirical (Without Stopwords)", color="red", s = 10)
    
#     # Linear regression fits
#     plt.plot(log_ranks_with, fitted_with, linestyle="--", color="blue", label="Best-fit Line (With Stopwords)", alpha=0.4)
#     plt.plot(log_ranks_without, fitted_without, linestyle="--", color="red", label="Best-fit Line (Without Stopwords)", alpha=0.4)
    
#     # Theoretical Zipf's Law curves
#     plt.plot(np.log(ranks_with), np.log(theoretical_zipf_with), linestyle="--", color="green", label="Theoretical Zipf (With Stopwords)")
#     #plt.plot(np.log(ranks_without), np.log(theoretical_zipf_without), linestyle="--", color="purple", label="Theoretical Zipf (Without Stopwords)")

#     plt.xlabel("Log Rank")
#     plt.ylabel("Log Probability")
#     plt.title("Zipf's Law: Log-Log Plot with Theoretical and Empirical Distributions")
#     plt.legend()
#     plt.grid(False)
#     plt.savefig("zipf_plot_BOTH.pdf", format="pdf", bbox_inches="tight")
#     plt.show()


# if __name__ == "__main__":
#     file_path = 'passage-collection.txt'

#     # Process with stopwords
#     processor_with_stopwords = TextProcessor(file_path, remove_stopwords=False)
#     processor_with_stopwords.read_file()
#     processor_with_stopwords.preprocess()
#     term_frequencies_with_stopwords = processor_with_stopwords.get_term_frequencies()

#     # Process without stopwords
#     processor_without_stopwords = TextProcessor(file_path, remove_stopwords=True)
#     processor_without_stopwords.read_file()
#     processor_without_stopwords.preprocess()
#     term_frequencies_without_stopwords = processor_without_stopwords.get_term_frequencies()

#     # Plot the comparison
#     plot_zipf_law(term_frequencies_with_stopwords, term_frequencies_without_stopwords)


# import numpy as np
# import matplotlib.pyplot as plt
# import re
# import time  # Added for time tracking
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
# from scipy.stats import linregress
# import nltk

# nltk.download('stopwords')

# class TextProcessor: 
#     def __init__(self, file_path, stop_words=None):
#         self.file_path = file_path
#         self.raw_text = ""
#         self.cleaned_text = "" 
#         self.tokens = []
#         self.vocabulary = set()
#         self.term_frequencies = {}
#         self.stop_words = stop_words if stop_words else set(stopwords.words('english'))
#         self.stemmer = PorterStemmer()  # Initialize the stemmer

#     def read_file(self):
#         """Reads the file and stores the raw text."""
#         with open(self.file_path, 'r') as file:
#             self.raw_text = file.read()
#         return self.raw_text
    
#     def to_lowercase(self): 
#         """Converts raw text to lowercase."""
#         result = ""
#         for char in self.raw_text:
#             if 'A' <= char <= 'Z':
#                 result += char.lower()
#             else: 
#                 result += char
#         self.cleaned_text = result
        
#     def remove_punctuation(self): 
#         """Removes punctuation from the text."""
#         self.cleaned_text = re.sub(r'[^a-zA-Z0-9]', ' ', self.cleaned_text)

#     def tokenize(self):
#         """Tokenizes the text."""
#         tokens = []
#         word = ""

#         for char in self.cleaned_text:
#             if char == " " or char == "\n": 
#                 if word:    
#                     alnum_split = re.findall(r'[a-zA-Z]+|\d+', word)
#                     tokens.extend(alnum_split)
#                     word = ""
#             else:
#                 word += char

#         if word: 
#             alnum_split = re.findall(r'[a-zA-Z]+|\d+', word)
#             tokens.extend(alnum_split)
        
#         self.tokens = tokens 

#     def stem_tokens(self):
#         """Applies stemming to the tokens."""
#         self.tokens = [self.stemmer.stem(word) for word in self.tokens]

#     def remove_stopwords(self):
#         """Removes stopwords from the tokens."""
#         self.tokens = [word for word in self.tokens if word not in self.stop_words]

#     def unique_vocabulary(self):
#         """Returns the unique vocabulary of the text."""
#         self.vocabulary = set(self.tokens)

#     def calculate_term_frequencies(self): 
#         for token in self.tokens: 
#             if token in self.term_frequencies:
#                 self.term_frequencies[token] += 1
#             else: 
#                 self.term_frequencies[token] = 1
 
#     def process(self): 
#         start_time = time.time()
        
#         self.read_file()
#         self.to_lowercase()
#         self.remove_punctuation()
#         self.tokenize()
#         self.stem_tokens() 
#         self.remove_stopwords()
#         self.unique_vocabulary() 
#         self.calculate_term_frequencies() 
        
#         end_time = time.time()
#         print(f"Text processing completed in {end_time - start_time:.2f} seconds")

#     def get_tokens(self):
#         return self.tokens
    
#     def get_vocabulary(self):
#         return self.vocabulary
    
#     def get_term_frequencies(self):
#         return self.term_frequencies    


# def plot_zipf_law(term_frequencies_with_stopwords, term_frequencies_without_stopwords):
#     """Plots Zipf's Law comparison with theoretical Zipf distribution and linear regression."""
#     start_time = time.time()

#     def compute_zipf_data(term_frequencies):
#         """Sorts terms by frequency and computes Zipf distribution."""
#         sorted_terms = sorted(term_frequencies.items(), key=lambda x: x[1], reverse=True)
#         ranks = np.arange(1, len(sorted_terms) + 1)
#         frequencies = np.array([freq for _, freq in sorted_terms], dtype=np.float64)
#         total_terms = np.sum(frequencies)
#         probabilities = frequencies / total_terms
#         return ranks, probabilities

#     def perform_linear_regression(ranks, probabilities):
#         """Performs log-log linear regression and returns the fitted line."""
#         log_ranks = np.log(ranks)
#         log_probs = np.log(probabilities)
#         slope, intercept, r_value, p_value, std_err_slope = linregress(log_ranks, log_probs)
#         fitted_line = intercept + slope * log_ranks
#         return log_ranks, log_probs, fitted_line, slope, intercept, std_err_slope

#     def compute_theoretical_zipf(ranks):
#         """Computes the expected Zipf distribution."""
#         harmonic_sum = np.sum(1 / ranks)
#         C_expected = 1 / harmonic_sum
#         return C_expected / ranks

#     # Compute Zipf data for both cases
#     ranks_with, probs_with = compute_zipf_data(term_frequencies_with_stopwords)
#     ranks_without, probs_without = compute_zipf_data(term_frequencies_without_stopwords)

#     # Perform regression
#     log_ranks_with, log_probs_with, fitted_with, slope_with, intercept_with, std_err_slope_with = perform_linear_regression(ranks_with, probs_with)
#     log_ranks_without, log_probs_without, fitted_without, slope_without, intercept_without, std_err_slope_without = perform_linear_regression(ranks_without, probs_without)

#     # Compute theoretical Zipf distribution
#     theoretical_zipf_with = compute_theoretical_zipf(ranks_with)
#     theoretical_zipf_without = compute_theoretical_zipf(ranks_without)

#     # Print estimated parameters
#     print(f"\nZipf's Law Analysis (With Stopwords):")
#     print(f"  Slope (Exponent): {-slope_with:.4f} ± {std_err_slope_with:.4f}")
#     print(f"  Intercept: {intercept_with:.4f}")
    
#     print(f"\nZipf's Law Analysis (Without Stopwords):")
#     print(f"  Slope (Exponent): {-slope_without:.4f} ± {std_err_slope_without:.4f}")
#     print(f"  Intercept: {intercept_without:.4f}")

#     # Plot Zipf's Law
#     plt.figure(figsize=(10, 6))
    
#     # Scatter plots for empirical data
#     plt.scatter(log_ranks_with, log_probs_with, marker=".", alpha=0.5, label="Empirical (With Stopwords)", color="blue", s=10)
#     plt.scatter(log_ranks_without, log_probs_without, marker=".", alpha=0.5, label="Empirical (Without Stopwords)", color="red", s=10)
    
#     # Linear regression fits
#     plt.plot(log_ranks_with, fitted_with, linestyle="--", color="blue", label="Best-fit Line (With Stopwords)", alpha=0.4)
#     plt.plot(log_ranks_without, fitted_without, linestyle="--", color="red", label="Best-fit Line (Without Stopwords)", alpha=0.4)
    
#     # Theoretical Zipf's Law curves
#     plt.plot(np.log(ranks_with), np.log(theoretical_zipf_with), linestyle="--", color="green", label="Theoretical Zipf (With Stopwords)")

#     plt.xlabel("Log Rank")
#     plt.ylabel("Log Probability")
#     plt.title("Zipf's Law: Log-Log Plot with Theoretical and Empirical Distributions")
#     plt.legend()
#     plt.grid(False)
#     plt.savefig("zipf_plot_BOTH.pdf", format="pdf", bbox_inches="tight")
#     plt.show()

#     end_time = time.time()
#     print(f"Zipf's Law plotting completed in {end_time - start_time:.2f} seconds")


# if __name__ == "__main__":
#     start_time = time.time()

#     file_path = 'passage-collection.txt'

#     processor_with_stopwords = TextProcessor(file_path)
#     processor_with_stopwords.process()
#     term_frequencies_with_stopwords = processor_with_stopwords.get_term_frequencies()

#     processor_without_stopwords = TextProcessor(file_path, stop_words=set())
#     processor_without_stopwords.process()
#     term_frequencies_without_stopwords = processor_without_stopwords.get_term_frequencies()

#     plot_zipf_law(term_frequencies_with_stopwords, term_frequencies_without_stopwords)

#     end_time = time.time()
#     print(f"Total execution time: {end_time - start_time:.2f} seconds")