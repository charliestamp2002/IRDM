
import numpy as np
import matplotlib.pyplot as plt
import re
import time 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.stats import linregress
import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

class TextProcessor: 
    def __init__(self, file_path, stop_words=None):
        self.file_path = file_path
        self.raw_text = ""
        self.cleaned_text = "" 
        self.tokens = []
        self.vocabulary = set()
        self.term_frequencies = {}
        self.stop_words = stop_words if stop_words else set(stopwords.words('english'))
        self.stemmer = PorterStemmer()  # Initialize the stemmer

    def read_file(self):
        """Reads the file and stores the raw text."""
        with open(self.file_path, 'r') as file:
            self.raw_text = file.read()
        return self.raw_text
    
    def to_lowercase(self): 
        """Converts raw text to lowercase."""
        result = ""
        for char in self.raw_text:
            if 'A' <= char <= 'Z':
                result += char.lower()
            else: 
                result += char
        self.cleaned_text = result
        
    def remove_punctuation(self): 
        """Removes punctuation from the text."""
        self.cleaned_text = re.sub(r'[^a-zA-Z0-9]', ' ', self.cleaned_text)

    def tokenize(self):
        """Tokenizes the text."""
        tokens = []
        word = ""

        for char in self.cleaned_text:
            if char == " " or char == "\n": 
                if word:    
                    alnum_split = re.findall(r'[a-zA-Z]+|\d+', word)
                    tokens.extend(alnum_split)
                    word = ""
            else:
                word += char

        if word: 
            alnum_split = re.findall(r'[a-zA-Z]+|\d+', word)
            tokens.extend(alnum_split)
        
        self.tokens = tokens 

    def stem_tokens(self):
        """Applies stemming to the tokens."""
        self.tokens = [self.stemmer.stem(word) for word in self.tokens]

    def remove_stopwords(self):
        """Removes stopwords from the tokens."""
        self.tokens = [word for word in self.tokens if word not in self.stop_words]

    def unique_vocabulary(self):
        """Returns the unique vocabulary of the text."""
        self.vocabulary = set(self.tokens)

    def calculate_term_frequencies(self): 
        for token in self.tokens: 
            if token in self.term_frequencies:
                self.term_frequencies[token] += 1
            else: 
                self.term_frequencies[token] = 1
 
    def process(self): 
        start_time = time.time()
        
        self.read_file()
        self.to_lowercase()
        self.remove_punctuation()
        self.tokenize()
        self.stem_tokens() 
        self.remove_stopwords()
        self.unique_vocabulary() 
        self.calculate_term_frequencies() 
        
        end_time = time.time()
        print(f"Text processing completed in {end_time - start_time:.2f} seconds")

    def get_tokens(self):
        return self.tokens
    
    def get_vocabulary(self):
        return self.vocabulary
    
    def get_term_frequencies(self):
        return self.term_frequencies    


def plot_zipf_law(term_frequencies_with_stopwords, term_frequencies_without_stopwords):
    """Plots Zipf's Law comparison with theoretical Zipf distribution and linear regression."""
    start_time = time.time()

    def compute_zipf_data(term_frequencies):
        """Sorts terms by frequency and computes Zipf distribution."""
        sorted_terms = sorted(term_frequencies.items(), key=lambda x: x[1], reverse=True)
        ranks = np.arange(1, len(sorted_terms) + 1)
        frequencies = np.array([freq for _, freq in sorted_terms], dtype=np.float64)
        total_terms = np.sum(frequencies)
        probabilities = frequencies / total_terms
        return ranks, probabilities

    def perform_linear_regression(ranks, probabilities):
        """Performs log-log linear regression and returns the fitted line."""
        log_ranks = np.log(ranks)
        log_probs = np.log(probabilities)
        slope, intercept, r_value, p_value, std_err_slope = linregress(log_ranks, log_probs)
        fitted_line = intercept + slope * log_ranks
        return log_ranks, log_probs, fitted_line, slope, intercept, std_err_slope

    def compute_theoretical_zipf(ranks):
        """Computes the expected Zipf distribution."""
        harmonic_sum = np.sum(1 / ranks)
        C_expected = 1 / harmonic_sum
        return C_expected / ranks

    # Compute Zipf data for both cases
    ranks_with, probs_with = compute_zipf_data(term_frequencies_with_stopwords)
    ranks_without, probs_without = compute_zipf_data(term_frequencies_without_stopwords)

    # Perform regression
    log_ranks_with, log_probs_with, fitted_with, slope_with, intercept_with, std_err_slope_with = perform_linear_regression(ranks_with, probs_with)
    log_ranks_without, log_probs_without, fitted_without, slope_without, intercept_without, std_err_slope_without = perform_linear_regression(ranks_without, probs_without)

    # Compute theoretical Zipf distribution
    theoretical_zipf_with = compute_theoretical_zipf(ranks_with)
    theoretical_zipf_without = compute_theoretical_zipf(ranks_without)

    # Print estimated parameters
    print(f"\nZipf's Law Analysis (With Stopwords):")
    print(f"  Slope (Exponent): {-slope_with:.4f} ± {std_err_slope_with:.4f}")
    print(f"  Intercept: {intercept_with:.4f}")
    
    print(f"\nZipf's Law Analysis (Without Stopwords):")
    print(f"  Slope (Exponent): {-slope_without:.4f} ± {std_err_slope_without:.4f}")
    print(f"  Intercept: {intercept_without:.4f}")

    # Plot Zipf's Law
    plt.figure(figsize=(10, 6))
    
    # Scatter plots for empirical data
    plt.scatter(log_ranks_with, log_probs_with, marker=".", alpha=0.5, label="Empirical (Stopwords Removed)", color="blue", s=10)
    plt.scatter(log_ranks_without, log_probs_without, marker=".", alpha=0.5, label="Empirical (Stopwords Included)", color="red", s=10)
    
    # Linear regression fits
    plt.plot(log_ranks_with, fitted_with, linestyle="--", color="blue", label="Best-fit Line (Stopwords Removed)", alpha=0.4)
    plt.plot(log_ranks_without, fitted_without, linestyle="--", color="red", label="Best-fit Line (Stopwords Included)", alpha=0.4)
    
    # Theoretical Zipf's Law curves
    plt.plot(np.log(ranks_with), np.log(theoretical_zipf_with), linestyle="--", color="green", label="Theoretical Zipf's Law")

    plt.xlabel("Log Rank", fontsize=16)
    plt.ylabel("Log Normalised Frequency", fontsize=16)
    plt.title("Zipf's Law: Log-Log Plot with Theoretical and Empirical Distributions", fontsize = 16)
    #plt.xscale('log')  # Ensure log scale
    #plt.yscale('log')
    # Increase tick label size
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.minorticks_on() 
    plt.tick_params(axis='both', which='major', labelsize=14, length=6)  # Major tick size
    plt.tick_params(axis='both', which='minor', labelsize=12, length=4)  # Minor tick size

    plt.legend(fontsize = 14)       
    plt.grid(False)
    plt.savefig("fig_2.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    end_time = time.time()
    print(f"Zipf's Law plotting completed in {end_time - start_time:.2f} seconds")

def plot_normalized_frequency_with_zipf(term_frequencies, title, color):
    """Plots normalized frequency of terms against rank and adds the theoretical Zipf's Law curve."""
    
    # Sort terms by frequency and compute normalized frequencies
    sorted_terms = sorted(term_frequencies.items(), key=lambda x: x[1], reverse=True)
    ranks = np.arange(1, len(sorted_terms) + 1)
    frequencies = np.array([freq for _, freq in sorted_terms], dtype=np.float64)
    total_terms = np.sum(frequencies)
    normalized_frequencies = frequencies / total_terms  # Normalize

    # Compute theoretical Zipf's Law distribution
    harmonic_sum = np.sum(1 / ranks)  # Harmonic series sum for normalization
    zipf_theoretical = (1 / ranks) / harmonic_sum  # Theoretical Zipf curve

    # Plot empirical data
    plt.figure(figsize=(10, 6))
    plt.scatter(ranks, normalized_frequencies, color=color, alpha=0.6, s=10, label=f"Empirical {title}")
    plt.plot(ranks, normalized_frequencies, color=color, alpha=0.4, linestyle="--")

    # Plot theoretical Zipf's Law curve
    plt.plot(ranks, zipf_theoretical, linestyle="dotted", color="green", label="Theoretical Zipf's Law", linewidth=2)

    plt.xlabel("Rank", fontsize=14)
    plt.ylabel("Normalised Frequency", fontsize=14)
    plt.title(f"{title}", fontsize=16)

    plt.minorticks_on()
    plt.tick_params(axis='both', which='major', labelsize=14, length=6)  # Major tick size
    plt.tick_params(axis='both', which='minor', labelsize=12, length=4)  # Minor tick size

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(False)
    plt.savefig(f"{title.replace(' ', '_')}_zipf.pdf", format="pdf", bbox_inches="tight")
    plt.show()

def calculate_least_squares(term_frequencies, dataset_name):
    """Computes the least squares error between empirical distribution and theoretical Zipf's Law."""
    
    # Sort terms by frequency
    sorted_terms = sorted(term_frequencies.items(), key=lambda x: x[1], reverse=True)
    ranks = np.arange(1, len(sorted_terms) + 1)
    frequencies = np.array([freq for _, freq in sorted_terms], dtype=np.float64)
    total_terms = np.sum(frequencies)
    normalized_frequencies = frequencies / total_terms  # Normalise empirical data

    # Compute theoretical Zipf's Law distribution
    harmonic_sum = np.sum(1 / ranks)  # Harmonic series sum for normalisation
    zipf_theoretical = (1 / ranks) / harmonic_sum  # Theoretical Zipf distribution

    # Compute Sum of Squared Differences (Least Squares)
    least_squares_error = np.sum((normalized_frequencies - zipf_theoretical) ** 2)

    print(f"\nLeast Squares Error for {dataset_name}: {least_squares_error:.6e}")

    return least_squares_error

if __name__ == "__main__":
    start_time = time.time()

    file_path = 'passage-collection.txt'

    processor_with_stopwords = TextProcessor(file_path)
    processor_with_stopwords.process()
    term_frequencies_with_stopwords = processor_with_stopwords.get_term_frequencies()

    # Processing manually to exclude stopword removal
    processor_without_stopwords = TextProcessor(file_path)
    processor_without_stopwords.read_file()
    processor_without_stopwords.to_lowercase()
    processor_without_stopwords.remove_punctuation()
    processor_without_stopwords.tokenize()
    processor_without_stopwords.stem_tokens()
    processor_without_stopwords.unique_vocabulary()
    processor_without_stopwords.calculate_term_frequencies()

    vocabulary_with_stopwords = processor_with_stopwords.get_vocabulary()
    print(f"Vocabulary size (Stopwords Removed): {len(vocabulary_with_stopwords)}")

    vocabulary_without_stopwords = processor_without_stopwords.get_vocabulary()
    print(f"Vocabulary size (Stopwords Included): {len(vocabulary_without_stopwords)}")

    term_frequencies_without_stopwords = processor_without_stopwords.get_term_frequencies()

    plot_normalized_frequency_with_zipf(term_frequencies_with_stopwords, "Zipf's Law: Normalised Frequency against Rank (Stopwords Removed)", "blue")
    plot_normalized_frequency_with_zipf(term_frequencies_without_stopwords, "Zipf's Law: Normalised Frequency against Rank (Stopwords Included)", "red")

    plot_zipf_law(term_frequencies_with_stopwords, term_frequencies_without_stopwords)

    lsq_with_stopwords = calculate_least_squares(term_frequencies_with_stopwords, "Stopwords Removed")
    lsq_without_stopwords = calculate_least_squares(term_frequencies_without_stopwords, "Stopwords Included")

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")


    

        

        

                
       

 
    


    
    





    

    
    


