from collections import Counter

def get_ngrams(sequence, n):
    """Generate n-grams from a sequence of words."""
    return [tuple(sequence[i:i+n]) for i in range(len(sequence)-n+1)]

def modified_precision(references, candidate, n):
    """
    Calculate the modified precision for a candidate translation given references and n-gram order.
    
    :param references: List of reference translations (each a list of words).
    :param candidate: Candidate translation (a list of words).
    :param n: The n-gram order.
    :return: A tuple (clipped_count_sum, total_count).
    """
    # Get n-grams for the candidate
    candidate_ngrams = get_ngrams(candidate, n)
    candidate_ngram_counts = Counter(candidate_ngrams)
    
    # Initialize maximum reference n-gram counts
    max_ref_ngram_counts = Counter()
    
    # Calculate maximum n-gram counts from references
    for reference in references:
        reference_ngrams = get_ngrams(reference, n)
        reference_ngram_counts = Counter(reference_ngrams)
        
        # Update maximum counts for each n-gram
        for ngram in reference_ngram_counts:
            max_ref_ngram_counts[ngram] = max(max_ref_ngram_counts[ngram], reference_ngram_counts[ngram])
    
    # Calculate clipped counts
    clipped_count_sum = 0
    for ngram in candidate_ngram_counts:
        clipped_count_sum += min(candidate_ngram_counts[ngram], max_ref_ngram_counts[ngram])
    
    # Total number of n-grams in the candidate
    total_count = sum(candidate_ngram_counts.values())
    
    return clipped_count_sum, total_count

