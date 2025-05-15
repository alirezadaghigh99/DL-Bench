import os
from collections import Counter

def build_charlm_vocab(file_path, cutoff=1):
    """
    Builds a vocabulary for a CharacterLanguageModel from files in the specified path.
    
    Parameters:
    - file_path: str, path to the file or directory containing text files.
    - cutoff: int, optional, minimum frequency for a character to be included in the vocabulary.
    
    Returns:
    - vocab: list of characters that meet the frequency cutoff.
    
    Raises:
    - ValueError: if the training data is empty or all characters are less frequent than the cutoff.
    """
    char_counter = Counter()
    
    # Check if the path is a directory or a single file
    if os.path.isdir(file_path):
        files = [os.path.join(file_path, f) for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
    else:
        files = [file_path]
    
    # Read each file and update the character counter
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                char_counter.update(line)
    
    # Filter characters based on the cutoff
    vocab = [char for char, count in char_counter.items() if count >= cutoff]
    
    # Check if the vocabulary is empty
    if not vocab:
        raise ValueError("Training data is empty or all characters are less frequent than the cutoff.")
    
    return vocab

