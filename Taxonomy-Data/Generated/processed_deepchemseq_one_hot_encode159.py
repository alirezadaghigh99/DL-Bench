import numpy as np
from Bio.SeqRecord import SeqRecord
from typing import Iterator, Union

def seq_one_hot_encode(sequences: Union[np.ndarray, Iterator[SeqRecord]], letters: str = "ATCGN") -> np.ndarray:
    # Convert sequences to a list of strings if they are SeqRecord objects
    if isinstance(sequences, Iterator):
        sequences = [str(record.seq) for record in sequences]
    elif isinstance(sequences, np.ndarray):
        sequences = sequences.tolist()
    
    # Check if all sequences are of the same length
    sequence_lengths = [len(seq) for seq in sequences]
    if len(set(sequence_lengths)) != 1:
        raise ValueError("All sequences must be of the same length.")
    
    sequence_length = sequence_lengths[0]
    N_sequences = len(sequences)
    N_letters = len(letters)
    
    # Create a mapping from letter to index
    letter_to_index = {letter: idx for idx, letter in enumerate(letters)}
    
    # Initialize the one-hot encoded array
    one_hot_encoded = np.zeros((N_sequences, N_letters, sequence_length, 1), dtype=np.float32)
    
    # Fill the one-hot encoded array
    for i, seq in enumerate(sequences):
        for j, letter in enumerate(seq):
            if letter in letter_to_index:
                one_hot_encoded[i, letter_to_index[letter], j, 0] = 1.0
    
    return one_hot_encoded

