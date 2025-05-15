def hamming_distance(hash1: str, hash2: str) -> float:
    # Ensure both hashes are 64 bits long by padding with zeros if necessary
    hash1 = hash1.ljust(64, '0')
    hash2 = hash2.ljust(64, '0')
    
    # Calculate the Hamming distance
    distance = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
    
    return float(distance)

