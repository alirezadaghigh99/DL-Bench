def pauli_basis_strings(num_wires):
    from itertools import product

    # Define the Pauli operators
    pauli_operators = ['I', 'X', 'Y', 'Z']

    # Generate all possible combinations of Pauli operators for the given number of wires
    all_combinations = product(pauli_operators, repeat=num_wires)

    # Filter out the identity string 'I' * num_wires
    pauli_words = [''.join(combination) for combination in all_combinations if 'I' * num_wires != ''.join(combination)]

    return pauli_words

