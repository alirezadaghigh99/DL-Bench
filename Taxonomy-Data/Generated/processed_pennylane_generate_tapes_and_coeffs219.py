def _generate_tapes_and_coeffs(tape, idx, atol, cache):
    """
    Generate modified tapes and coefficients for the pulse generator derivative
    of a tape with respect to a specified trainable parameter.

    Args:
        tape (QuantumTape): The quantum tape to differentiate.
        idx (int): The index of the trainable parameter.
        atol (float): Absolute tolerance for numerical stability.
        cache (dict): A dictionary for caching previously computed results.

    Returns:
        tuple: A tuple containing:
            - A list of modified tapes for differentiation.
            - A tuple with the start and end indices into the total list of tapes.
            - A list of coefficients for contraction.
            - The updated cache dictionary.
    """
    # Check if the result is already cached
    if idx in cache:
        return [], (0, 0), [], cache

    # Initialize the list of modified tapes and coefficients
    modified_tapes = []
    coefficients = []

    # Iterate over the operations in the tape
    for op_idx, op in enumerate(tape.operations):
        # Check if the operation is parameterized and the parameter is trainable
        if op.is_parameterized and idx in op.trainable_params:
            # Create a modified tape for the derivative
            modified_tape = tape.copy()
            # Apply a small perturbation to the parameter
            param_shift = atol
            original_param = op.parameters[idx]
            
            # Forward shift
            modified_tape.set_parameters(op_idx, idx, original_param + param_shift)
            modified_tapes.append(modified_tape)
            coefficients.append(1.0 / (2 * param_shift))
            
            # Backward shift
            modified_tape = tape.copy()
            modified_tape.set_parameters(op_idx, idx, original_param - param_shift)
            modified_tapes.append(modified_tape)
            coefficients.append(-1.0 / (2 * param_shift))

    # Update the cache
    cache[idx] = (modified_tapes, coefficients)

    # Return the modified tapes, indices, coefficients, and updated cache
    start_idx = 0
    end_idx = len(modified_tapes)
    return modified_tapes, (start_idx, end_idx), coefficients, cache