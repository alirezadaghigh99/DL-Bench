import pennylane as qml
from collections.abc import Sequence

def extract_tensors(x):
    """
    Iterate through an iterable, and extract any PennyLane tensors that appear.

    Args:
        x (.tensor or Sequence): an input tensor or sequence

    Yields:
        tensor: the next tensor in the sequence. If the input was a single
        tensor, then the tensor is yielded and the iterator completes.
    """
    # Check if the input is a single PennyLane tensor
    if isinstance(x, qml.numpy.tensor):
        yield x
    # Check if the input is a sequence
    elif isinstance(x, Sequence):
        for item in x:
            # Recursively yield tensors from the sequence
            yield from extract_tensors(item)
    else:
        # If the input is neither a tensor nor a sequence, do nothing
        return

