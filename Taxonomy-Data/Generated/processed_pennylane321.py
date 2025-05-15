import numpy as np
import torch

def allequal(tensor1, tensor2, **kwargs):
    """Returns True if two tensors are element-wise equal along a given axis.

    This function is equivalent to calling ``np.all(tensor1 == tensor2, **kwargs)``,
    but allows for ``tensor1`` and ``tensor2`` to differ in type.

    Args:
        tensor1 (tensor_like): tensor to compare
        tensor2 (tensor_like): tensor to compare
        **kwargs: Accepts any keyword argument that is accepted by ``np.all``,
            such as ``axis``, ``out``, and ``keepdims``. See the `NumPy documentation
            <https://numpy.org/doc/stable/reference/generated/numpy.all.html>`__ for
            more details.

    Returns:
        ndarray, bool: If ``axis=None``, a logical AND reduction is applied to all elements
        and a boolean will be returned, indicating if all elements evaluate to ``True``. Otherwise,
        a boolean NumPy array will be returned.

    **Example**

    >>> a = torch.tensor([1, 2])
    >>> b = np.array([1, 2])
    >>> allequal(a, b)
    True
    """
    # Convert tensor1 to a NumPy array if it is a PyTorch tensor
    if isinstance(tensor1, torch.Tensor):
        tensor1 = tensor1.numpy()
    
    # Convert tensor2 to a NumPy array if it is a PyTorch tensor
    if isinstance(tensor2, torch.Tensor):
        tensor2 = tensor2.numpy()
    
    # Use np.all to check if all elements are equal
    return np.all(tensor1 == tensor2, **kwargs)