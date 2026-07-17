"""
This file contains functionalities for kernel related costs.
See `here <https://www.doi.org/10.1007/s10462-012-9369-4>`_ for a review.
"""
import numpy as np
from ..math import frobenius_inner_product
from .utils import square_kernel_matrix

def polarity(X, Y, kernel, assume_normalized_kernel=False, rescale_class_labels=True, normalize=False):
    """Generate a Python function called polarity that calculates the polarity of a given kernel function. The function takes in the following parameters: X (list of datapoints), Y (list of class labels of datapoints, assumed to be either -1 or 1), kernel (function that maps datapoints to kernel value), assume_normalized_kernel (optional boolean), rescale_class_labels (optional boolean), and normalize (boolean). 

The polarity of the kernel function is calculated as the sum of the product of class labels and kernel values for all pairs of datapoints. If the dataset is unbalanced, the class labels are rescaled based on the number of datapoints in each class. The function returns the kernel polarity as a float value.

An example is provided using a simple kernel function based on AngleEmbedding, where the polarity is computed for a set of feature vectors X with corresponding class labels Y. The function is called with X, Y, and the kernel function, and returns the computed polarity value."""
    if rescale_class_labels:
        nplus = np.count_nonzero(np.array(Y) == 1)
        nminus = len(Y) - nplus
        _Y = np.array([y / nplus if y == 1 else y / nminus for y in Y])
    else:
        _Y = np.array(Y)

    K = square_kernel_matrix(X, kernel, assume_normalized_kernel=assume_normalized_kernel)
    T = np.outer(_Y, _Y)

    return frobenius_inner_product(K, T, normalize=normalize)

def target_alignment(X, Y, kernel, assume_normalized_kernel=False, rescale_class_labels=True):
    """Target alignment of a given kernel function.

    This function is an alias for :func:`~.kernels.polarity` with ``normalize=True``.

    For a dataset with feature vectors :math:`\\{x_i\\}` and associated labels :math:`\\{y_i\\}`, the
    target alignment of the kernel function :math:`k` is given by

    .. math ::

        \\operatorname{TA}(k) = \\frac{\\sum_{i,j=1}^n y_i y_j k(x_i, x_j)}
        {\\sqrt{\\sum_{i,j=1}^n y_i y_j} \\sqrt{\\sum_{i,j=1}^n k(x_i, x_j)^2}}

    If the dataset is unbalanced, that is if the numbers of datapoints in the
    two classes :math:`n_+` and :math:`n_-` differ,
    ``rescale_class_labels=True`` will apply a rescaling according to
    :math:`\\tilde{y}_i = \\frac{y_i}{n_{y_i}}`. This is activated by default
    and only results in a prefactor that depends on the size of the dataset
    for balanced datasets.

    Args:
        X (list[datapoint]): List of datapoints
        Y (list[float]): List of class labels of datapoints, assumed to be either -1 or 1.
        kernel ((datapoint, datapoint) -> float): Kernel function that maps datapoints to kernel value.
        assume_normalized_kernel (bool, optional): Assume that the kernel is normalized, i.e.
            the kernel evaluates to 1 when both arguments are the same datapoint.
        rescale_class_labels (bool, optional): Rescale the class labels. This is important to take
            care of unbalanced datasets.

    Returns:
        float: The kernel-target alignment.

    **Example:**

    Consider a simple kernel function based on :class:`~.templates.embeddings.AngleEmbedding`:

    .. code-block :: python

        dev = qml.device('default.qubit', wires=2)
        @qml.qnode(dev)
        def circuit(x1, x2):
            qml.templates.AngleEmbedding(x1, wires=dev.wires)
            qml.adjoint(qml.templates.AngleEmbedding)(x2, wires=dev.wires)
            return qml.probs(wires=dev.wires)

        kernel = lambda x1, x2: circuit(x1, x2)[0]

    We can then compute the kernel-target alignment on a set of 4 (random)
    feature vectors ``X`` with labels ``Y`` via

    >>> X = np.random.random((4, 2))
    >>> Y = np.array([-1, -1, 1, 1])
    >>> qml.kernels.target_alignment(X, Y, kernel)
    tensor(0.01124802, requires_grad=True)

    We can see that this is equivalent to using ``normalize=True`` in
    ``polarity``:

    >>> target_alignment = qml.kernels.target_alignment(X, Y, kernel)
    >>> normalized_polarity = qml.kernels.polarity(X, Y, kernel, normalize=True)
    >>> np.isclose(target_alignment, normalized_polarity)
    tensor(True, requires_grad=True)
    """
    return polarity(X, Y, kernel, assume_normalized_kernel=assume_normalized_kernel, rescale_class_labels=rescale_class_labels, normalize=True)
