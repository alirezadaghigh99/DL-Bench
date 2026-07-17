"""
This submodule contains the discrete-variable quantum operations that
accept a hermitian or an unitary matrix as a parameter.
"""
import warnings
from itertools import product
from typing import Optional, Union
import numpy as np
from scipy.linalg import fractional_matrix_power
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.math import cast, conj, eye, norm, sqrt, sqrt_matrix, transpose, zeros
from pennylane.operation import AnyWires, DecompositionUndefinedError, FlatPytree, Operation
from pennylane.typing import TensorLike
from pennylane.wires import Wires, WiresLike
_walsh_hadamard_matrix = np.array([[1, 1], [1, -1]]) / 2

def _walsh_hadamard_transform(D: TensorLike, n: Optional[int]=None):
    """Compute the Walsh-Hadamard Transform of a tensor or array ``D`` along its last axis.

    Args:
        D (tensor_like): The tensor to transform. The size of its last axis must be a
            power of two, ``2**n``. ``D`` may optionally carry one leading batch axis.
        n (int): Number of qubits/bits the transform acts on, i.e. the last axis of
            ``D`` has size ``2**n``. Defaults to being inferred from the shape of ``D``.

    Returns:
        tensor_like: The transformed tensor, with the same shape as the input ``D``.

    The transform is carried out via a sequence of tensor contractions with the
    ``2 x 2`` Hadamard matrix, one per bit, which keeps the computation compatible
    with autodifferentiation frameworks.
    """
    orig_shape = qml.math.shape(D)
    if n is None:
        n = int(qml.math.log2(orig_shape[-1]))
    # Split the last axis into `n` axes of size 2 each, keeping a possible batch axis intact.
    broadcasted = len(orig_shape) > 1
    new_shape = (orig_shape[0],) + (2,) * n if broadcasted else (2,) * n
    D = qml.math.reshape(D, new_shape)
    for i in range(n):
        D = qml.math.tensordot(_walsh_hadamard_matrix, D, axes=[[1], [i + broadcasted]])
    # Each contraction prepends the newly created axis, reversing the axis order (and
    # pushing the batch axis, if any, to the end), so undo that before reshaping back.
    return qml.math.reshape(qml.math.transpose(D), orig_shape)

class QubitUnitary(Operation):
    """QubitUnitary(U, wires)
    Apply an arbitrary unitary matrix with a dimension that is a power of two.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Number of dimensions per parameter: (2,)
    * Gradient recipe: None

    Args:
        U (array[complex]): square unitary matrix
        wires (Sequence[int] or int): the wire(s) the operation acts on
        id (str): custom label given to an operator instance,
            can be useful for some applications where the instance has to be identified
        unitary_check (bool): check for unitarity of the given matrix

    Raises:
        ValueError: if the number of wires doesn't fit the dimensions of the matrix

    **Example**

    >>> dev = qml.device('default.qubit', wires=1)
    >>> U = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
    >>> @qml.qnode(dev)
    ... def example_circuit():
    ...     qml.QubitUnitary(U, wires=0)
    ...     return qml.expval(qml.Z(0))
    >>> print(example_circuit())
    0.0
    """
    num_wires = AnyWires
    'int: Number of wires that the operator acts on.'
    num_params = 1
    'int: Number of trainable parameters that the operator depends on.'
    ndim_params = (2,)
    'tuple[int]: Number of dimensions per trainable parameter that the operator depends on.'
    grad_method = None
    'Gradient computation method.'

    def __init__(self, U: TensorLike, wires: WiresLike, id: Optional[str]=None, unitary_check: bool=False):
        wires = Wires(wires)
        U_shape = qml.math.shape(U)
        dim = 2 ** len(wires)
        if len(U_shape) not in {2, 3} or U_shape[-2:] != (dim, dim):
            raise ValueError(f'Input unitary must be of shape {(dim, dim)} or (batch_size, {dim}, {dim}) to act on {len(wires)} wires.')
        if unitary_check and (not (qml.math.is_abstract(U) or qml.math.allclose(qml.math.einsum('...ij,...kj->...ik', U, qml.math.conj(U)), qml.math.eye(dim), atol=1e-06))):
            warnings.warn(f'Operator {U}\n may not be unitary. Verify unitarity of operation, or use a datatype with increased precision.', UserWarning)
        super().__init__(U, wires=wires, id=id)

    @staticmethod
    def compute_matrix(U: TensorLike):
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.QubitUnitary.matrix`

        Args:
            U (tensor_like): unitary matrix

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> U = np.array([[0.98877108+0.j, 0.-0.14943813j], [0.-0.14943813j, 0.98877108+0.j]])
        >>> qml.QubitUnitary.compute_matrix(U)
        [[0.98877108+0.j, 0.-0.14943813j],
        [0.-0.14943813j, 0.98877108+0.j]]
        """
        return U

    @staticmethod
    def compute_decomposition(U: TensorLike, wires: WiresLike):
        """Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \\dots O_n.

        A decomposition is only defined for matrices that act on either one or two wires. For more
        than two wires, this method raises a ``DecompositionUndefined``.

        See :func:`~.transforms.one_qubit_decomposition` and :func:`~.ops.two_qubit_decomposition`
        for more information on how the decompositions are computed.

        .. seealso:: :meth:`~.QubitUnitary.decomposition`.

        Args:
            U (array[complex]): square unitary matrix
            wires (Iterable[Any] or Wires): the wire(s) the operation acts on

        Returns:
            list[Operator]: decomposition of the operator

        **Example:**

        >>> U = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
        >>> qml.QubitUnitary.compute_decomposition(U, 0)
        [Rot(tensor(3.14159265, requires_grad=True), tensor(1.57079633, requires_grad=True), tensor(0., requires_grad=True), wires=[0])]

        """
        shape = qml.math.shape(U)
        is_batched = len(shape) == 3
        shape_without_batch_dim = shape[1:] if is_batched else shape
        if shape_without_batch_dim == (2, 2):
            return qml.ops.one_qubit_decomposition(U, Wires(wires)[0])
        if shape_without_batch_dim == (4, 4):
            if is_batched:
                raise DecompositionUndefinedError('The decomposition of a two-qubit QubitUnitary does not support broadcasting.')
            return qml.ops.two_qubit_decomposition(U, Wires(wires))
        return super(QubitUnitary, QubitUnitary).compute_decomposition(U, wires=wires)

    @property
    def has_decomposition(self) -> bool:
        return len(self.wires) < 3

    def adjoint(self) -> 'QubitUnitary':
        U = self.matrix()
        return QubitUnitary(qml.math.moveaxis(qml.math.conj(U), -2, -1), wires=self.wires)

    def pow(self, z: Union[int, float]):
        mat = self.matrix()
        if isinstance(z, int) and qml.math.get_deep_interface(mat) != 'tensorflow':
            pow_mat = qml.math.linalg.matrix_power(mat, z)
        elif self.batch_size is not None or qml.math.shape(z) != ():
            return super().pow(z)
        else:
            pow_mat = qml.math.convert_like(fractional_matrix_power(mat, z), mat)
        return [QubitUnitary(pow_mat, wires=self.wires)]

    def _controlled(self, wire):
        return qml.ControlledQubitUnitary(*self.parameters, control_wires=wire, wires=self.wires)

    def label(self, decimals: Optional[int]=None, base_label: Optional[str]=None, cache: Optional[dict]=None) -> str:
        return super().label(decimals=decimals, base_label=base_label or 'U', cache=cache)

class DiagonalQubitUnitary(Operation):
    """DiagonalQubitUnitary(D, wires)
    Apply an arbitrary diagonal unitary matrix with a dimension that is a power of two.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Number of dimensions per parameter: (1,)
    * Gradient recipe: None

    Args:
        D (array[complex]): diagonal of unitary matrix
        wires (Sequence[int] or int): the wire(s) the operation acts on
    """
    num_wires = AnyWires
    'int: Number of wires that the operator acts on.'
    num_params = 1
    'int: Number of trainable parameters that the operator depends on.'
    ndim_params = (1,)
    'tuple[int]: Number of dimensions per trainable parameter that the operator depends on.'
    grad_method = None
    'Gradient computation method.'

    @staticmethod
    def compute_matrix(D: TensorLike) -> TensorLike:
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.DiagonalQubitUnitary.matrix`

        Args:
            D (tensor_like): diagonal of the matrix

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.DiagonalQubitUnitary.compute_matrix(torch.tensor([1, -1]))
        tensor([[ 1,  0],
                [ 0, -1]])
        """
        D = qml.math.asarray(D)
        if not qml.math.is_abstract(D) and (not qml.math.allclose(D * qml.math.conj(D), qml.math.ones_like(D))):
            raise ValueError('Operator must be unitary.')
        if qml.math.ndim(D) == 2:
            return qml.math.stack([qml.math.diag(_D) for _D in D])
        return qml.math.diag(D)

    @staticmethod
    def compute_eigvals(D: TensorLike) -> TensorLike:
        """Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \\Sigma U^{\\dagger},

        where :math:`\\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.DiagonalQubitUnitary.eigvals`

        Args:
            D (tensor_like): diagonal of the matrix

        Returns:
            tensor_like: eigenvalues

        **Example**

        >>> qml.DiagonalQubitUnitary.compute_eigvals(torch.tensor([1, -1]))
        tensor([ 1, -1])
        """
        D = qml.math.asarray(D)
        if not (qml.math.is_abstract(D) or qml.math.allclose(D * qml.math.conj(D), qml.math.ones_like(D))):
            raise ValueError('Operator must be unitary.')
        return D

    @staticmethod
    def compute_decomposition(D: TensorLike, wires: WiresLike) -> list['qml.operation.Operator']:
        """Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \\dots O_n.

        ``DiagonalQubitUnitary`` decomposes into :class:`~.QubitUnitary`, :class:`~.RZ`,
        :class:`~.IsingZZ`, and/or :class:`~.MultiRZ` depending on the number of wires.

        .. note::

            The parameters of the decomposed operations are cast to the ``complex128`` dtype
            as real dtypes can lead to ``NaN`` values in the decomposition.

        .. seealso:: :meth:`~.DiagonalQubitUnitary.decomposition`.

        Args:
            D (tensor_like): diagonal of the matrix
            wires (Iterable[Any] or Wires): the wire(s) the operation acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> diag = np.exp(1j * np.array([0.4, 2.1, 0.5, 1.8]))
        >>> qml.DiagonalQubitUnitary.compute_decomposition(diag, wires=[0, 1])
        [QubitUnitary(array([[0.36235775+0.93203909j, 0.        +0.j        ],
         [0.        +0.j        , 0.36235775+0.93203909j]]), wires=[0]),
         RZ(1.5000000000000002, wires=[1]),
         RZ(-0.10000000000000003, wires=[0]),
         IsingZZ(0.2, wires=[0, 1])]

        """
        n = len(wires)
        D_casted = qml.math.cast(D, 'complex128')
        phases = qml.math.real(qml.math.log(D_casted) * -1j)
        coeffs = _walsh_hadamard_transform(phases, n).T
        global_phase = qml.math.exp(1j * coeffs[0])
        coeffs = coeffs * -2.0
        ops = [QubitUnitary(qml.math.tensordot(global_phase, qml.math.eye(2), axes=0), wires[0])]
        for wire0 in range(n):
            ops.append(qml.RZ(coeffs[1 << wire0], wires[n - 1 - wire0]))
            ops.extend((qml.IsingZZ(coeffs[(1 << wire0) + (1 << wire1)], [wires[n - 1 - wire0], wires[n - 1 - wire1]]) for wire1 in range(wire0)))
        ops.extend((qml.MultiRZ(c, [wires[k] for k in np.where(term)[0]]) for (c, term) in zip(coeffs, product((0, 1), repeat=n)) if sum(term) > 2))
        return ops

    def adjoint(self) -> 'DiagonalQubitUnitary':
        return DiagonalQubitUnitary(qml.math.conj(self.parameters[0]), wires=self.wires)

    def pow(self, z) -> list['DiagonalQubitUnitary']:
        cast_data = qml.math.cast(self.data[0], np.complex128)
        return [DiagonalQubitUnitary(cast_data ** z, wires=self.wires)]

    def _controlled(self, control: WiresLike):
        return DiagonalQubitUnitary(qml.math.hstack([np.ones_like(self.parameters[0]), self.parameters[0]]), wires=control + self.wires)

    def label(self, decimals: Optional[int]=None, base_label: Optional[str]=None, cache: Optional[dict]=None):
        return super().label(decimals=decimals, base_label=base_label or 'U', cache=cache)

class BlockEncode(Operation):
    """BlockEncode(A, wires)
    Construct a unitary :math:`U(A)` such that an arbitrary matrix :math:`A`
    is encoded in the top-left block.

    .. math::

        \\begin{align}
             U(A) &=
             \\begin{bmatrix}
                A & \\sqrt{I-AA^\\dagger} \\\\
                \\sqrt{I-A^\\dagger A} & -A^\\dagger
            \\end{bmatrix}.
        \\end{align}

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Number of dimensions per parameter: (2,)
    * Gradient recipe: None

    Args:
        A (tensor_like): a general :math:`(n \\times m)` matrix to be encoded
        wires (Iterable[int, str], Wires): the wires the operation acts on
        id (str or None): String representing the operation (optional)

    Raises:
        ValueError: if the number of wires doesn't fit the dimensions of the matrix

    **Example**

    We can define a matrix and a block-encoding circuit as follows:

    >>> A = [[0.1,0.2],[0.3,0.4]]
    >>> dev = qml.device('default.qubit', wires=2)
    >>> @qml.qnode(dev)
    ... def example_circuit():
    ...     qml.BlockEncode(A, wires=range(2))
    ...     return qml.state()

    We can see that :math:`A` has been block encoded in the matrix of the circuit:

    >>> print(qml.matrix(example_circuit)())
    [[ 0.1         0.2         0.97283788 -0.05988708]
     [ 0.3         0.4        -0.05988708  0.86395228]
     [ 0.94561648 -0.07621992 -0.1        -0.3       ]
     [-0.07621992  0.89117368 -0.2        -0.4       ]]

    We can also block-encode a non-square matrix and check the resulting unitary matrix:

    >>> A = [[0.2, 0, 0.2],[-0.2, 0.2, 0]]
    >>> op = qml.BlockEncode(A, wires=range(3))
    >>> print(np.round(qml.matrix(op), 2))
    [[ 0.2   0.    0.2   0.96  0.02  0.    0.    0.  ]
     [-0.2   0.2   0.    0.02  0.96  0.    0.    0.  ]
     [ 0.96  0.02 -0.02 -0.2   0.2   0.    0.    0.  ]
     [ 0.02  0.98  0.   -0.   -0.2   0.    0.    0.  ]
     [-0.02  0.    0.98 -0.2  -0.    0.    0.    0.  ]
     [ 0.    0.    0.    0.    0.    1.    0.    0.  ]
     [ 0.    0.    0.    0.    0.    0.    1.    0.  ]
     [ 0.    0.    0.    0.    0.    0.    0.    1.  ]]

    .. note::
        If the operator norm of :math:`A`  is greater than 1, we normalize it to ensure
        :math:`U(A)` is unitary. The normalization constant can be
        accessed through :code:`op.hyperparameters["norm"]`.

        Specifically, the norm is computed as the maximum of
        :math:`\\| AA^\\dagger \\|` and
        :math:`\\| A^\\dagger A \\|`.
    """
    num_params = 1
    'int: Number of trainable parameters that the operator depends on.'
    num_wires = AnyWires
    'int: Number of wires that the operator acts on.'
    ndim_params = (2,)
    'tuple[int]: Number of dimensions per trainable parameter that the operator depends on.'
    grad_method = None
    'Gradient computation method.'

    def __init__(self, A: TensorLike, wires: WiresLike, id: Optional[str]=None):
        wires = Wires(wires)
        shape_a = qml.math.shape(A)
        if shape_a == () or all((x == 1 for x in shape_a)):
            A = qml.math.reshape(A, [1, 1])
            normalization = qml.math.abs(A)
            subspace = (1, 1, 2 ** len(wires))
        else:
            if len(shape_a) == 1:
                A = qml.math.reshape(A, [1, len(A)])
                shape_a = qml.math.shape(A)
            normalization = qml.math.maximum(norm(A @ qml.math.transpose(qml.math.conj(A)), ord=pnp.inf), norm(qml.math.transpose(qml.math.conj(A)) @ A, ord=pnp.inf))
            subspace = (*shape_a, 2 ** len(wires))
        A = qml.math.array(A) / qml.math.maximum(normalization, qml.math.ones_like(normalization))
        if subspace[2] < subspace[0] + subspace[1]:
            raise ValueError(f'Block encoding a ({subspace[0]} x {subspace[1]}) matrix requires a Hilbert space of size at least ({subspace[0] + subspace[1]} x {subspace[0] + subspace[1]}). Cannot be embedded in a {len(wires)} qubit system.')
        super().__init__(A, wires=wires, id=id)
        self.hyperparameters['norm'] = normalization
        self.hyperparameters['subspace'] = subspace

    def _flatten(self) -> FlatPytree:
        return (self.data, (self.wires, ()))

    @staticmethod
    def compute_matrix(*params, **hyperparams):
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.BlockEncode.matrix`

        Args:
            *params (list): trainable parameters of the operator, as stored in the ``parameters`` attribute
            **hyperparams (dict): non-trainable hyperparameters of the operator, as stored in the ``hyperparameters`` attribute


        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> A = np.array([[0.1,0.2],[0.3,0.4]])
        >>> A
        tensor([[0.1, 0.2],
                [0.3, 0.4]])
        >>> qml.BlockEncode.compute_matrix(A, subspace=[2,2,4])
        array([[ 0.1       ,  0.2       ,  0.97283788, -0.05988708],
               [ 0.3       ,  0.4       , -0.05988708,  0.86395228],
               [ 0.94561648, -0.07621992, -0.1       , -0.3       ],
               [-0.07621992,  0.89117368, -0.2       , -0.4       ]])
        """
        A = params[0]
        (n, m, k) = hyperparams['subspace']
        shape_a = qml.math.shape(A)

        def _stack(lst, h=False, like=None):
            if like == 'tensorflow':
                axis = 1 if h else 0
                return qml.math.concat(lst, like=like, axis=axis)
            return qml.math.hstack(lst) if h else qml.math.vstack(lst)
        interface = qml.math.get_interface(A)
        if qml.math.sum(shape_a) <= 2:
            col1 = _stack([A, sqrt(1 - A * conj(A))], like=interface)
            col2 = _stack([sqrt(1 - A * conj(A)), -conj(A)], like=interface)
            u = _stack([col1, col2], h=True, like=interface)
        else:
            (d1, d2) = shape_a
            col1 = _stack([A, sqrt_matrix(cast(eye(d2, like=A), A.dtype) - qml.math.transpose(conj(A)) @ A)], like=interface)
            col2 = _stack([sqrt_matrix(cast(eye(d1, like=A), A.dtype) - A @ transpose(conj(A))), -transpose(conj(A))], like=interface)
            u = _stack([col1, col2], h=True, like=interface)
        if n + m < k:
            r = k - (n + m)
            col1 = _stack([u, zeros((r, n + m), like=A)], like=interface)
            col2 = _stack([zeros((n + m, r), like=A), eye(r, like=A)], like=interface)
            u = _stack([col1, col2], h=True, like=interface)
        return u

    def adjoint(self) -> 'BlockEncode':
        A = self.parameters[0]
        return BlockEncode(qml.math.transpose(qml.math.conj(A)), wires=self.wires)

    def label(self, decimals: Optional[int]=None, base_label: Optional[str]=None, cache: Optional[dict]=None):
        return super().label(decimals=decimals, base_label=base_label or 'BlockEncode', cache=cache)
