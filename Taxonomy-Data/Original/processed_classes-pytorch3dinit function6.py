    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        device: Device = "cpu",
        matrix: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Args:
            dtype: The data type of the transformation matrix.
                to be used if `matrix = None`.
            device: The device for storing the implemented transformation.
                If `matrix != None`, uses the device of input `matrix`.
            matrix: A tensor of shape (4, 4) or of shape (minibatch, 4, 4)
                representing the 4x4 3D transformation matrix.
                If `None`, initializes with identity using
                the specified `device` and `dtype`.
        """

        if matrix is None:
            self._matrix = torch.eye(4, dtype=dtype, device=device).view(1, 4, 4)
        else:
            if matrix.ndim not in (2, 3):
                raise ValueError('"matrix" has to be a 2- or a 3-dimensional tensor.')
            if matrix.shape[-2] != 4 or matrix.shape[-1] != 4:
                raise ValueError(
                    '"matrix" has to be a tensor of shape (minibatch, 4, 4) or (4, 4).'
                )
            # set dtype and device from matrix
            dtype = matrix.dtype
            device = matrix.device
            self._matrix = matrix.view(-1, 4, 4)

        self._transforms = []  # store transforms to compose
        self._lu = None
        self.device = make_device(device)
        self.dtype = dtype