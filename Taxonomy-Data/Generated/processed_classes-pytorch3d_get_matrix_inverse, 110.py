import torch

class Rotate(Transform3d):
    def __init__(
        self,
        R: torch.Tensor,
        dtype: torch.dtype = torch.float32,
        device: Optional[Device] = None,
        orthogonal_tol: float = 1e-5,
    ) -> None:
        device_ = get_device(R, device)
        super().__init__(device=device_, dtype=dtype)
        if R.dim() == 2:
            R = R[None]
        if R.shape[-2:] != (3, 3):
            msg = "R must have shape (3, 3) or (N, 3, 3); got %s"
            raise ValueError(msg % repr(R.shape))
        R = R.to(device=device_, dtype=dtype)
        if os.environ.get("PYTORCH3D_CHECK_ROTATION_MATRICES", "0") == "1":
            _check_valid_rotation_matrix(R, tol=orthogonal_tol)
        N = R.shape[0]
        mat = torch.eye(4, dtype=dtype, device=device_)
        mat = mat.view(1, 4, 4).repeat(N, 1, 1)
        mat[:, :3, :3] = R
        self._matrix = mat

    def _get_matrix_inverse(self) -> torch.Tensor:
        # Extract the rotation part of the matrix
        R = self._matrix[:, :3, :3]
        # Transpose the rotation part to get the inverse
        R_inv = R.transpose(-1, -2)
        # Create an identity matrix for the 4x4 structure
        mat_inv = torch.eye(4, dtype=self._matrix.dtype, device=self._matrix.device)
        # Repeat the identity matrix for the batch size
        mat_inv = mat_inv.view(1, 4, 4).repeat(R_inv.shape[0], 1, 1)
        # Set the top-left 3x3 part to the transposed rotation matrix
        mat_inv[:, :3, :3] = R_inv
        return mat_inv