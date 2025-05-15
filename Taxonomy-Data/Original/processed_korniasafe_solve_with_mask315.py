def safe_solve_with_mask(B: Tensor, A: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    r"""Helper function, which avoids crashing because of singular matrix input and outputs the mask of valid
    solution."""
    if not torch_version_ge(1, 10):
        sol = _torch_solve_cast(A, B)
        warnings.warn("PyTorch version < 1.10, solve validness mask maybe not correct", RuntimeWarning)
        return sol, sol, torch.ones(len(A), dtype=torch.bool, device=A.device)
    # Based on https://github.com/pytorch/pytorch/issues/31546#issuecomment-694135622
    if not isinstance(B, Tensor):
        raise AssertionError(f"B must be Tensor. Got: {type(B)}.")
    dtype: torch.dtype = B.dtype
    if dtype not in (torch.float32, torch.float64):
        dtype = torch.float32

    if TYPE_CHECKING:
        # TODO: remove this branch when kornia relies on torch >= 1.13
        A_LU: Tensor
        pivots: Tensor
        info: Tensor
    elif torch_version_ge(1, 13):
        A_LU, pivots, info = torch.linalg.lu_factor_ex(A.to(dtype))
    else:
        # TODO: remove this branch when kornia relies on torch >= 1.13
        A_LU, pivots, info = torch.lu(A.to(dtype), True, get_infos=True)

    valid_mask: Tensor = info == 0
    n_dim_B = len(B.shape)
    n_dim_A = len(A.shape)
    if n_dim_A - n_dim_B == 1:
        B = B.unsqueeze(-1)

    if TYPE_CHECKING:
        # TODO: remove this branch when kornia relies on torch >= 1.13
        X: Tensor
    elif torch_version_ge(1, 13):
        X = torch.linalg.lu_solve(A_LU, pivots, B.to(dtype))
    else:
        # TODO: remove this branch when kornia relies on torch >= 1.13
        X = torch.lu_solve(B.to(dtype), A_LU, pivots)

    return X.to(B.dtype), A_LU.to(A.dtype), valid_mask