import torch
import warnings

def safe_solve_with_mask(B, A):
    # Check if B is a tensor
    if not isinstance(B, torch.Tensor):
        raise AssertionError("B must be a PyTorch tensor.")
    
    # Ensure B is of type float32 or float64
    if B.dtype not in [torch.float32, torch.float64]:
        B = B.to(torch.float32)
    
    # Check PyTorch version
    pytorch_version = torch.__version__.split('.')
    major_version = int(pytorch_version[0])
    minor_version = int(pytorch_version[1])
    
    # Initialize the valid_mask as all True
    valid_mask = torch.ones(B.size(0), dtype=torch.bool)
    
    try:
        if major_version > 1 or (major_version == 1 and minor_version >= 10):
            # Use torch.linalg.lu_factor and lu_solve for PyTorch >= 1.10
            A_LU, pivots = torch.linalg.lu_factor(A)
            X = torch.linalg.lu_solve(A_LU, pivots, B)
        else:
            # Fallback for PyTorch < 1.10
            warnings.warn("PyTorch version < 1.10 detected. Using fallback method. Validity mask may not be correct.")
            X, _ = torch.solve(B, A)
            A_LU = None  # LU decomposition not available in this version
            return X, A_LU, valid_mask
        
        # Check for singular matrices by verifying if any diagonal element of A_LU is zero
        if A_LU is not None:
            singular_rows = torch.any(A_LU.diagonal(dim1=-2, dim2=-1) == 0, dim=-1)
            valid_mask = ~singular_rows
        
    except RuntimeError as e:
        # Handle singular matrix case
        warnings.warn(f"RuntimeError encountered: {e}. Returning zero tensor for X.")
        X = torch.zeros_like(B)
        valid_mask = torch.zeros(B.size(0), dtype=torch.bool)
        A_LU = None
    
    return X, A_LU, valid_mask