import torch

def _get_perspective_coeffs(startpoints, endpoints):
    if len(startpoints) != 4 or len(endpoints) != 4:
        raise ValueError("Exactly four startpoints and endpoints are required.")
    
    # Convert points to torch tensors
    startpoints = torch.tensor(startpoints, dtype=torch.float64)
    endpoints = torch.tensor(endpoints, dtype=torch.float64)
    
    # Prepare the matrix A and vector B for the equation A * coeffs = B
    A = []
    B = []
    
    for (x, y), (x_prime, y_prime) in zip(startpoints, endpoints):
        A.append([x, y, 1, 0, 0, 0, -x_prime * x, -x_prime * y])
        A.append([0, 0, 0, x, y, 1, -y_prime * x, -y_prime * y])
        B.append(x_prime)
        B.append(y_prime)
    
    A = torch.tensor(A, dtype=torch.float64)
    B = torch.tensor(B, dtype=torch.float64)
    
    # Solve the linear system A * coeffs = B using least squares
    coeffs, _ = torch.lstsq(B, A)
    
    # The result from torch.lstsq includes extra rows, we only need the first 8 coefficients
    coeffs = coeffs[:8].squeeze()
    
    # Convert coefficients to single precision
    coeffs = coeffs.to(torch.float32)
    
    return coeffs.tolist()

