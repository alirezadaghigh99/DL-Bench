import torch

def su2_generators(k):
    # Calculate j from k
    j = k / 2
    dim = int(2 * j + 1)
    
    # Initialize matrices
    J_plus = torch.zeros((dim, dim), dtype=torch.complex64)
    J_minus = torch.zeros((dim, dim), dtype=torch.complex64)
    J_z = torch.zeros((dim, dim), dtype=torch.complex64)
    
    # Fill the matrices
    for m in range(dim):
        m_val = j - m
        if m < dim - 1:
            J_plus[m, m + 1] = torch.sqrt((j - m_val) * (j + m_val + 1))
        if m > 0:
            J_minus[m, m - 1] = torch.sqrt((j + m_val) * (j - m_val + 1))
        J_z[m, m] = m_val
    
    # Calculate J_x and J_y from J_plus and J_minus
    J_x = 0.5 * (J_plus + J_minus)
    J_y = -0.5j * (J_plus - J_minus)
    
    # Stack the generators into a single tensor
    generators = torch.stack((J_x, J_y, J_z))
    
    return generators

