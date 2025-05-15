import torch

def kron(t1, t2):
    # Get the shapes of the input tensors
    t1_rows, t1_cols = t1.shape
    t2_rows, t2_cols = t2.shape
    
    # Compute the shape of the resulting Kronecker product
    kron_rows = t1_rows * t2_rows
    kron_cols = t1_cols * t2_cols
    
    # Initialize the result tensor with the appropriate shape
    kron_product = torch.zeros((kron_rows, kron_cols), dtype=t1.dtype, device=t1.device)
    
    # Compute the Kronecker product
    for i in range(t1_rows):
        for j in range(t1_cols):
            # Compute the block for element (i, j) of t1
            kron_product[i*t2_rows:(i+1)*t2_rows, j*t2_cols:(j+1)*t2_cols] = t1[i, j] * t2
    
    return kron_product

