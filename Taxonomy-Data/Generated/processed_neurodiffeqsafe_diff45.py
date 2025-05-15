import torch

def safe_diff(u, t, order=1):
    # Check if u and t have the correct shape
    if u.shape != t.shape or u.shape[1] != 1:
        raise ValueError("Both u and t must have the shape (n_samples, 1) and must be the same shape.")
    
    # Ensure that u and t are PyTorch tensors
    if not isinstance(u, torch.Tensor) or not isinstance(t, torch.Tensor):
        raise ValueError("Both u and t must be PyTorch tensors.")
    
    # Check if order is a positive integer
    if not isinstance(order, int) or order < 1:
        raise ValueError("Order must be a positive integer.")
    
    # Initialize the derivative
    derivative = u
    
    # Compute the derivative of the specified order
    for _ in range(order):
        # Ensure that the current derivative tensor requires gradients
        derivative.requires_grad_(True)
        
        # Compute the gradient of the current derivative with respect to t
        grad_outputs = torch.ones_like(derivative)
        derivative = torch.autograd.grad(outputs=derivative, inputs=t, grad_outputs=grad_outputs, create_graph=True)[0]
    
    return derivative

