import torch

def jacobians_naive(model, data):
    """
    Calculate the Jacobian matrix of a given model with respect to the input data.

    Parameters:
    - model: The neural network model.
    - data: The input data (torch.Tensor).

    Returns:
    - Jacs: The Jacobian matrix.
    - f: The output tensor of the model.
    """
    # Ensure the input data requires gradients
    data = data.clone().detach().requires_grad_(True)
    
    # Zero the gradients
    model.zero_grad()
    
    # Compute the output of the model
    f = model(data)
    
    # Initialize a list to store the Jacobian matrices for each output
    Jacs = []
    
    # Iterate over each element in the output tensor
    for i in range(f.shape[0]):
        # Create a zero tensor with the same shape as the output
        grad_output = torch.zeros_like(f)
        
        # Set the current element to 1 to compute its gradient
        grad_output[i] = 1
        
        # Compute the gradients of the current output element with respect to the input
        f.backward(grad_output, retain_graph=True)
        
        # Append the gradient (Jacobian row) to the list
        Jacs.append(data.grad.clone().detach())
        
        # Zero the gradients for the next iteration
        data.grad.zero_()
    
    # Stack the list of Jacobian rows into a single tensor
    Jacs = torch.stack(Jacs)
    
    # Detach the output tensor from the computation graph
    f = f.clone().detach()
    
    return Jacs, f

