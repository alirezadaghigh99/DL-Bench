import torch
from torch.optim import Adam
from gpytorch.mlls import VariationalELBO
from gpytorch.models import ApproximateGP

def train(gpmodule: ApproximateGP, 
          optimizer: torch.optim.Optimizer = None, 
          loss_fn = None, 
          retain_graph: bool = False, 
          num_steps: int = 100) -> list:
    """
    Trains a Gaussian Process module using Stochastic Variational Inference (SVI).

    Parameters:
    - gpmodule: A Gaussian Process module (instance of ApproximateGP).
    - optimizer: A PyTorch optimizer instance (default is Adam with learning rate 0.01).
    - loss_fn: A loss function that calculates the ELBO loss (default is VariationalELBO).
    - retain_graph: An optional flag for torch.autograd.backward.
    - num_steps: Number of steps to run SVI.

    Returns:
    - A list of losses during the training procedure.
    """
    # Set default optimizer if not provided
    if optimizer is None:
        optimizer = Adam(gpmodule.parameters(), lr=0.01)
    
    # Set default loss function if not provided
    if loss_fn is None:
        # Assuming a likelihood and train_loader are defined elsewhere
        likelihood = gpmodule.likelihood
        train_loader = gpmodule.train_loader
        loss_fn = VariationalELBO(likelihood, gpmodule, num_data=len(train_loader.dataset))
    
    # List to store the loss values
    losses = []

    # Training loop
    gpmodule.train()
    for step in range(num_steps):
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = gpmodule(x_batch)
            loss = -loss_fn(output, y_batch)
            loss.backward(retain_graph=retain_graph)
            optimizer.step()
            losses.append(loss.item())
            print(f"Step {step + 1}/{num_steps}, Loss: {loss.item()}")

    return losses