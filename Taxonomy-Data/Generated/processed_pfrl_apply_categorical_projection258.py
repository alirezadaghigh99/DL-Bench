import torch

def _apply_categorical_projection(y, y_probs, z):
    """
    Projects the probabilities of atoms onto a new set of atom values.

    Parameters:
    y (torch.Tensor): Values of atoms before projection with shape (batch_size, n_atoms).
    y_probs (torch.Tensor): Probabilities of atoms with shape (batch_size, n_atoms).
    z (torch.Tensor): Values of atoms after projection with shape (n_atoms,).

    Returns:
    torch.Tensor: Projected probabilities with shape (batch_size, n_atoms).
    """
    # Ensure input shapes are as expected
    assert y.ndim == 2, "y should be a 2D tensor with shape (batch_size, n_atoms)"
    assert y_probs.ndim == 2, "y_probs should be a 2D tensor with shape (batch_size, n_atoms)"
    assert z.ndim == 1, "z should be a 1D tensor with shape (n_atoms,)"
    assert y.shape == y_probs.shape, "y and y_probs should have the same shape"
    assert y.shape[1] == z.shape[0], "The number of atoms in y and z should match"

    batch_size, n_atoms = y.shape

    # Calculate the width of each atom in z
    delta_z = z[1] - z[0]

    # Initialize the projected probabilities
    projected_probs = torch.zeros_like(y_probs)

    # Iterate over each batch
    for b in range(batch_size):
        # Clamp y values to be within the range of z
        y_clamped = torch.clamp(y[b], min=z[0], max=z[-1])

        # Calculate bj
        bj = (y_clamped - z[0]) / delta_z

        # Calculate floor and ceil of bj
        l = torch.floor(bj).long()
        u = torch.ceil(bj).long()

        # Accumulate m_l and m_u
        m_l = (u.float() - bj) * y_probs[b]
        m_u = (bj - l.float()) * y_probs[b]

        # Accumulate probabilities into the projected_probs tensor
        for j in range(n_atoms):
            if l[j] >= 0 and l[j] < n_atoms:
                projected_probs[b, l[j]] += m_l[j]
            if u[j] >= 0 and u[j] < n_atoms:
                projected_probs[b, u[j]] += m_u[j]

    return projected_probs