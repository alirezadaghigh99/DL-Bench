import GPy

def make_non_linear_kernels(base_kernel_class, n_fidelities, n_input_dims, ARD=False):
    """
    Constructs a list of structured multi-fidelity kernels using a specified base kernel class from GPy.

    Parameters:
    - base_kernel_class: The GPy kernel class to use.
    - n_fidelities: Number of fidelity levels.
    - n_input_dims: Number of input dimensions.
    - ARD: Boolean indicating whether to use Automatic Relevance Determination.

    Returns:
    - A list of kernels, one per fidelity level, starting from the lowest to the highest fidelity.
    """
    kernels = []

    # Create the kernel for the first fidelity level
    first_fidelity_kernel = base_kernel_class(input_dim=n_input_dims, ARD=ARD)
    kernels.append(first_fidelity_kernel)

    # Create kernels for subsequent fidelity levels
    for i in range(1, n_fidelities):
        # Kernel for the current fidelity
        current_fidelity_kernel = base_kernel_class(input_dim=n_input_dims, ARD=ARD)
        
        # Kernel for the previous fidelity
        previous_fidelity_kernel = kernels[i - 1]
        
        # Construct the kernel for the current fidelity level
        fidelity_kernel = (current_fidelity_kernel * previous_fidelity_kernel) + base_kernel_class(input_dim=n_input_dims, ARD=ARD)
        
        # Append the constructed kernel to the list
        kernels.append(fidelity_kernel)

    return kernels

