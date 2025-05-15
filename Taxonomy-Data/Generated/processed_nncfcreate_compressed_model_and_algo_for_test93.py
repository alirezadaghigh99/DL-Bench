from nncf import NNCFConfig
from nncf import create_compressed_model
from nncf import load_state
from nncf import register_default_init_args

def create_compressed_model_and_algo_for_test(model, nncf_config=None, dummy_forward_fn=None, wrap_inputs_fn=None, compression_state_dict=None):
    """
    Creates a compressed model and a compression algorithm controller.

    :param model: The original neural network model to be compressed.
    :param nncf_config: An optional NNCFConfig object for configuring the compression.
    :param dummy_forward_fn: A function that performs a dummy forward pass on the model.
    :param wrap_inputs_fn: A function that wraps the inputs for the model.
    :param compression_state_dict: A state dictionary for restoring a previously saved compression state.
    :return: A tuple containing the compressed model and the compression algorithm controller.
    """
    # Validate the NNCFConfig if provided
    if nncf_config is not None:
        if not isinstance(nncf_config, NNCFConfig):
            raise ValueError("The nncf_config must be an instance of NNCFConfig.")

    # Register default initialization arguments
    if dummy_forward_fn is not None and wrap_inputs_fn is not None:
        nncf_config = register_default_init_args(nncf_config, dummy_forward_fn, wrap_inputs_fn)

    # Create the compressed model and the compression algorithm controller
    compressed_model, compression_ctrl = create_compressed_model(model, nncf_config)

    # Load the compression state if provided
    if compression_state_dict is not None:
        load_state(compression_ctrl, compression_state_dict)

    return compressed_model, compression_ctrl