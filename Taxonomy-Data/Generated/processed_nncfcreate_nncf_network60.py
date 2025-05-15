import torch
from nncf import NNCFNetwork
from nncf.dynamic_graph.context import TracingContext
from nncf.dynamic_graph.graph_builder import create_input_infos
from nncf.nncf_network import InputInfo

def create_nncf_network(model, config, dummy_forward_fn=None, wrap_inputs_fn=None, wrap_outputs_fn=None):
    """
    Prepares a PyTorch model for compression using NNCF.

    :param model: The original model. Should have its parameters already loaded from a checkpoint or another source.
    :param config: A configuration object used to determine the exact compression modifications to be applied to the model.
    :param dummy_forward_fn: If supplied, will be used instead of a *forward* function call to build the internal graph representation via tracing.
    :param wrap_inputs_fn: If supplied, will be used on the module's input arguments during a regular, non-dummy forward call.
    :param wrap_outputs_fn: Same as `wrap_inputs_fn`, but for marking model outputs.

    :return: A model wrapped by NNCFNetwork, which is ready for adding compression.
    """
    # Create input information from the config
    input_infos = create_input_infos(config)

    # Create a tracing context
    tracing_context = TracingContext()

    # Wrap the model with NNCFNetwork
    nncf_network = NNCFNetwork(
        model,
        input_infos=input_infos,
        dummy_forward_fn=dummy_forward_fn,
        wrap_inputs_fn=wrap_inputs_fn,
        wrap_outputs_fn=wrap_outputs_fn,
        tracing_context=tracing_context
    )

    return nncf_network