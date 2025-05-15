def create_nncf_network(
    model: torch.nn.Module,
    config: NNCFConfig,
    dummy_forward_fn: Callable[[Module], Any] = None,
    wrap_inputs_fn: WrapInputsFnType = None,
    wrap_outputs_fn: WrapOutputsFnType = None,
) -> NNCFNetwork:
    """
    The main function used to produce a model ready for adding compression from an original PyTorch
    model and a configuration object.

    :param model: The original model. Should have its parameters already loaded from a checkpoint or another
        source.
    :param config: A configuration object used to determine the exact compression modifications to be applied
        to the model
    :param dummy_forward_fn: if supplied, will be used instead of a *forward* function call to build
        the internal graph representation via tracing. Specifying this is useful when the original training pipeline
        has special formats of data loader output or has additional *forward* arguments other than input tensors.
        Otherwise, the *forward* call of the model during graph tracing will be made with mock tensors according
        to the shape specified in the config object. The dummy_forward_fn code MUST contain calls to
        nncf.nncf_model_input
        functions made with each compressed model input tensor in the underlying model's args/kwargs tuple, and these
        calls should be exactly the same as in the wrap_inputs_fn function code (see below); if dummy_forward_fn is
        specified, then wrap_inputs_fn also must be specified.
    :param wrap_inputs_fn: if supplied, will be used on the module's input arguments during a regular, non-dummy
        forward call before passing the inputs to the underlying compressed model. This is required if the model's input
        tensors that are important for compression are not supplied as arguments to the model's forward call directly,
        but instead are located in a container (such as list), and the model receives the container as an argument.
        wrap_inputs_fn should take as input two arguments - the tuple of positional arguments to the underlying
        model's forward call, and a dict of keyword arguments to the same. The function should wrap each tensor among
        the supplied model's args and kwargs that is important for compression (e.g. quantization) with an
        nncf.nncf_model_input function, which is a no-operation function and marks the tensors as inputs to be traced
        by NNCF in the internal graph representation. Output is the tuple of (args, kwargs), where args and kwargs are
        the same as were supplied in input, but each tensor in the original input. Must be specified if
        dummy_forward_fn is specified.
    :param wrap_outputs_fn: Same as `wrap_inputs_fn`, but for marking model outputs with

    :return: A model wrapped by NNCFNetwork, which is ready for adding compression."""

    if dummy_forward_fn is not None and wrap_inputs_fn is None:
        raise ValueError(
            "A custom dummy forward function was specified, but the corresponding input wrapping function "
            "was not. In case a custom dummy forward function is specified for purposes of NNCF graph "
            "building, then the wrap_inputs_fn parameter MUST also be specified and be consistent with "
            "the input wrapping done in dummy_forward_fn."
        )

    # Preserve `.training`/`.requires_grad` state since we will be building NNCFNetwork in `.eval` mode
    with training_mode_switcher(model, is_training=False):
        # Compress model that will be deployed for the inference on target device. No need to compress parts of the
        # model that are used on training stage only (e.g. AuxLogits of Inception-v3 model) or unused modules with
        # weights. As a consequence, no need to care about spoiling BN statistics, as they're disabled in eval mode.

        input_info = get_input_info_from_config(config)
        scopes_without_shape_matching = config.get("scopes_without_shape_matching", [])
        ignored_scopes = config.get("ignored_scopes")
        target_scopes = config.get("target_scopes")

        nncf_network = NNCFNetwork(
            model,
            input_info=input_info,
            dummy_forward_fn=dummy_forward_fn,
            wrap_inputs_fn=wrap_inputs_fn,
            wrap_outputs_fn=wrap_outputs_fn,
            ignored_scopes=ignored_scopes,
            target_scopes=target_scopes,
            scopes_without_shape_matching=scopes_without_shape_matching,
        )

        nncf_network.nncf.get_tracing_context().disable_trace_dynamic_graph()

    synchronize_all_processes_in_distributed_mode()
    return nncf_network