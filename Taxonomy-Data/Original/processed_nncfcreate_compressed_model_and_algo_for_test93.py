def create_compressed_model_and_algo_for_test(
    model: Module,
    config: NNCFConfig = None,
    dummy_forward_fn: Callable[[Module], Any] = None,
    wrap_inputs_fn: Callable[[Tuple, Dict], Tuple[Tuple, Dict]] = None,
    compression_state: Dict[str, Any] = None,
) -> Tuple[NNCFNetwork, PTCompressionAlgorithmController]:
    if config is not None:
        assert isinstance(config, NNCFConfig)
        NNCFConfig.validate(config)
    algo, model = create_compressed_model(
        model,
        config,
        dump_graphs=False,
        dummy_forward_fn=dummy_forward_fn,
        wrap_inputs_fn=wrap_inputs_fn,
        compression_state=compression_state,
    )
    return model, algo