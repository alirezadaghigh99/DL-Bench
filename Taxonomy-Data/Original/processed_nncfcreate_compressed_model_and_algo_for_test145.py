def create_compressed_model_and_algo_for_test(model, config, compression_state=None, force_no_init=False):
    assert isinstance(config, NNCFConfig)
    tf.keras.backend.clear_session()
    if force_no_init:
        compression_state = {BaseCompressionAlgorithmController.BUILDER_STATE: {}}
    algo, model = create_compressed_model(model, config, compression_state)
    return model, algo