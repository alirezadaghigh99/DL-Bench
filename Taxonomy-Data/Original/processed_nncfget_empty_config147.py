def get_empty_config(
    model_size=4, input_sample_sizes: Union[Tuple[List[int]], List[int]] = None, input_info: Dict = None
) -> NNCFConfig:
    if input_sample_sizes is None:
        input_sample_sizes = [1, 1, 4, 4]

    def _create_input_info():
        if isinstance(input_sample_sizes, tuple):
            return [{"sample_size": sizes} for sizes in input_sample_sizes]
        return [{"sample_size": input_sample_sizes}]

    config = NNCFConfig()
    config.update(
        {
            "model": "empty_config",
            "model_size": model_size,
            "input_info": input_info if input_info else _create_input_info(),
        }
    )
    return config