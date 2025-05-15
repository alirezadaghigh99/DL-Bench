# Mock version of NNCFConfig for demonstration purposes
class NNCFConfig:
    def __init__(self):
        self.config = {}

    def update(self, new_config):
        self.config.update(new_config)

    def __repr__(self):
        return f"NNCFConfig({self.config})"

def get_empty_config(model_size=4, input_sample_sizes=None, input_info=None):
    if input_sample_sizes is None:
        input_sample_sizes = [1, 1, 4, 4]

    def _create_input_info():
        return [{"sample_size": size} for size in input_sample_sizes]

    nncf_config = NNCFConfig()
    nncf_config.update({
        "model": "empty_config",
        "model_size": model_size,
        "input_info": input_info if input_info is not None else _create_input_info()
    })

    return nncf_config

