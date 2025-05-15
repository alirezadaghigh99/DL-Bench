from nncf import NNCFConfig
from nncf.tensorflow import create_compressed_model
import tensorflow as tf

def create_compressed_model_and_algo_for_test(model, config, compression_state=None, force_no_init=False):
    # Assert that config is an instance of NNCFConfig
    assert isinstance(config, NNCFConfig), "config must be an instance of NNCFConfig"
    
    # Clear the TensorFlow backend session
    tf.keras.backend.clear_session()
    
    # If force_no_init is True, set compression_state to an empty dictionary
    if force_no_init:
        compression_state = {}
    
    # Create a compressed model and algorithm
    compressed_model, compression_algorithm = create_compressed_model(model, config, compression_state=compression_state)
    
    # Return the compressed model and algorithm
    return compressed_model, compression_algorithm