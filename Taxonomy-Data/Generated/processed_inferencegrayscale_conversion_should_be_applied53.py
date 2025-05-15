def grayscale_conversion_should_be_applied(preprocessing_config, disable_preproc_grayscale):
    # Check if 'GRAYSCALE' is a key in preprocessing_config
    if 'GRAYSCALE' not in preprocessing_config:
        return False
    
    # Check if DISABLE_PREPROC_GRAYSCALE is not True
    if preprocessing_config.get('DISABLE_PREPROC_GRAYSCALE', False):
        return False
    
    # Check if disable_preproc_grayscale is not True
    if disable_preproc_grayscale:
        return False
    
    # Check if preprocessing_config['GRAYSCALE']['ENABLED'] is True
    if preprocessing_config['GRAYSCALE'].get('ENABLED', False):
        return True
    
    return False