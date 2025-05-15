import timm

def swin_v2_b(pretrained=False, **kwargs):
    """
    Returns a Swin Transformer v2 base model.

    Parameters:
    - pretrained (bool): If True, loads pre-trained weights.
    - **kwargs: Additional arguments to pass to the model constructor.

    Returns:
    - model: A Swin Transformer v2 base model.
    """
    model_name = 'swinv2_base_window12_192_22k'  # This is a common Swin v2 base model
    model = timm.create_model(model_name, pretrained=pretrained, **kwargs)
    return model

