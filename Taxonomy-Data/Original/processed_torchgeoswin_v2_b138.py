def swin_v2_b(
    weights: Swin_V2_B_Weights | None = None, *args: Any, **kwargs: Any
) -> SwinTransformer:
    """Swin Transformer v2 base model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2111.09883

    .. versionadded:: 0.6

    Args:
        weights: Pre-trained model weights to use.
        *args: Additional arguments to
            pass to :class:`torchvision.models.swin_transformer.SwinTransformer`.
        **kwargs: Additional keyword arguments to
            pass to :class:`torchvision.models.swin_transformer.SwinTransformer`.

    Returns:
        A Swin Transformer Base model.
    """
    model: SwinTransformer = torchvision.models.swin_v2_b(weights=None, *args, **kwargs)

    if weights:
        num_channels = weights.meta['in_chans']
        out_channels = model.features[0][0].out_channels
        # https://github.com/allenai/satlaspretrain_models/blob/main/satlaspretrain_models/models/backbones.py#L27
        model.features[0][0] = torch.nn.Conv2d(
            num_channels, out_channels, kernel_size=(4, 4), stride=(4, 4)
        )
        missing_keys, unexpected_keys = model.load_state_dict(
            weights.get_state_dict(progress=True), strict=False
        )
        assert set(missing_keys) <= set()
        assert not unexpected_keys

    return model