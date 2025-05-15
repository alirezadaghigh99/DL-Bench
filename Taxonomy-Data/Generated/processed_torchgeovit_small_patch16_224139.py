import timm
from timm.models.vision_transformer import VisionTransformer

def vit_small_patch16_224(pretrained_weights=None, **kwargs):
    """
    Create a Vision Transformer (ViT) model with small patch size 16.

    Parameters:
    - pretrained_weights (str or None): Path to pre-trained weights. If None, no weights are loaded.
    - **kwargs: Additional arguments to pass to the VisionTransformer constructor.

    Returns:
    - model (VisionTransformer): The ViT small patch16 224 model.
    """
    # Create the ViT small patch16 224 model
    model = timm.create_model('vit_small_patch16_224', pretrained=False, **kwargs)

    # Load pre-trained weights if provided
    if pretrained_weights is not None:
        # Load the state dictionary from the provided weights
        state_dict = torch.load(pretrained_weights, map_location='cpu')
        
        # Adjust the input channels if necessary
        if 'patch_embed.proj.weight' in state_dict:
            in_channels = state_dict['patch_embed.proj.weight'].shape[1]
            if in_channels != model.patch_embed.proj.weight.shape[1]:
                # Adjust the input channels of the patch embedding layer
                model.patch_embed.proj = torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=model.patch_embed.proj.out_channels,
                    kernel_size=model.patch_embed.proj.kernel_size,
                    stride=model.patch_embed.proj.stride,
                    padding=model.patch_embed.proj.padding,
                    bias=model.patch_embed.proj.bias is not None
                )
        
        # Load the state dictionary into the model
        model.load_state_dict(state_dict, strict=False)

    return model

