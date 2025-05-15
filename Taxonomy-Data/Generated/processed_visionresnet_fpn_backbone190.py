import torch
from torch import nn
from torchvision.models import resnet
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.faster_rcnn import LastLevelMaxPool
from torchvision.ops import misc as misc_nn_ops
from typing import Optional, Callable, List
from torchvision.models._utils import IntermediateLayerGetter

def resnet_fpn_backbone(
    *,
    backbone_name: str,
    weights: Optional[resnet.WeightsEnum],
    norm_layer: Callable[..., nn.Module] = misc_nn_ops.FrozenBatchNorm2d,
    trainable_layers: int = 3,
    returned_layers: Optional[List[int]] = None,
    extra_blocks: Optional[BackboneWithFPN.ExtraFPNBlock] = None,
) -> BackboneWithFPN:
    """
    Constructs a specified ResNet backbone with FPN on top. Freezes the specified number of layers in the backbone.
    """
    # Validate the backbone name
    resnet_backbones = {
        'resnet18': resnet.resnet18,
        'resnet34': resnet.resnet34,
        'resnet50': resnet.resnet50,
        'resnet101': resnet.resnet101,
        'resnet152': resnet.resnet152,
        'resnext50_32x4d': resnet.resnext50_32x4d,
        'resnext101_32x8d': resnet.resnext101_32x8d,
        'wide_resnet50_2': resnet.wide_resnet50_2,
        'wide_resnet101_2': resnet.wide_resnet101_2,
    }
    
    if backbone_name not in resnet_backbones:
        raise ValueError(f"Unsupported backbone name {backbone_name}. Supported names are: {list(resnet_backbones.keys())}")

    # Load the ResNet model
    backbone = resnet_backbones[backbone_name](weights=weights, norm_layer=norm_layer)

    # Freeze layers
    for name, parameter in backbone.named_parameters():
        if not any([name.startswith(f"layer{i}") for i in range(5 - trainable_layers, 5)]):
            parameter.requires_grad_(False)

    # If returned_layers is not specified, return all layers
    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    assert all(1 <= layer <= 4 for layer in returned_layers), "Each layer must be in [1, 4]."

    # Create the IntermediateLayerGetter
    return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}
    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256

    # Create the FPN
    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()
    
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)