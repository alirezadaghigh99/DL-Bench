import torch
import torch.nn as nn
from torch import Tensor

class MKDDescriptor(nn.Module):
    def __init__(
        self,
        patch_size: int = 32,
        kernel_type: str = "concat",
        whitening: str = "pcawt",
        training_set: str = "liberty",
        output_dims: int = 128,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.kernel_type = kernel_type
        self.whitening = whitening
        self.training_set = training_set
        self.sigma = 1.4 * (patch_size / 64)
        self.smoothing = GaussianBlur2d((5, 5), (self.sigma, self.sigma), "replicate")
        self.gradients = MKDGradients()
        polar_s = "polar"
        cart_s = "cart"
        self.parametrizations = [polar_s, cart_s] if self.kernel_type == "concat" else [self.kernel_type]
        self.odims = 0
        relative_orientations = {polar_s: True, cart_s: False}
        self.feats = {}
        for parametrization in self.parametrizations:
            gradient_embedding = EmbedGradients(patch_size=patch_size, relative=relative_orientations[parametrization])
            spatial_encoding = ExplicitSpacialEncoding(
                kernel_type=parametrization, fmap_size=patch_size, in_dims=gradient_embedding.kernel.d
            )
            self.feats[parametrization] = nn.Sequential(gradient_embedding, spatial_encoding)
            self.odims += spatial_encoding.odims
        self.output_dims = min(output_dims, self.odims)
        if self.whitening is not None:
            whitening_models = torch.hub.load_state_dict_from_url(
                urls[self.kernel_type], map_location=map_location_to_cpu
            )
            whitening_model = whitening_models[training_set]
            self.whitening_layer = Whitening(
                whitening, whitening_model, in_dims=self.odims, output_dims=self.output_dims
            )
            self.odims = self.output_dims
        self.eval()

    def forward(self, patches: Tensor) -> Tensor:
        # Apply Gaussian smoothing
        smoothed_patches = self.smoothing(patches)
        
        # Compute gradients
        gradients = self.gradients(smoothed_patches)
        
        # Initialize a list to collect features from different parametrizations
        features = []
        
        # Process each parametrization
        for parametrization in self.parametrizations:
            # Apply the feature extraction pipeline for the current parametrization
            feat = self.feats[parametrization](gradients)
            features.append(feat)
        
        # Concatenate features from all parametrizations
        concatenated_features = torch.cat(features, dim=1)
        
        # Apply whitening if specified
        if self.whitening is not None:
            output = self.whitening_layer(concatenated_features)
        else:
            output = concatenated_features
        
        return output