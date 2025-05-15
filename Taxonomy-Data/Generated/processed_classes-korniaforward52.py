import torch
from torch import nn, Tensor
from typing import Optional, Any, Dict

class LoFTR(nn.Module):
    def __init__(self, pretrained: Optional[str] = "outdoor", config: Dict[str, Any] = default_cfg) -> None:
        super().__init__()
        self.config = config
        self.backbone = build_backbone(config)
        self.pos_encoding = PositionEncodingSine(config["coarse"]["d_model"])
        self.loftr_coarse = LocalFeatureTransformer(config["coarse"])
        self.coarse_matching = CoarseMatching(config["match_coarse"])
        self.fine_preprocess = FinePreprocess(config)
        self.loftr_fine = LocalFeatureTransformer(config["fine"])
        self.fine_matching = FineMatching()
        self.pretrained = pretrained
        if pretrained is not None:
            pretrained_dict = torch.hub.load_state_dict_from_url(urls[pretrained], map_location=map_location_to_cpu)
            self.load_state_dict(pretrained_dict["state_dict"])
        self.eval()

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        image0, image1 = data['image0'], data['image1']
        mask0, mask1 = data.get('mask0', None), data.get('mask1', None)

        # Step 1: Extract features using the backbone
        feat0_c, feat1_c = self.backbone(image0), self.backbone(image1)

        # Step 2: Apply positional encoding
        feat0_c = self.pos_encoding(feat0_c)
        feat1_c = self.pos_encoding(feat1_c)

        # Step 3: Coarse-level matching
        coarse_matches = self.loftr_coarse(feat0_c, feat1_c)

        # Step 4: Handle masks if provided
        if mask0 is not None and mask1 is not None:
            mask0_resized = nn.functional.interpolate(mask0, size=feat0_c.shape[-2:], mode='bilinear', align_corners=False)
            mask1_resized = nn.functional.interpolate(mask1, size=feat1_c.shape[-2:], mode='bilinear', align_corners=False)
            coarse_matches = self.coarse_matching(coarse_matches, mask0_resized, mask1_resized)
        else:
            coarse_matches = self.coarse_matching(coarse_matches)

        # Step 5: Fine-level matching
        fine_matches = self.loftr_fine(coarse_matches)

        # Step 6: Compile results
        results = {
            'keypoints0': fine_matches['keypoints0'],
            'keypoints1': fine_matches['keypoints1'],
            'confidence': fine_matches['confidence'],
            'batch_indexes': fine_matches['batch_indexes']
        }

        return results