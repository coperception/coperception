import torch

from coperception.models.seg.FusionBase import FusionBase


class MaxFusion(FusionBase):
    def __init__(self, n_channels, n_classes, num_agent=5, compress_level=0):
        super().__init__(
            n_channels, n_classes, num_agent=num_agent, compress_level=compress_level
        )

    def fusion(self):
        return torch.max(torch.stack(self.neighbor_feat_list), dim=0).values
