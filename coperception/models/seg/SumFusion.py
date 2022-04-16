import torch

from coperception.models.seg.FusionBase import FusionBase


class SumFusion(FusionBase):
    def __init__(self, n_channels, n_classes, num_agent=5):
        super().__init__(n_channels, n_classes, num_agent=num_agent)

    def fusion(self):
        return torch.sum(torch.stack(self.neighbor_feat_list), dim=0)

