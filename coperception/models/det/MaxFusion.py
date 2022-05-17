import torch
from coperception.models.det.base import FusionBase


class MaxFusion(FusionBase):
    "Maximum fusion. Used as a lower-bound in the DiscoNet fusion."

    def __init__(
        self,
        config,
        layer=3,
        in_channels=13,
        kd_flag=True,
        num_agent=5,
        compress_level=0,
    ):
        super().__init__(config, layer, in_channels, kd_flag, num_agent, compress_level)

    def fusion(self):
        return torch.max(torch.stack(self.neighbor_feat_list), dim=0).values
