import torch
from coperception.models.det.base import FusionBase


class SumFusion(FusionBase):
    """Sum fusion. Used as a lower-bound in the DiscoNet fusion."""

    def __init__(
        self,
        config,
        layer=3,
        in_channels=13,
        kd_flag=True,
        num_agent=5,
        compress_level=0,
        only_v2i=False,
    ):
        super().__init__(config, layer, in_channels, kd_flag, num_agent, compress_level, only_v2i)

    def fusion(self):
        return torch.sum(torch.stack(self.neighbor_feat_list), dim=0)
