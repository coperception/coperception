import torch
import torch.nn as nn
import torch.nn.functional as F
from coperception.models.det.base import FusionBase


class CatFusion(FusionBase):
    """Concatenate fusion. Used as a lower-bound in the DisoNet paper."""

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
        self._modulation_layer_3 = ModulationLayer3()

    def fusion(self):
        mean_feat = torch.mean(torch.stack(self.neighbor_feat_list), dim=0)  # [c, h, w]
        cat_feat = torch.cat([self.tg_agent, mean_feat], dim=0)
        cat_feat = cat_feat.unsqueeze(0)  # [1, 1, c, h, w]
        return self._modulation_layer_3(cat_feat)


class ModulationLayer3(nn.Module):
    def __init__(self):
        super(ModulationLayer3, self).__init__()

        self._conv1_1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self._bn1_1 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x_1 = F.relu(self._bn1_1(self._conv1_1(x)))

        return x_1
