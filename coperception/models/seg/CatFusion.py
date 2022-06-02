import torch
import torch.nn as nn
import torch.nn.functional as F

from coperception.models.seg.FusionBase import FusionBase


class CatFusion(FusionBase):
    def __init__(self, n_channels, n_classes, num_agent, compress_level, only_v2i):
        super().__init__(
            n_channels, n_classes, num_agent=num_agent, compress_level=compress_level, only_v2i=only_v2i
        )
        self.modulation_layer_3 = ModulationLayer3()

    def fusion(self):
        mean_feat = torch.mean(torch.stack(self.neighbor_feat_list), dim=0)  # [c, h, w]
        cat_feat = torch.cat([self.tg_agent, mean_feat], dim=0)
        cat_feat = cat_feat.unsqueeze(0)  # [1, 1, c, h, w]
        return self.modulation_layer_3(cat_feat)


# FIXME: Change size
class ModulationLayer3(nn.Module):
    def __init__(self):
        super(ModulationLayer3, self).__init__()

        self.conv1_1 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        self.bn1_1 = nn.BatchNorm2d(512)

    def forward(self, x):
        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))

        return x_1
