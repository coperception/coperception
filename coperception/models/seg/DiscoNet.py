import torch
import torch.nn as nn
import torch.nn.functional as F

from coperception.models.seg.FusionBase import FusionBase


class DiscoNet(FusionBase):
    def __init__(
        self, n_channels, n_classes, num_agent, kd_flag=True, compress_level=0, only_v2i=False
    ):
        super().__init__(
            n_channels,
            n_classes,
            num_agent,
            kd_flag=kd_flag,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
        self.pixel_weighted_fusion = PixelWeightedFusionSoftmax(512)
 
    def fusion(self):
        tmp_agent_weight_list = list()
        sum_weight = 0
        nb_len = len(self.neighbor_feat_list)
        for k in range(nb_len):
            cat_feat = torch.cat([self.tg_agent, self.neighbor_feat_list[k]], dim=0)
            cat_feat = cat_feat.unsqueeze(0)
            agent_weight = torch.squeeze(self.pixel_weighted_fusion(cat_feat))
            tmp_agent_weight_list.append(torch.exp(agent_weight))
            sum_weight = sum_weight + torch.exp(agent_weight)

        agent_weight_list = list()
        for k in range(nb_len):
            agent_weight = torch.div(tmp_agent_weight_list[k], sum_weight)
            agent_weight.expand([256, -1, -1])
            agent_weight_list.append(agent_weight)

        agent_wise_weight_feat = 0
        for k in range(nb_len):
            agent_wise_weight_feat = (
                agent_wise_weight_feat
                + agent_weight_list[k] * self.neighbor_feat_list[k]
            )

        return agent_wise_weight_feat


class PixelWeightedFusionSoftmax(nn.Module):
    def __init__(self, channel):
        super(PixelWeightedFusionSoftmax, self).__init__()

        self.conv1_1 = nn.Conv2d(channel * 2, 128, kernel_size=1, stride=1, padding=0)
        self.bn1_1 = nn.BatchNorm2d(128)

        self.conv1_2 = nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0)
        self.bn1_2 = nn.BatchNorm2d(32)

        self.conv1_3 = nn.Conv2d(32, 8, kernel_size=1, stride=1, padding=0)
        self.bn1_3 = nn.BatchNorm2d(8)

        self.conv1_4 = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))
        x_1 = F.relu(self.bn1_3(self.conv1_3(x_1)))
        x_1 = F.relu(self.conv1_4(x_1))

        return x_1
