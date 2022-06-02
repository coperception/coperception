import torch
import torch.nn as nn
import torch.nn.functional as F

from coperception.models.seg.FusionBase import FusionBase


class AgentWiseWeightedFusion(FusionBase):
    def __init__(self, n_channels, n_classes, num_agent=5, compress_level=0, only_v2i=False):
        super().__init__(
            n_channels, n_classes, num_agent=num_agent, compress_level=compress_level, only_v2i=only_v2i
        )
        self.agent_weighted_fusion = AgentWeightedFusion()

    def fusion(self):
        agent_weight_list = list()
        for k in range(self.com_num_agent):
            cat_feat = torch.cat([self.tg_agent, self.neighbor_feat_list[k]], dim=0)
            cat_feat = cat_feat.unsqueeze(0)
            agent_weight = self.agent_weighted_fusion(cat_feat)
            agent_weight_list.append(agent_weight)

        soft_agent_weight_list = torch.squeeze(
            F.softmax(torch.tensor(agent_weight_list).unsqueeze(0), dim=1)
        )

        agent_wise_weight_feat = 0
        for k in range(self.com_num_agent):
            agent_wise_weight_feat = (
                agent_wise_weight_feat
                + soft_agent_weight_list[k] * self.neighbor_feat_list[k]
            )

        return agent_wise_weight_feat


# FIXME: Change size
class AgentWeightedFusion(nn.Module):
    def __init__(self):
        super(AgentWeightedFusion, self).__init__()

        # self.conv1_1 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        # self.bn1_1 = nn.BatchNorm2d(128)
        #
        # self.conv1_2 = nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0)
        # self.bn1_2 = nn.BatchNorm2d(32)
        #
        # self.conv1_3 = nn.Conv2d(32, 8, kernel_size=1, stride=1, padding=0)
        # self.bn1_3 = nn.BatchNorm2d(8)
        #
        # self.conv1_4 = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0)
        #
        # # self.conv1_1 = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)
        # # self.bn1_1 = nn.BatchNorm2d(1)
        # self.conv1_5 = nn.Conv2d(1, 1, kernel_size=32, stride=1, padding=0)
        # # # self.bn1_2 = nn.BatchNorm2d(1)

        self.conv1_1 = nn.Conv2d(1024, 128, kernel_size=1, stride=1, padding=0)
        self.bn1_1 = nn.BatchNorm2d(128)

        self.conv1_2 = nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0)
        self.bn1_2 = nn.BatchNorm2d(32)

        self.conv1_3 = nn.Conv2d(32, 8, kernel_size=1, stride=1, padding=0)
        self.bn1_3 = nn.BatchNorm2d(8)

        self.conv1_4 = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0)

        self.conv1_5 = nn.Conv2d(1, 1, kernel_size=32, stride=1, padding=0)

    def forward(self, x):
        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))
        x_1 = F.relu(self.bn1_3(self.conv1_3(x_1)))
        x_1 = F.relu(self.conv1_4(x_1))
        x_1 = F.relu(self.conv1_5(x_1))

        return x_1
