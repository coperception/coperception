import torch
import torch.nn as nn
import torch.nn.functional as F

from coperception.models.seg.FusionBase import FusionBase


class DiscoNet(FusionBase):
    def __init__(self, n_channels, n_classes, num_agent, kd_flag=False):
        super().__init__(n_channels, n_classes, num_agent)
        self.pixel_weighted_fusion = PixelWeightedFusionSoftmax(512)
        self.kd_flag = kd_flag

    def forward(self, x, trans_matrices, num_agent_tensor):
        device = x.device
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)  # b 512 32 32
        size = (1, 512, 32, 32)

        batch_size = x.size(0) // self.num_agent
        feat_map, feat_list = super().build_feat_map_and_feat_list(x4, batch_size)

        local_com_mat = torch.cat(tuple(feat_list), 1)
        local_com_mat_update = torch.cat(tuple(feat_list), 1)

        for b in range(batch_size):
            self.current_num_agent = num_agent_tensor[b, 0]

            agent_feat_list = list()
            for nb in range(self.num_agent):
                agent_feat_list.append(local_com_mat[b, nb])

            for i in range(self.num_agent):
                self.tg_agent = local_com_mat[b, i]
                all_warp = trans_matrices[b, i]

                self.neighbor_feat_list = list()
                self.neighbor_feat_list.append(self.tg_agent)

                for j in range(self.num_agent):
                    if j != i:
                        self.neighbor_feat_list.append(super().feature_transformation(b, j, local_com_mat,
                                                                                      all_warp, device, size))

                local_com_mat_update[b, i] = self.fusion()

        feat_list = []
        for i in range(self.num_agent):
            feat_list.append(local_com_mat_update[:, i, :, :, :])
        feat_mat = torch.cat(feat_list, 0)

        x5 = self.down4(feat_mat)
        x6 = self.up1(x5, feat_mat)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        logits = self.outc(x9)

        if self.kd_flag:
            return logits, x9, x8, x7, x6, x5, feat_mat
        else:
            return logits

    def fusion(self):
        tmp_agent_weight_list = list()
        sum_weight = 0
        for k in range(self.num_agent):
            cat_feat = torch.cat([self.tg_agent, self.neighbor_feat_list[k]], dim=0)
            cat_feat = cat_feat.unsqueeze(0)
            agent_weight = torch.squeeze(self.pixel_weighted_fusion(cat_feat))
            tmp_agent_weight_list.append(torch.exp(agent_weight))
            sum_weight = sum_weight + torch.exp(agent_weight)

        agent_weight_list = list()
        for k in range(self.num_agent):
            agent_weight = torch.div(tmp_agent_weight_list[k], sum_weight)
            agent_weight.expand([256, -1, -1])
            agent_weight_list.append(agent_weight)

        agent_wise_weight_feat = 0
        for k in range(self.num_agent):
            agent_wise_weight_feat = agent_wise_weight_feat + agent_weight_list[k] * \
                                     self.neighbor_feat_list[k]

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
