import torch
import torch.nn as nn
import torch.nn.functional as F
from coperception.models.det.base import IntermediateModelBase


class DiscoNet(IntermediateModelBase):
    """DiscoNet.

    https://github.com/ai4ce/DiscoNet

    Args:
        config (object): The config object.
        layer (int, optional): Collaborate on which layer. Defaults to 3.
        in_channels (int, optional): The input channels. Defaults to 13.
        kd_flag (bool, optional): Whether to use knowledge distillation. Defaults to True.
        num_agent (int, optional): The number of agents (including RSU). Defaults to 5.

    """

    def __init__(self, config, layer=3, in_channels=13, kd_flag=True, num_agent=5):
        super(DiscoNet, self).__init__(config, layer, in_channels, kd_flag, num_agent)
        if self.layer == 3:
            self.pixel_weighted_fusion = PixelWeightedFusionSoftmax(256)
        elif self.layer == 2:
            self.pixel_weighted_fusion = PixelWeightedFusionSoftmax(128)

    def forward(self, bevs, trans_matrices, num_agent_tensor, batch_size=1):
        """Forward pass.

        Args:
            bevs (tensor): BEV data
            trans_matrices (tensor): Matrix for transforming features among agents.
            num_agent_tensor (tensor): Number of agents to communicate for each agent.
            batch_size (int, optional): The batch size. Defaults to 1.

        Returns:
            result, all decoded layers, and fused feature maps if kd_flag is set.
            else return result and list of weights for each agent.
        """

        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)
        encoded_layers = self.u_encoder(bevs)
        device = bevs.device

        feat_maps, size = super().get_feature_maps_and_size(encoded_layers)

        feat_list = super().build_feature_list(batch_size, feat_maps)

        local_com_mat = super().build_local_communication_matrix(
            feat_list
        )  # [2 5 512 16 16] [batch, agent, channel, height, width]
        local_com_mat_update = super().build_local_communication_matrix(
            feat_list
        )  # to avoid the inplace operation

        save_agent_weight_list = list()

        for b in range(batch_size):
            num_agent = num_agent_tensor[b, 0]
            for i in range(num_agent):
                tg_agent = local_com_mat[b, i]
                all_warp = trans_matrices[b, i]  # transformation [2 5 5 4 4]

                self.neighbor_feat_list = list()
                self.neighbor_feat_list.append(tg_agent)

                if super().outage():
                    agent_wise_weight_feat = self.neighbor_feat_list[0]
                else:
                    super().build_neighbors_feature_list(
                        b, i, all_warp, num_agent, local_com_mat, device, size
                    )

                    # agent-wise weighted fusion
                    tmp_agent_weight_list = list()
                    sum_weight = 0
                    for k in range(num_agent):
                        cat_feat = torch.cat(
                            [tg_agent, self.neighbor_feat_list[k]], dim=0
                        )
                        cat_feat = cat_feat.unsqueeze(0)
                        agent_weight = torch.squeeze(
                            self.pixel_weighted_fusion(cat_feat)
                        )
                        tmp_agent_weight_list.append(torch.exp(agent_weight))
                        sum_weight = sum_weight + torch.exp(agent_weight)

                    agent_weight_list = list()
                    for k in range(num_agent):
                        agent_weight = torch.div(tmp_agent_weight_list[k], sum_weight)
                        agent_weight.expand([256, -1, -1])
                        agent_weight_list.append(agent_weight)

                    agent_wise_weight_feat = 0
                    for k in range(num_agent):
                        agent_wise_weight_feat = (
                            agent_wise_weight_feat
                            + agent_weight_list[k] * self.neighbor_feat_list[k]
                        )

                # feature update
                local_com_mat_update[b, i] = agent_wise_weight_feat

                save_agent_weight_list.append(agent_weight_list)

        # weighted feature maps is passed to decoder
        feat_fuse_mat = super().agents_to_batch(local_com_mat_update)

        decoded_layers = super().get_decoded_layers(
            encoded_layers, feat_fuse_mat, batch_size
        )
        x = decoded_layers[0]

        cls_preds, loc_preds, result = super().get_cls_loc_result(x)

        if self.kd_flag == 1:
            return (result, *decoded_layers, feat_fuse_mat)
        else:
            # return result
            return result, save_agent_weight_list


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
        # self.bn1_4 = nn.BatchNorm2d(1)

    def forward(self, x):
        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))
        x_1 = F.relu(self.bn1_3(self.conv1_3(x_1)))
        x_1 = F.relu(self.conv1_4(x_1))

        return x_1
