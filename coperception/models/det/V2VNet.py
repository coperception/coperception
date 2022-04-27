import torch
import coperception.utils.convolutional_rnn as convrnn
from coperception.models.det.base import IntermediateModelBase
import torch.nn.functional as F


class V2VNet(IntermediateModelBase):
    """V2V Net

    https://arxiv.org/abs/2008.07519

    """

    def __init__(
        self, config, gnn_iter_times, layer, layer_channel, in_channels=13, num_agent=5
    ):
        super(V2VNet, self).__init__(config, layer, in_channels, num_agent=num_agent)

        self.layer_channel = layer_channel
        self.gnn_iter_num = gnn_iter_times
        self.convgru = convrnn.Conv2dGRU(
            in_channels=self.layer_channel * 2,
            out_channels=self.layer_channel,
            kernel_size=3,
            num_layers=1,
            bidirectional=False,
            dilation=1,
            stride=1,
        )

    def forward(self, bevs, trans_matrices, num_agent_tensor, batch_size=1):
        # trans_matrices [batch 5 5 4 4]
        # num_agent_tensor, shape: [batch, num_agent]; how many non-empty agent in this scene

        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)
        encoded_layers = self.u_encoder(bevs, requires_compression=False)
        device = bevs.device

        feat_maps, size = super().get_feature_maps_and_size(encoded_layers)

        for b in range(batch_size):
            num_agent = num_agent_tensor[b, 0]

            for _ in range(self.gnn_iter_num):

                # compress & decompress
                feat_maps = F.relu(
                    self.u_encoder.bn_compress(self.u_encoder.com_compresser(feat_maps))
                )
                feat_maps = F.relu(
                    self.u_encoder.bn_decompress(
                        self.u_encoder.com_decompresser(feat_maps)
                    )
                )

                # get feat maps for each agent [10 512 16 16] -> [2 5 512 16 16]
                feat_list = super().build_feature_list(batch_size, feat_maps)

                local_com_mat = super().build_local_communication_matrix(
                    feat_list
                )  # [2 5 512 16 16] [batch, agent, channel, height, width]
                local_com_mat_update = super().build_local_communication_matrix(
                    feat_list
                )  # to avoid the inplace operation

                agent_feat_list = list()
                for nb in range(self.agent_num):
                    agent_feat_list.append(local_com_mat[b, nb])

                updated_feats_list = []

                for i in range(num_agent):
                    self.neighbor_feat_list = []
                    all_warp = trans_matrices[b, i]  # transformation [2 5 5 4 4]

                    if super().outage():
                        updated_feats_list.append(agent_feat_list[i])

                    else:
                        super().build_neighbors_feature_list(
                            b, i, all_warp, num_agent, local_com_mat, device, size
                        )

                        mean_feat = torch.mean(
                            torch.stack(self.neighbor_feat_list), dim=0
                        )  # [c, h, w]
                        cat_feat = torch.cat([agent_feat_list[i], mean_feat], dim=0)
                        cat_feat = cat_feat.unsqueeze(0).unsqueeze(0)  # [1, 1, c, h, w]
                        updated_feat, _ = self.convgru(cat_feat, None)
                        updated_feat = torch.squeeze(
                            torch.squeeze(updated_feat, 0), 0
                        )  # [c, h, w]
                        updated_feats_list.append(updated_feat)

                agent_feat_list = updated_feats_list

                for k in range(num_agent):
                    local_com_mat_update[b, k] = agent_feat_list[k]
                feat_maps = super().agents_to_batch(local_com_mat_update)

        decoded_layers = super().get_decoded_layers(
            encoded_layers, feat_maps, batch_size
        )
        x = decoded_layers[0]

        cls_pred, loc_preds, result = super().get_cls_loc_result(x)
        return result
