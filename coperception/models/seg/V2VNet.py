import torch

import coperception.utils.convolutional_rnn as convrnn
from coperception.models.seg.SegModelBase import SegModelBase


class V2VNet(SegModelBase):
    def __init__(self, n_channels, n_classes, num_agent=5):
        super().__init__(n_channels, n_classes, num_agent=num_agent)
        self.layer_channel = 512
        self.gnn_iter_num = 1
        self.convgru = convrnn.Conv2dGRU(in_channels=self.layer_channel * 2,
                                         out_channels=self.layer_channel,
                                         kernel_size=3,
                                         num_layers=1,
                                         bidirectional=False,
                                         dilation=1,
                                         stride=1)

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
            num_agent = self.num_agent

            agent_feat_list = list()
            for nb in range(self.num_agent):
                agent_feat_list.append(local_com_mat[b, nb])

            for _ in range(self.gnn_iter_num):
                updated_feats_list = list()

                for i in range(num_agent):
                    tg_agent = local_com_mat[b, i]
                    all_warp = trans_matrices[b, i]

                    neighbor_feat_list = list()
                    neighbor_feat_list.append(tg_agent)

                    for j in range(num_agent):
                        if j != i:
                            neighbor_feat_list.append(super().feature_transformation(b, j, local_com_mat,
                                                                                     all_warp, device, size))

                    mean_feat = torch.mean(torch.stack(neighbor_feat_list), dim=0)
                    cat_feat = torch.cat([agent_feat_list[i], mean_feat], dim=0)
                    cat_feat = cat_feat.unsqueeze(0).unsqueeze(0)
                    updated_feat, _ = self.convgru(cat_feat, None)
                    updated_feat = torch.squeeze(torch.squeeze(updated_feat, 0), 0)
                    updated_feats_list.append(updated_feat)
                agent_feat_list = updated_feats_list
            for k in range(num_agent):
                local_com_mat_update[b, k] = agent_feat_list[k]

        feat_list = []
        for i in range(self.num_agent):
            feat_list.append(local_com_mat_update[:, i, :, :, :])
        feat_mat = torch.cat(feat_list, 0)

        x5 = self.down4(feat_mat)
        x = self.up1(x5, feat_mat)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
