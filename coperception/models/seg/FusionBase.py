from coperception.models.seg.SegModelBase import SegModelBase
import torch


class FusionBase(SegModelBase):
    def __init__(self, n_channels, n_classes, num_agent=5):
        super().__init__(n_channels, n_classes, num_agent=num_agent)
        self.neighbor_feat_list = None
        self.tg_agent = None
        self.current_num_agent = None

    def fusion(self):
        raise NotImplementedError(
            "Please implement this method for specific fusion strategies"
        )

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
            for nb in range(self.current_num_agent):
                agent_feat_list.append(local_com_mat[b, nb])

            for i in range(self.current_num_agent):
                self.tg_agent = local_com_mat[b, i]
                all_warp = trans_matrices[b, i]

                self.neighbor_feat_list = list()
                self.neighbor_feat_list.append(self.tg_agent)

                for j in range(self.current_num_agent):
                    if j != i:
                        self.neighbor_feat_list.append(
                            super().feature_transformation(
                                b, j, local_com_mat, all_warp, device, size
                            )
                        )

                local_com_mat_update[b, i] = self.fusion()

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
