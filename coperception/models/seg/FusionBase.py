from coperception.models.seg.SegModelBase import SegModelBase
import torch


class FusionBase(SegModelBase):
    def __init__(self, n_channels, n_classes, num_agent=5, kd_flag=False):
        super().__init__(n_channels, n_classes, num_agent=num_agent)
        self.neighbor_feat_list = None
        self.tg_agent = None
        self.current_num_agent = None
        self.kd_flag = kd_flag

    def fusion(self):
        raise NotImplementedError(
            "Please implement this method for specific fusion strategies"
        )

    def forward(self, x, trans_matrices, num_agent_tensor):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)  # b 512 32 32
        size = (1, 512, 32, 32)

        batch_size = x.size(0) // self.num_agent
        feat_list = super().build_feat_list(x4, batch_size)

        local_com_mat = torch.cat(tuple(feat_list), 1)
        local_com_mat_update = torch.cat(tuple(feat_list), 1)

        for b in range(batch_size):
            self.com_num_agent = num_agent_tensor[b, 0]

            agent_feat_list = list()
            for nb in range(self.com_num_agent):
                agent_feat_list.append(local_com_mat[b, nb])

            for i in range(self.com_num_agent):
                self.tg_agent = local_com_mat[b, i]

                self.neighbor_feat_list = list()
                self.neighbor_feat_list.append(self.tg_agent)

                for j in range(self.com_num_agent):
                    if j != i:
                        self.neighbor_feat_list.append(
                            super().feature_transformation(
                                b,
                                j,
                                i,
                                local_com_mat,
                                size,
                                trans_matrices,
                            )
                        )

                local_com_mat_update[b, i] = self.fusion()

        feat_mat = super().agents_to_batch(local_com_mat_update)

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
