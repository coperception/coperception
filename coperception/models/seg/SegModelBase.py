import torch
import torch.nn as nn
import torch.nn.functional as F


class SegModelBase(nn.Module):
    def __init__(
        self, n_channels, n_classes, bilinear=True, num_agent=5, compress_level=0
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.num_agent = num_agent

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        self.compress_level = compress_level
        if compress_level > 0:
            assert compress_level <= 9
            feat_map_channel_num = 512
            compress_channel_num = feat_map_channel_num // (2**compress_level)

            self.com_compresser = nn.Conv2d(
                feat_map_channel_num, compress_channel_num, kernel_size=1, stride=1
            )
            self.bn_compress = nn.BatchNorm2d(compress_channel_num)

            self.com_decompresser = nn.Conv2d(
                compress_channel_num, feat_map_channel_num, kernel_size=1, stride=1
            )
            self.bn_decompress = nn.BatchNorm2d(feat_map_channel_num)

    def build_feat_list(self, feat_maps, batch_size):
        feat_maps = torch.flip(feat_maps, (2,))

        tmp_feat_map = {}
        feat_list = []
        for i in range(self.num_agent):
            tmp_feat_map[i] = torch.unsqueeze(
                feat_maps[batch_size * i : batch_size * (i + 1)], 1
            )
            feat_list.append(tmp_feat_map[i])

        return feat_list

    @staticmethod
    def feature_transformation(b, j, agent_idx, local_com_mat, size, trans_matrices):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        nb_agent = torch.unsqueeze(local_com_mat[b, j], 0)

        tfm_ji = trans_matrices[b, j, agent_idx]
        M = (
            torch.hstack((tfm_ji[:2, :2], -tfm_ji[:2, 3:4])).float().unsqueeze(0)
        )  # [1,2,3]
        M = M.to(device)

        mask = torch.tensor([[[1, 1, 4 / 128], [1, 1, 4 / 128]]], device=M.device)

        M *= mask

        grid = F.affine_grid(M, size=torch.Size(size))
        warp_feat = F.grid_sample(nb_agent, grid).squeeze()
        return warp_feat

    def agents_to_batch(self, feats):
        feat_list = []
        for i in range(self.num_agent):
            feat_list.append(feats[:, i, :, :, :])
        feat_mat = torch.cat(feat_list, 0)

        feat_mat = torch.flip(feat_mat, (2,))

        return feat_mat


##################
#      Unet      # ref: https://github.com/milesial/Pytorch-UNet
##################
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
