from coperception.models.seg.SegModelBase import SegModelBase
import torch.nn.functional as F


class UNet(SegModelBase):
    def __init__(
        self,
        n_channels,
        n_classes,
        bilinear=True,
        num_agent=5,
        kd_flag=False,
        compress_level=0,
    ):
        super().__init__(
            n_channels,
            n_classes,
            bilinear,
            num_agent=num_agent,
            compress_level=compress_level,
        )
        self.kd_flag = kd_flag

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        if self.compress_level > 0:
            x4 = F.relu(self.bn_compress(self.com_compresser(x4)))
            x4 = F.relu(self.bn_decompress(self.com_decompresser(x4)))

        x5 = self.down4(x4)
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        logits = self.outc(x9)

        if self.kd_flag:
            return logits, x9, x8, x7, x6, x5, x4
        else:
            return logits
