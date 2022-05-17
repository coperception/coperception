from coperception.models.det.base import NonIntermediateModelBase


class FaFNet(NonIntermediateModelBase):
    """The model of early fusion. Used as lower-bound and upper-bound depending on the input features (fused or not).

    https://arxiv.org/pdf/2012.12395.pdf

    Args:
        config (object): The Config object.
        layer (int, optional): Collaborate on which layer. Defaults to 3.
        in_channels (int, optional): The input channels. Defaults to 13.
        kd_flag (bool, optional): Whether to use knowledge distillation (for DiscoNet to ues). Defaults to True.
        num_agent (int, optional): The number of agents (including RSU). Defaults to 5.
    """

    def __init__(
        self,
        config,
        layer=3,
        in_channels=13,
        kd_flag=True,
        num_agent=5,
        compress_level=0,
    ):
        super().__init__(config, layer, in_channels, kd_flag, num_agent, compress_level)

    def forward(self, bevs, maps=None, vis=None, batch_size=None):
        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)

        x_8, x_7, x_6, x_5, x_3, x_2 = self.stpn(bevs)
        x = x_8

        cls_preds, loc_preds, result = super().get_cls_loc_result(x)

        if self.kd_flag == 1:
            return result, x_8, x_7, x_6, x_5, x_3
        else:
            return result
