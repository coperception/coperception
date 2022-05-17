from coperception.models.det.backbone.Backbone import *
from coperception.models.det.base.DetModelBase import DetModelBase


class IntermediateModelBase(DetModelBase):
    """Abstract class. The parent class for all intermediate models.

    Attributes:
        u_encoder (nn.Module): The feature encoder.
        decoder (nn.Module): The feature decoder.
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
        super().__init__(config, layer, in_channels, kd_flag, num_agent=num_agent)
        self.u_encoder = LidarEncoder(in_channels, compress_level)
        self.decoder = LidarDecoder(height_feat_size=in_channels)
