from coperception.models.det.backbone.Backbone import *
from coperception.models.det.base.DetModelBase import DetModelBase


class NonIntermediateModelBase(DetModelBase):
    """Abstract class. The parent class for non-intermediate models.

    Attributes:
        stpn (nn.Module): Pass the features through encoder, then decoder.
    """

    def __init__(self, config, layer=3, in_channels=13, kd_flag=True, num_agent=5):
        super(NonIntermediateModelBase, self).__init__(
            config, layer, in_channels, kd_flag, num_agent
        )
        self.stpn = STPN_KD(height_feat_size=config.map_dims[2])
