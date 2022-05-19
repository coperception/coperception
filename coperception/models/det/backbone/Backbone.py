import torch.nn.functional as F
import torch.nn as nn
import torch


class Backbone(nn.Module):
    """The backbone class that contains encode and decode function"""

    def __init__(self, height_feat_size, compress_level=0):
        super().__init__()
        self.conv_pre_1 = nn.Conv2d(
            height_feat_size, 32, kernel_size=3, stride=1, padding=1
        )
        self.conv_pre_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1 = nn.BatchNorm2d(32)
        self.bn_pre_2 = nn.BatchNorm2d(32)

        self.conv3d_1 = Conv3D(
            64, 64, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0)
        )
        self.conv3d_2 = Conv3D(
            128, 128, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0)
        )

        self.conv1_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512 + 256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv6_1 = nn.Conv2d(256 + 128, 128, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv7_1 = nn.Conv2d(128 + 64, 64, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv8_1 = nn.Conv2d(64 + 32, 32, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(64)

        self.bn2_1 = nn.BatchNorm2d(128)
        self.bn2_2 = nn.BatchNorm2d(128)

        self.bn3_1 = nn.BatchNorm2d(256)
        self.bn3_2 = nn.BatchNorm2d(256)

        self.bn4_1 = nn.BatchNorm2d(512)
        self.bn4_2 = nn.BatchNorm2d(512)

        self.bn5_1 = nn.BatchNorm2d(256)
        self.bn5_2 = nn.BatchNorm2d(256)

        self.bn6_1 = nn.BatchNorm2d(128)
        self.bn6_2 = nn.BatchNorm2d(128)

        self.bn7_1 = nn.BatchNorm2d(64)
        self.bn7_2 = nn.BatchNorm2d(64)

        self.bn8_1 = nn.BatchNorm2d(32)
        self.bn8_2 = nn.BatchNorm2d(32)

        if compress_level > 0:
            assert compress_level <= 8
            self.compress_level = compress_level
            compress_channel_num = 256 // (2**compress_level)

            # currently only support compress/decompress at layer x_3
            self.com_compresser = nn.Conv2d(
                256, compress_channel_num, kernel_size=1, stride=1
            )
            self.bn_compress = nn.BatchNorm2d(compress_channel_num)

            self.com_decompresser = nn.Conv2d(
                compress_channel_num, 256, kernel_size=1, stride=1
            )
            self.bn_decompress = nn.BatchNorm2d(256)

    def encode(self, x):
        """Encode the input BEV features.

        Args:
            x (tensor): the input BEV features.

        Returns:
            A list that contains all the encoded layers.
        """
        batch, seq, z, h, w = x.size()

        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x = x.to(torch.float)
        x = F.relu(self.bn_pre_1(self.conv_pre_1(x)))
        x = F.relu(self.bn_pre_2(self.conv_pre_2(x)))

        # -------------------------------- Encoder Path --------------------------------
        # -- STC block 1
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))

        x_1 = x_1.view(
            batch, -1, x_1.size(1), x_1.size(2), x_1.size(3)
        ).contiguous()  # (batch, seq, c, h, w)
        x_1 = self.conv3d_1(x_1)
        x_1 = x_1.view(
            -1, x_1.size(2), x_1.size(3), x_1.size(4)
        ).contiguous()  # (batch * seq, c, h, w)

        # -- STC block 2
        x_2 = F.relu(self.bn2_1(self.conv2_1(x_1)))
        x_2 = F.relu(self.bn2_2(self.conv2_2(x_2)))

        x_2 = x_2.view(
            batch, -1, x_2.size(1), x_2.size(2), x_2.size(3)
        ).contiguous()  # (batch, seq, c, h, w)
        x_2 = self.conv3d_2(x_2)
        x_2 = x_2.view(
            -1, x_2.size(2), x_2.size(3), x_2.size(4)
        ).contiguous()  # (batch * seq, c, h, w), seq = 1

        # -- STC block 3
        x_3 = F.relu(self.bn3_1(self.conv3_1(x_2)))
        x_3 = F.relu(self.bn3_2(self.conv3_2(x_3)))

        # -- STC block 4
        x_4 = F.relu(self.bn4_1(self.conv4_1(x_3)))
        x_4 = F.relu(self.bn4_2(self.conv4_2(x_4)))

        # compress x_3 (the layer that agents communicates on)
        if self.compress_level > 0:
            x_3 = F.relu(self.bn_compress(self.com_compresser(x_3)))
            x_3 = F.relu(self.bn_decompress(self.com_decompresser(x_3)))

        return [x, x_1, x_2, x_3, x_4]

    def decode(
        self,
        x,
        x_1,
        x_2,
        x_3,
        x_4,
        batch,
        kd_flag=False,
        requires_adaptive_max_pool3d=False,
    ):
        """Decode the input features.

        Args:
            x (tensor): layer-0 features.
            x_1 (tensor): layer-1 features.
            x_2 (tensor): layer-2 features.
            x_3 (tensor): layer-3 features.
            x_4 (tensor): layer-4 featuers.
            batch (int): The batch size.
            kd_flag (bool, optional): Required to be true for DiscoNet. Defaults to False.
            requires_adaptive_max_pool3d (bool, optional): If set to true, use adaptive max pooling 3d. Defaults to False.

        Returns:
            if kd_flag is true, return a list of output from layer-8 to layer-5
            else return a list of a single element: the output after passing through the decoder
        """
        # -------------------------------- Decoder Path --------------------------------
        x_5 = F.relu(
            self.bn5_1(
                self.conv5_1(
                    torch.cat((F.interpolate(x_4, scale_factor=(2, 2)), x_3), dim=1)
                )
            )
        )
        x_5 = F.relu(self.bn5_2(self.conv5_2(x_5)))

        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3))
        x_2 = x_2.permute(0, 2, 1, 3, 4).contiguous()
        x_2 = (
            F.adaptive_max_pool3d(x_2, (1, None, None))
            if requires_adaptive_max_pool3d
            else x_2
        )
        x_2 = x_2.permute(0, 2, 1, 3, 4).contiguous()
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()

        x_6 = F.relu(
            self.bn6_1(
                self.conv6_1(
                    torch.cat((F.interpolate(x_5, scale_factor=(2, 2)), x_2), dim=1)
                )
            )
        )
        x_6 = F.relu(self.bn6_2(self.conv6_2(x_6)))

        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3))
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous()
        x_1 = (
            F.adaptive_max_pool3d(x_1, (1, None, None))
            if requires_adaptive_max_pool3d
            else x_1
        )
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous()
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()

        x_7 = F.relu(
            self.bn7_1(
                self.conv7_1(
                    torch.cat((F.interpolate(x_6, scale_factor=(2, 2)), x_1), dim=1)
                )
            )
        )
        x_7 = F.relu(self.bn7_2(self.conv7_2(x_7)))

        x = x.view(batch, -1, x.size(1), x.size(2), x.size(3))
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = (
            F.adaptive_max_pool3d(x, (1, None, None))
            if requires_adaptive_max_pool3d
            else x
        )
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(-1, x.size(2), x.size(3), x.size(4)).contiguous()

        x_8 = F.relu(
            self.bn8_1(
                self.conv8_1(
                    torch.cat((F.interpolate(x_7, scale_factor=(2, 2)), x), dim=1)
                )
            )
        )
        res_x = F.relu(self.bn8_2(self.conv8_2(x_8)))

        if kd_flag:
            return [res_x, x_7, x_6, x_5]
        else:
            return [res_x]


class STPN_KD(Backbone):
    """Used by non-intermediate models. Pass the output from encoder directly to decoder."""

    def __init__(self, height_feat_size=13, compress_level=0):
        super().__init__(height_feat_size, compress_level)

    def forward(self, x):
        batch, seq, z, h, w = x.size()
        encoded_layers = super().encode(x)
        decoded_layers = super().decode(
            *encoded_layers, batch, kd_flag=True, requires_adaptive_max_pool3d=True
        )
        return (*decoded_layers, encoded_layers[3], encoded_layers[4])


class LidarEncoder(Backbone):
    """The encoder class. Encodes input features in forward pass."""

    def __init__(self, height_feat_size=13, compress_level=0):
        super().__init__(height_feat_size, compress_level)

    def forward(self, x):
        return super().encode(x)


class LidarDecoder(Backbone):
    """The decoder class. Decodes input features in forward pass."""

    def __init__(self, height_feat_size=13):
        super().__init__(height_feat_size)

    def forward(self, x, x_1, x_2, x_3, x_4, batch, kd_flag=False):
        return super().decode(x, x_1, x_2, x_3, x_4, batch, kd_flag)


class Conv3D(nn.Module):
    """3D cnn used in the encoder."""

    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(Conv3D, self).__init__()
        self.conv3d = nn.Conv3d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn3d = nn.BatchNorm3d(out_channel)

    def forward(self, x):
        # input x: (batch, seq, c, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, c, seq_len, h, w)
        x = F.relu(self.bn3d(self.conv3d(x)))
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, seq_len, c, h, w)

        return x


"""""" """""" """""" """
Added by Yiming

""" """""" """""" """"""


class Conv2DBatchNormRelu(nn.Module):
    def __init__(
        self,
        in_channels,
        n_filters,
        k_size,
        stride,
        padding,
        bias=True,
        dilation=1,
        is_batchnorm=True,
    ):
        super(Conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        if is_batchnorm:
            self.cbr_unit = nn.Sequential(
                conv_mod, nn.BatchNorm2d(int(n_filters)), nn.ReLU(inplace=True)
            )
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class Sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self, dim=None):
        """Initialize sparsemax activation

        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsemax, self).__init__()

        self.dim = -1 if dim is None else dim

    def forward(self, input):
        """Forward function.
        Args:
            input (torch.Tensor): Input tensor. First dimension should be the batch size
        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor
        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape and reshape back after sparsemax
        original_size = input.size()
        input = input.view(-1, input.size(self.dim))

        dim = 1
        number_of_logits = input.size(dim)

        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.range(start=1, end=number_of_logits, device=input.device).view(
            1, -1
        )
        range = range.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        # Sparsemax
        self.output = torch.max(torch.zeros_like(input), input - taus)

        output = self.output.view(original_size)

        return output

    def backward(self, grad_output):
        """Backward function."""
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input
