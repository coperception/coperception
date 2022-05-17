from coperception.models.seg.SegModelBase import SegModelBase
import torch.nn.functional as F
import torch.nn as nn
import torch
import copy
import numpy as np
from coperception.models.seg.SegModelBase import Down, DoubleConv


class When2Com_UNet(SegModelBase):
    def __init__(
        self,
        config,
        n_classes=21,
        in_channels=13,
        has_query=True,
        sparse=False,
        layer=3,
        warp_flag=1,
        image_size=512,
        shared_img_encoder="unified",
        key_size=1024,
        query_size=32,
        num_agent=5,
        compress_level=0,
    ):
        super().__init__(
            in_channels, n_classes, num_agent=num_agent, compress_level=compress_level
        )
        self.motion_state = config.motion_state
        if config.only_det:
            self.out_seq_len = 1
        else:
            self.out_seq_len = config.pred_len
        self.box_code_size = config.box_code_size
        self.category_num = config.category_num
        self.use_map = config.use_map
        self.anchor_num_per_loc = len(config.anchor_size)
        # self.classification = ClassificationHead(config)
        # self.regression = SingleRegressionHead(config)

        self.sparse = sparse
        # self.u_encoder = lidar_encoder(height_feat_size=in_channels)
        self.key_size = key_size
        self.query_size = query_size
        self.shared_img_encoder = shared_img_encoder
        self.has_query = has_query
        self.warp_flag = warp_flag
        self.layer = layer

        self.key_net = KmGenerator(
            out_size=self.key_size, input_feat_sz=image_size / 32
        )
        self.attention_net = MIMOGeneralDotProductAttention(
            self.query_size, self.key_size, self.warp_flag
        )
        # # Message generator
        self.query_key_net = PolicyNet4(in_channels=in_channels)
        if self.has_query:
            self.query_net = KmGenerator(
                out_size=self.query_size, input_feat_sz=image_size / 32
            )

        # Detection decoder
        # self.decoder = lidar_decoder(height_feat_size=in_channels)

        # List the parameters of each modules
        self.attention_paras = list(self.attention_net.parameters())
        # if self.shared_img_encoder == 'unified':
        #    self.img_net_paras = list(self.u_encoder.parameters()) + list(self.decoder.parameters())

        self.policy_net_paras = (
            list(self.query_key_net.parameters())
            + list(self.key_net.parameters())
            + self.attention_paras
        )
        if self.has_query:
            self.policy_net_paras = self.policy_net_paras + list(
                self.query_net.parameters()
            )

        # self.all_paras = self.img_net_paras + self.policy_net_paras

        # if self.motion_state:
        #    self.motion_cls = MotionStateHead(config)

    def argmax_select(self, warp_flag, val_mat, prob_action):
        # v(batch, query_num, channel, size, size)
        cls_num = prob_action.shape[1]

        coef_argmax = F.one_hot(prob_action.max(dim=1)[1], num_classes=cls_num).type(
            torch.cuda.FloatTensor
        )
        coef_argmax = coef_argmax.transpose(1, 2)
        attn_shape = coef_argmax.shape
        bats, key_num, query_num = attn_shape[0], attn_shape[1], attn_shape[2]
        coef_argmax_exp = coef_argmax.view(bats, key_num, query_num, 1, 1, 1)

        if warp_flag == 1:
            v_exp = val_mat
        else:
            v_exp = torch.unsqueeze(val_mat, 2)
            v_exp = v_exp.expand(-1, -1, query_num, -1, -1, -1)

        output = coef_argmax_exp * v_exp  # (batch,4,channel,size,size)
        feat_argmax = output.sum(1)  # (batch,1,channel,size,size)

        # compute connect
        count_coef = copy.deepcopy(coef_argmax)
        ind = np.diag_indices(self.num_agent)
        count_coef[:, ind[0], ind[1]] = 0
        num_connect = torch.nonzero(count_coef).shape[0] / (
            self.num_agent * count_coef.shape[0]
        )

        return feat_argmax, coef_argmax, num_connect

    def activated_select(self, warp_flag, val_mat, prob_action, thres=0.2):

        coef_act = torch.mul(prob_action, (prob_action > thres).float())
        attn_shape = coef_act.shape
        bats, key_num, query_num = attn_shape[0], attn_shape[1], attn_shape[2]
        coef_act_exp = coef_act.view(bats, key_num, query_num, 1, 1, 1)

        if warp_flag == 1:
            v_exp = val_mat
        else:
            v_exp = torch.unsqueeze(val_mat, 2)
            v_exp = v_exp.expand(-1, -1, query_num, -1, -1, -1)

        output = coef_act_exp * v_exp  # (batch,4,channel,size,size)
        feat_act = output.sum(1)  # (batch,1,channel,size,size)

        # compute connect
        count_coef = coef_act.clone()
        ind = np.diag_indices(self.num_agent)
        count_coef[:, ind[0], ind[1]] = 0
        num_connect = torch.nonzero(count_coef).shape[0] / (
            self.num_agent * count_coef.shape[0]
        )
        return feat_act, coef_act, num_connect

    def forward(
        self,
        bevs,
        trans_matrices,
        num_agent_tensor,
        maps=None,
        vis=None,
        training=True,
        MO_flag=True,
        inference="activated",
        batch_size=1,
    ):
        device = bevs.device
        x1 = self.inc(bevs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)  # b 512 32 32
        size = (1, 512, 32, 32)

        if self.compress_level > 0:
            x4 = F.relu(self.bn_compress(self.com_compresser(x4)))
            x4 = F.relu(self.bn_decompress(self.com_decompresser(x4)))

        batch_size = bevs.size(0) // self.num_agent
        val_mat = torch.zeros(
            batch_size, self.num_agent, self.num_agent, 512, 32, 32
        ).to(device)

        # get feat maps for each agent
        feat_list = super().build_feat_list(x4, batch_size)

        """""" """""" """""" """""" """""" """""" """""" """""" """""" """
         generate value matrix for each agent, Yiming, 2021.4.22

        """ """""" """""" """""" """""" """""" """""" """""" """""" """"""
        if self.warp_flag == 1:
            local_com_mat = torch.cat(
                tuple(feat_list), 1
            )  # [2 5 512 16 16] [batch, agent, channel, height, width]
            for b in range(batch_size):
                com_num_agent = num_agent_tensor[b, 0]
                for i in range(com_num_agent):
                    tg_agent = local_com_mat[b, i]
                    for j in range(com_num_agent):
                        if j == i:
                            val_mat[b, i, j] = tg_agent
                        else:
                            val_mat[b, i, j] = super().feature_transformation(
                                b,
                                j,
                                i,
                                local_com_mat,
                                size,
                                trans_matrices,
                            )
        else:
            val_mat = torch.cat(tuple(feat_list), 1)

        # pass feature maps through key and query generator
        query_key_maps = self.query_key_net(bevs)
        keys = self.key_net(query_key_maps)

        if self.has_query:
            querys = self.query_net(query_key_maps)
        # get key and query
        key = {}
        query = {}
        key_list = []
        query_list = []

        for i in range(self.num_agent):
            key[i] = torch.unsqueeze(keys[batch_size * i : batch_size * (i + 1)], 1)
            key_list.append(key[i])
            if self.has_query:
                query[i] = torch.unsqueeze(
                    querys[batch_size * i : batch_size * (i + 1)], 1
                )
            else:
                query[i] = torch.ones(batch_size, 1, self.query_size).to("cuda")
            query_list.append(query[i])

        key_mat = torch.cat(tuple(key_list), 1)
        query_mat = torch.cat(tuple(query_list), 1)
        if MO_flag:
            query_mat = query_mat
        else:
            query_mat = torch.unsqueeze(query_mat[:, 0, :], 1)

        feat_fuse, prob_action = self.attention_net(
            query_mat, key_mat, val_mat, sparse=self.sparse
        )
        # print(query_mat.shape, key_mat.shape, val_mat.shape, feat_fuse.shape)
        # weighted feature maps is passed to decoder
        feat_fuse_mat = self.agents_to_batch(feat_fuse)

        # not related to how we combine the feature (prefer to use the agnets' own frames: to reduce the bandwidth)
        small_bis = torch.eye(prob_action.shape[1]) * 0.001
        small_bis = small_bis.reshape((1, prob_action.shape[1], prob_action.shape[2]))
        small_bis = small_bis.repeat(prob_action.shape[0], 1, 1).cuda()
        prob_action = prob_action + small_bis

        if training:
            # action = torch.argmax(prob_action, dim=1)
            # num_connect = self.num_agent - 1
            x5 = self.down4(feat_fuse_mat)
            x = self.up1(x5, feat_fuse_mat)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
        else:
            if inference == "softmax":
                # action = torch.argmax(prob_action, dim=1)
                # num_connect = self.num_agent - 1
                x5 = self.down4(feat_fuse_mat)
                x = self.up1(x5, feat_fuse_mat)
                x = self.up2(x, x3)
                x = self.up3(x, x2)
                x = self.up4(x, x1)
            elif inference == "argmax_test":
                # print('argmax_test')
                feat_argmax, connect_mat, num_connect = self.argmax_select(
                    self.warp_flag, val_mat, prob_action
                )
                feat_argmax_mat = self.agents_to_batch(
                    feat_argmax
                )  # (batchsize*agent_num, channel, size, size)
                feat_argmax_mat = feat_argmax_mat.detach()
                # pred_argmax = self.decoder(x, x_1, x_2, feat_argmax_mat, x_4, batch_size)

                x5 = self.down4(feat_argmax_mat)
                x = self.up1(x5, feat_argmax_mat)
                x = self.up2(x, x3)
                x = self.up3(x, x2)
                x = self.up4(x, x1)
                # action = torch.argmax(connect_mat, dim=1)
                # return pred_argmax, prob_action, action, num_connect
                # x=pred_argmax
            elif inference == "activated":
                # print('activated')
                feat_act, connect_mat, num_connect = self.activated_select(
                    self.warp_flag, val_mat, prob_action
                )
                feat_act_mat = self.agents_to_batch(
                    feat_act
                )  # (batchsize*agent_num, channel, size, size)
                feat_act_mat = feat_act_mat.detach()
                # if self.layer ==4:
                #    pred_act = self.decoder(x, x_1, x_2, x_3, feat_act_mat,batch_size)
                # elif self.layer == 3:
                #    pred_act = self.decoder(x, x_1, x_2, feat_act_mat, x_4, batch_size)
                # elif self.layer == 2:
                #    pred_act = self.decoder(x, x_1, feat_act_mat, x_3, x_4, batch_size)
                x5 = self.down4(feat_act_mat)
                x = self.up1(x5, feat_act_mat)
                x = self.up2(x, x3)
                x = self.up3(x, x2)
                x = self.up4(x, x1)
            else:
                raise ValueError("Incorrect inference mode")
        logits = self.outc(x)
        return logits


class PolicyNet4(nn.Module):
    def __init__(self, in_channels=13, input_feat_sz=32):
        super().__init__()
        feat_map_sz = input_feat_sz // 4
        self.n_feat = int(256 * feat_map_sz * feat_map_sz)
        # self.lidar_encoder = lidar_encoder(height_feat_size=in_channels)

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        # Encoder
        # down 1
        self.conv1 = conv2DBatchNormRelu(512, 512, k_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(512, 256, k_size=3, stride=1, padding=1)
        self.conv3 = conv2DBatchNormRelu(256, 256, k_size=3, stride=2, padding=1)

        # down 2
        self.conv4 = conv2DBatchNormRelu(256, 256, k_size=3, stride=1, padding=1)
        self.conv5 = conv2DBatchNormRelu(256, 256, k_size=3, stride=2, padding=1)

    def forward(self, features_map):
        # _, _, _, _, outputs1 = self.lidar_encoder(features_map)
        outputs1 = self.down3(self.down2(self.down1(self.inc(features_map))))
        outputs = self.conv1(outputs1)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        outputs = self.conv4(outputs)
        outputs = self.conv5(outputs)
        return outputs


class conv2DBatchNormRelu(nn.Module):
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
        super(conv2DBatchNormRelu, self).__init__()

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


class KmGenerator(nn.Module):
    def __init__(self, out_size=128, input_feat_sz=32):
        super().__init__()
        feat_map_sz = input_feat_sz // 4
        self.n_feat = int(256 * feat_map_sz * feat_map_sz)
        self.fc = nn.Sequential(
            nn.Linear(self.n_feat, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_size),
        )

    def forward(self, features_map):
        outputs = self.fc(features_map.view(-1, self.n_feat))
        return outputs


# MIMO (non warp)
class MIMOGeneralDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, query_size, key_size, warp_flag, attn_dropout=0.1):
        super().__init__()
        self.sparsemax = Sparsemax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.linear = nn.Linear(query_size, key_size)
        self.warp_flag = warp_flag
        print("Msg size: ", query_size, "  Key size: ", key_size)

    def forward(self, qu, k, v, sparse=True):
        # qu (batch,5,32)
        # k (batch,5,1024)
        # v (batch,5,channel,size,size)
        query = self.linear(qu)  # (batch,5,key_size)

        # normalization
        # query_norm = query.norm(p=2,dim=2).unsqueeze(2).expand_as(query)
        # query = query.div(query_norm + 1e-9)

        # k_norm = k.norm(p=2,dim=2).unsqueeze(2).expand_as(k)
        # k = k.div(k_norm + 1e-9)
        # generate the
        attn_orig = torch.bmm(
            k, query.transpose(2, 1)
        )  # (batch,5,5)  column: differnt keys and the same query

        # scaling [not sure]
        # scaling = torch.sqrt(torch.tensor(k.shape[2],dtype=torch.float32)).cuda()
        # attn_orig = attn_orig/ scaling # (batch,5,5)  column: differnt keys and the same query

        attn_orig_softmax = self.softmax(attn_orig)  # (batch,5,5)

        attn_shape = attn_orig_softmax.shape
        bats, key_num, query_num = attn_shape[0], attn_shape[1], attn_shape[2]
        attn_orig_softmax_exp = attn_orig_softmax.view(
            bats, key_num, query_num, 1, 1, 1
        )

        if self.warp_flag == 1:
            v_exp = v
        else:
            v_exp = torch.unsqueeze(v, 2)
            v_exp = v_exp.expand(-1, -1, query_num, -1, -1, -1)

        output = attn_orig_softmax_exp * v_exp  # (batch,5,channel,size,size)
        output_sum = output.sum(1)  # (batch,1,channel,size,size)

        return output_sum, attn_orig_softmax


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
