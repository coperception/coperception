from coperception.models.det.backbone.Backbone import *
import numpy as np


class DetModelBase(nn.Module):
    """Abstract class. The super class for all detection models.
    
    Attributes:
        motion_state (bool): To return motion state in the loss calculation method or not
        out_seq_len (int): Length of output sequence
        box_code_size (int): The specification for bounding box encoding.
        category_num (int): Number of categories.
        use_map (bool): use_map
        anchor_num_per_loc (int): Anchor number per location.
        classification (nn.Module): The classification head.
        regression (nn.Module): The regression head.
        agent_num (int): The number of agent (including RSU and vehicles)
        kd_flag (bool): Required for DiscoNet.
        layer (int): Collaborate at which layer.
        p_com_outage (float): The probability of communication outage.
        neighbor_feat_list (list): The list of neighbor features.
        tg_agent (tensor): Features of the current target agent.
    """    
    def __init__(self, config, layer=3, in_channels=13, kd_flag=True, p_com_outage=0.0, num_agent=5):
        super(DetModelBase, self).__init__()

        self.motion_state = config.motion_state
        self.out_seq_len = 1 if config.only_det else config.pred_len
        self.box_code_size = config.box_code_size
        self.category_num = config.category_num
        self.use_map = config.use_map
        self.anchor_num_per_loc = len(config.anchor_size)
        self.classification = ClassificationHead(config)
        self.regression = SingleRegressionHead(config)
        self.agent_num = num_agent
        self.kd_flag = kd_flag
        self.layer = layer
        self.p_com_outage = p_com_outage
        self.neighbor_feat_list = []
        self.tg_agent = None

    def agents_to_batch(self, feats):
        """Concatenate the features of all agents back into a bacth.

        Args:
            feats (tensor): features

        Returns:
            The concatenated feature matrix of all agents.
        """        
        feat_list = []
        for i in range(self.agent_num):
            feat_list.append(feats[:, i, :, :, :])
        feat_mat = torch.cat(tuple(feat_list), 0)
        return feat_mat

    def get_feature_maps_and_size(self, encoded_layers: list):
        """Get the features of the collaboration layer and return the corresponding size.

        Args:
            encoded_layers (list): The output from the encoder.

        Returns:
            Feature map of the collaboration layer and the corresponding size.
        """        
        feature_maps = encoded_layers[self.layer]

        size_tuple = (
            (1, 32, 256, 256),
            (1, 64, 128, 128),
            (1, 128, 64, 64),
            (1, 256, 32, 32),
            (1, 512, 16, 16)
        )
        size = size_tuple[self.layer]

        return feature_maps, size

    def build_feature_list(self, batch_size: int, feat_maps) -> list:
        """Get the feature maps for each agent 
        
        e.g: [10 512 16 16] -> [2 5 512 16 16] [batch size, agent num, channel, height, width]

        Args:
            batch_size (int): The batch size.
            feat_maps (tensor): The feature maps of the collaboration layer.

        Returns:
            A list of feature maps for each agent.
        """        
        feature_map = {}
        feature_list = []

        for i in range(self.agent_num):
            feature_map[i] = torch.unsqueeze(feat_maps[batch_size * i:batch_size * (i + 1)], 1)
            feature_list.append(feature_map[i])

        return feature_list

    @staticmethod
    def build_local_communication_matrix(feature_list: list):
        """Concatendate the feature list into a tensor.

        Args:
            feature_list (list): The input feature list for each agent.

        Returns:
            A tensor of concatenated features.
        """        
        return torch.cat(tuple(feature_list), 1)

    def outage(self) -> bool:
        """Simulate communication outage according to self.p_com_outage.

        Returns:
            A bool indicating if the communication outage happens.
        """        
        return np.random.choice([True, False], p=[self.p_com_outage, 1 - self.p_com_outage])

    @staticmethod
    def feature_transformation(b, j, local_com_mat, all_warp, device, size):
        """Transform the features of the other agent (j) to the coordinate system of the current agent.

        Args:
            b (int): The index of the sample in current batch.
            j (int): The index of the other agent.
            local_com_mat (tensor): The local communication matrix. Features of all the agents.
            all_warp (tensor): The warp matrix for current sample for the current agent.
            device: The device used for PyTorch.
            size (tuple): Size of the feature map.

        Returns:
            A tensor of transformed features of agent j.
        """              
        nb_agent = torch.unsqueeze(local_com_mat[b, j], 0)  # [1 512 16 16]
        nb_warp = all_warp[j]  # [4 4]
        # normalize the translation vector
        x_trans = (4 * nb_warp[0, 3]) / 128
        y_trans = -(4 * nb_warp[1, 3]) / 128

        theta_rot = torch.tensor(
            [[nb_warp[0, 0], nb_warp[0, 1], 0.0], [nb_warp[1, 0], nb_warp[1, 1], 0.0]]).type(
            dtype=torch.float).to(device)
        theta_rot = torch.unsqueeze(theta_rot, 0)
        grid_rot = F.affine_grid(theta_rot, size=torch.Size(size))  # get grid for grid sample

        theta_trans = torch.tensor([[1.0, 0.0, x_trans], [0.0, 1.0, y_trans]]).type(dtype=torch.float).to(
            device)
        theta_trans = torch.unsqueeze(theta_trans, 0)
        grid_trans = F.affine_grid(theta_trans, size=torch.Size(size))  # get grid for grid sample

        # first rotate the feature map, then translate it
        warp_feat_rot = F.grid_sample(nb_agent, grid_rot, mode='bilinear')
        warp_feat_trans = F.grid_sample(warp_feat_rot, grid_trans, mode='bilinear')
        return torch.squeeze(warp_feat_trans)

    def build_neighbors_feature_list(self, b, agent_idx, all_warp, num_agent, local_com_mat, device,
                                     size) -> None:
        """Append the features of the neighbors of current agent to the neighbor_feat_list list.

        Args:
            b (int): The index of the sample in current batch.
            agent_idx (int): The index of the current agent.
            all_warp (tensor): The warp matrix for current sample for the current agent.
            num_agent (int): The number of agents.
            local_com_mat (tensor): The local communication matrix. Features of all the agents.
            device: The device used for PyTorch.
            size (tuple): Size of the feature map.
        """                                     
        for j in range(num_agent):
            if j != agent_idx:
                warp_feat = DetModelBase.feature_transformation(b, j, local_com_mat, all_warp, device, size)
                self.neighbor_feat_list.append(warp_feat)

    def get_decoded_layers(self, encoded_layers, feature_fuse_matrix, batch_size):
        """Replace the collaboration layer of the output from the encoder with fused feature maps.

        Args:
            encoded_layers (list): The output from the encoder.
            feature_fuse_matrix (tensor): The fused feature maps.
            batch_size (int): The batch size.

        Returns:
            A list. Output from the decoder.
        """        
        encoded_layers[self.layer] = feature_fuse_matrix
        decoded_layers = self.decoder(*encoded_layers, batch_size, kd_flag=self.kd_flag)
        return decoded_layers

    def get_cls_loc_result(self, x):
        """Get the classification and localization result.

        Args:
            x (tensor): The output from the last layer of the decoder.

        Returns:
            cls_preds (tensor): Predictions of the classification head.
            loc_preds (tensor): Predications of the localization head.
            result (dict): A dictionary of classificaion, localization, and optional motion state classification result.
        """        
        # Cell Classification head
        cls_preds = self.classification(x)
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.view(cls_preds.shape[0], -1, self.category_num)

        # Detection head
        loc_preds = self.regression(x)
        loc_preds = loc_preds.permute(0, 2, 3, 1).contiguous()
        loc_preds = loc_preds.view(-1, loc_preds.size(1), loc_preds.size(2), self.anchor_num_per_loc, self.out_seq_len,
                                   self.box_code_size)

        # loc_pred (N * T * W * H * loc)
        result = {'loc': loc_preds,
                  'cls': cls_preds}

        # MotionState head
        if self.motion_state:
            motion_cat = 3
            motion_cls_preds = self.motion_cls(x)
            motion_cls_preds = motion_cls_preds.permute(0, 2, 3, 1).contiguous()
            motion_cls_preds = motion_cls_preds.view(cls_preds.shape[0], -1, motion_cat)
            result['state'] = motion_cls_preds

        return cls_preds, loc_preds, result


class ClassificationHead(nn.Module):
    """The classificaion head."""

    def __init__(self, config):
        super(ClassificationHead, self).__init__()

        category_num = config.category_num
        channel = 32
        if config.use_map:
            channel += 6
        if config.use_vis:
            channel += 13

        anchor_num_per_loc = len(config.anchor_size)

        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channel, category_num * anchor_num_per_loc, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(channel)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        return x


class SingleRegressionHead(nn.Module):
    """The regression head."""
    def __init__(self, config):
        super(SingleRegressionHead, self).__init__()

        channel = 32
        if config.use_map:
            channel += 6
        if config.use_vis:
            channel += 13

        anchor_num_per_loc = len(config.anchor_size)
        box_code_size = config.box_code_size
        out_seq_len = 1 if config.only_det else config.pred_len

        if config.binary:
            if config.only_det:
                self.box_prediction = nn.Sequential(
                    nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(channel),
                    nn.ReLU(),
                    nn.Conv2d(channel, anchor_num_per_loc * box_code_size * out_seq_len, kernel_size=1, stride=1,
                              padding=0))
            else:
                self.box_prediction = nn.Sequential(
                    nn.Conv2d(channel, 128, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(128, anchor_num_per_loc * box_code_size * out_seq_len, kernel_size=1, stride=1,
                              padding=0))

    def forward(self, x):
        box = self.box_prediction(x)

        return box
