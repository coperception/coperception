from coperception.utils.obj_util import *
from torch.utils.data import Dataset
import numpy as np
import os
import math
from multiprocessing import Manager


class V2XSimDet(Dataset):
    def __init__(
        self,
        dataset_roots=None,
        config=None,
        config_global=None,
        split=None,
        cache_size=10000,
        val=False,
        bound=None,
        kd_flag=False,
        no_cross_road=False,
    ):
        """
        This dataloader loads single sequence for a keyframe, and is not designed for computing the
         spatio-temporal consistency losses. It supports train, val and test splits.

        dataset_root: Data path to the preprocessed sparse nuScenes data (for training)
        split: [train/val/test]
        future_frame_skip: Specify to skip how many future frames
        voxel_size: The lattice resolution. Should be consistent with the preprocessed data
        area_extents: The area extents of the processed LiDAR data. Should be consistent with the preprocessed data
        category_num: The number of object categories (including the background)
        cache_size: The cache size for storing parts of data in the memory (for reducing the IO cost)
        """
        if split is None:
            self.split = config.split
        else:
            self.split = split
        self.voxel_size = config.voxel_size
        self.area_extents = config.area_extents
        self.category_num = config.category_num
        self.future_frame_skip = config.future_frame_skip
        self.pred_len = config.pred_len
        self.box_code_size = config.box_code_size
        self.anchor_size = config.anchor_size

        self.val = val
        self.only_det = config.only_det
        self.binary = config.binary
        self.config = config
        self.use_vis = config.use_vis

        self.bound = bound
        self.kd_flag = kd_flag
        self.no_cross_road = no_cross_road

        # dataset_root = dataset_root + '/'+split
        if dataset_roots is None:
            raise ValueError(
                "The {} dataset root is None. Should specify its value.".format(
                    self.split
                )
            )
        self.dataset_roots = dataset_roots
        self.num_agent = len(dataset_roots)
        self.seq_files = []
        self.seq_scenes = []
        for dataset_root in self.dataset_roots:
            # sort directories
            dir_list = [d.split("_") for d in os.listdir(dataset_root)]
            dir_list.sort(key=lambda x: (int(x[0]), int(x[1])))
            self.seq_scenes.append(
                [int(s[0]) for s in dir_list]
            )  # which scene this frame belongs to (required for visualization)
            dir_list = ["_".join(x) for x in dir_list]

            seq_dirs = [
                os.path.join(dataset_root, d)
                for d in dir_list
                if os.path.isdir(os.path.join(dataset_root, d))
            ]

            self.seq_files.append(
                [
                    os.path.join(seq_dir, f)
                    for seq_dir in seq_dirs
                    for f in os.listdir(seq_dir)
                    if os.path.isfile(os.path.join(seq_dir, f))
                ]
            )

        self.num_sample_seqs = len(self.seq_files[0])
        print("The number of {} sequences: {}".format(self.split, self.num_sample_seqs))
        # object information
        self.anchors_map = init_anchors_no_check(
            self.area_extents, self.voxel_size, self.box_code_size, self.anchor_size
        )
        self.map_dims = [
            int(
                (self.area_extents[0][1] - self.area_extents[0][0]) / self.voxel_size[0]
            ),
            int(
                (self.area_extents[1][1] - self.area_extents[1][0]) / self.voxel_size[1]
            ),
        ]
        self.reg_target_shape = (
            self.map_dims[0],
            self.map_dims[1],
            len(self.anchor_size),
            self.pred_len,
            self.box_code_size,
        )
        self.label_shape = (self.map_dims[0], self.map_dims[1], len(self.anchor_size))
        self.label_one_hot_shape = (
            self.map_dims[0],
            self.map_dims[1],
            len(self.anchor_size),
            self.category_num,
        )
        self.dims = config.map_dims
        self.num_past_pcs = config.num_past_pcs
        manager = Manager()
        self.cache = [manager.dict() for _ in range(self.num_agent)]
        self.cache_size = cache_size if split == "train" else 0

        if self.val:
            self.voxel_size_global = config_global.voxel_size
            self.area_extents_global = config_global.area_extents
            self.pred_len_global = config_global.pred_len
            self.box_code_size_global = config_global.box_code_size
            self.anchor_size_global = config_global.anchor_size
            # object information
            self.anchors_map_global = init_anchors_no_check(
                self.area_extents_global,
                self.voxel_size_global,
                self.box_code_size_global,
                self.anchor_size_global,
            )
            self.map_dims_global = [
                int(
                    (self.area_extents_global[0][1] - self.area_extents_global[0][0])
                    / self.voxel_size_global[0]
                ),
                int(
                    (self.area_extents_global[1][1] - self.area_extents_global[1][0])
                    / self.voxel_size_global[1]
                ),
            ]
            self.reg_target_shape_global = (
                self.map_dims_global[0],
                self.map_dims_global[1],
                len(self.anchor_size_global),
                self.pred_len_global,
                self.box_code_size_global,
            )
            self.dims_global = config_global.map_dims
        self.get_meta()

    def get_meta(self):
        meta = NuscenesDataset(
            dataset_root=self.dataset_roots[0],
            split=self.split,
            config=self.config,
            val=self.val,
        )
        if not self.val:
            (
                self.padded_voxel_points_meta,
                self.label_one_hot_meta,
                self.reg_target_meta,
                self.reg_loss_mask_meta,
                self.anchors_map_meta,
                _,
                _,
                self.vis_maps_meta,
            ) = meta[0]
        else:
            (
                self.padded_voxel_points_meta,
                self.label_one_hot_meta,
                self.reg_target_meta,
                self.reg_loss_mask_meta,
                self.anchors_map_meta,
                _,
                _,
                self.vis_maps_meta,
                _,
                _,
            ) = meta[0]
        del meta

    def __len__(self):
        return self.num_sample_seqs

    def get_one_hot(self, label, category_num):
        one_hot_label = np.zeros((label.shape[0], category_num))
        for i in range(label.shape[0]):
            one_hot_label[i][label[i]] = 1

        return one_hot_label

    def pick_single_agent(self, agent_id, idx):
        empty_flag = False
        if idx in self.cache[agent_id]:
            gt_dict = self.cache[agent_id][idx]
        else:
            seq_file = self.seq_files[agent_id][idx]
            gt_data_handle = np.load(seq_file, allow_pickle=True)
            if gt_data_handle == 0:
                empty_flag = True
                padded_voxel_points = []
                padded_voxel_points_teacher = []
                label_one_hot = np.zeros_like(self.label_one_hot_meta)
                reg_target = np.zeros_like(self.reg_target_meta)
                anchors_map = np.zeros_like(self.anchors_map_meta)
                vis_maps = np.zeros_like(self.vis_maps_meta)
                reg_loss_mask = np.zeros_like(self.reg_loss_mask_meta)

                if self.bound == "lowerbound":
                    padded_voxel_points = np.zeros_like(self.padded_voxel_points_meta)

                if self.kd_flag or self.bound == "upperbound":
                    padded_voxel_points_teacher = np.zeros_like(
                        self.padded_voxel_points_meta
                    )

                if self.val:
                    return (
                        padded_voxel_points,
                        padded_voxel_points_teacher,
                        label_one_hot,
                        reg_target,
                        reg_loss_mask,
                        anchors_map,
                        vis_maps,
                        [{"gt_box": []}],
                        [seq_file],
                        0,
                        0,
                        np.zeros((self.num_agent, 4, 4)),
                    )
                else:
                    return (
                        padded_voxel_points,
                        padded_voxel_points_teacher,
                        label_one_hot,
                        reg_target,
                        reg_loss_mask,
                        anchors_map,
                        vis_maps,
                        0,
                        0,
                        np.zeros((self.num_agent, 4, 4)),
                    )
            else:
                gt_dict = gt_data_handle.item()
                if len(self.cache[agent_id]) < self.cache_size:
                    self.cache[agent_id][idx] = gt_dict

        if not empty_flag:
            allocation_mask = gt_dict["allocation_mask"].astype(bool)
            reg_loss_mask = gt_dict["reg_loss_mask"].astype(bool)
            gt_max_iou = gt_dict["gt_max_iou"]

            # load regression target
            reg_target_sparse = gt_dict["reg_target_sparse"]
            # need to be modified Yiqi , only use reg_target and allocation_map
            reg_target = np.zeros(self.reg_target_shape).astype(reg_target_sparse.dtype)

            reg_target[allocation_mask] = reg_target_sparse
            reg_target[np.bitwise_not(reg_loss_mask)] = 0
            label_sparse = gt_dict["label_sparse"]

            one_hot_label_sparse = self.get_one_hot(label_sparse, self.category_num)
            label_one_hot = np.zeros(self.label_one_hot_shape)
            label_one_hot[:, :, :, 0] = 1
            label_one_hot[allocation_mask] = one_hot_label_sparse

            if self.only_det:
                reg_target = reg_target[:, :, :, :1]
                reg_loss_mask = reg_loss_mask[:, :, :, :1]

            # only center for pred
            elif self.config.pred_type in ["motion", "center"]:
                reg_loss_mask = np.expand_dims(reg_loss_mask, axis=-1)
                reg_loss_mask = np.repeat(reg_loss_mask, self.box_code_size, axis=-1)
                reg_loss_mask[:, :, :, 1:, 2:] = False

            # Prepare padded_voxel_points
            padded_voxel_points = []
            if self.bound == "lowerbound" or self.bound == "both":
                for i in range(self.num_past_pcs):
                    indices = gt_dict["voxel_indices_" + str(i)]
                    curr_voxels = np.zeros(self.dims, dtype=bool)
                    curr_voxels[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
                    curr_voxels = np.rot90(curr_voxels, 3)
                    padded_voxel_points.append(curr_voxels)
                padded_voxel_points = np.stack(padded_voxel_points, 0).astype(
                    np.float32
                )
                padded_voxel_points = padded_voxel_points.astype(np.float32)

            anchors_map = self.anchors_map

            if self.config.use_vis:
                vis_maps = np.zeros(
                    (
                        self.num_past_pcs,
                        self.config.map_dims[-1],
                        self.config.map_dims[0],
                        self.config.map_dims[1],
                    )
                )
                vis_free_indices = gt_dict["vis_free_indices"]
                vis_occupy_indices = gt_dict["vis_occupy_indices"]
                vis_maps[
                    vis_occupy_indices[0, :],
                    vis_occupy_indices[1, :],
                    vis_occupy_indices[2, :],
                    vis_occupy_indices[3, :],
                ] = math.log(0.7 / (1 - 0.7))
                vis_maps[
                    vis_free_indices[0, :],
                    vis_free_indices[1, :],
                    vis_free_indices[2, :],
                    vis_free_indices[3, :],
                ] = math.log(0.4 / (1 - 0.4))
                vis_maps = np.swapaxes(vis_maps, 2, 3)
                vis_maps = np.transpose(vis_maps, (0, 2, 3, 1))
                for v_id in range(vis_maps.shape[0]):
                    vis_maps[v_id] = np.rot90(vis_maps[v_id], 3)
                vis_maps = vis_maps[-1]
            else:
                vis_maps = np.zeros(0)

            if self.no_cross_road:
                trans_matrices = gt_dict["trans_matrices_no_cross_road"]
            else:
                trans_matrices = gt_dict["trans_matrices"]

            label_one_hot = label_one_hot.astype(np.float32)
            reg_target = reg_target.astype(np.float32)
            anchors_map = anchors_map.astype(np.float32)
            vis_maps = vis_maps.astype(np.float32)

            target_agent_id = gt_dict["target_agent_id"]
            num_sensor = gt_dict["num_sensor"]

            # Prepare padded_voxel_points_teacher
            padded_voxel_points_teacher = []
            if "voxel_indices_teacher" in gt_dict and (
                self.kd_flag or self.bound == "upperbound" or self.bound == "both"
            ):
                if self.no_cross_road:
                    indices_teacher = gt_dict["voxel_indices_teacher_no_cross_road"]
                else:
                    indices_teacher = gt_dict["voxel_indices_teacher"]

                curr_voxels_teacher = np.zeros(self.dims, dtype=bool)
                curr_voxels_teacher[
                    indices_teacher[:, 0], indices_teacher[:, 1], indices_teacher[:, 2]
                ] = 1
                curr_voxels_teacher = np.rot90(curr_voxels_teacher, 3)
                padded_voxel_points_teacher.append(curr_voxels_teacher)
                padded_voxel_points_teacher = np.stack(
                    padded_voxel_points_teacher, 0
                ).astype(np.float32)
                padded_voxel_points_teacher = padded_voxel_points_teacher.astype(
                    np.float32
                )

            if self.val:
                return (
                    padded_voxel_points,
                    padded_voxel_points_teacher,
                    label_one_hot,
                    reg_target,
                    reg_loss_mask,
                    anchors_map,
                    vis_maps,
                    [{"gt_box": gt_max_iou}],
                    [seq_file],
                    target_agent_id,
                    num_sensor,
                    trans_matrices,
                )

            else:
                return (
                    padded_voxel_points,
                    padded_voxel_points_teacher,
                    label_one_hot,
                    reg_target,
                    reg_loss_mask,
                    anchors_map,
                    vis_maps,
                    target_agent_id,
                    num_sensor,
                    trans_matrices,
                )

    def __getitem__(self, idx):
        res = []
        for i in range(self.num_agent):
            res.append(self.pick_single_agent(i, idx))
        return res


class NuscenesDataset(Dataset):
    def __init__(
        self, dataset_root=None, config=None, split=None, cache_size=10000, val=False
    ):
        """
        This dataloader loads single sequence for a keyframe, and is not designed for computing the
         spatio-temporal consistency losses. It supports train, val and test splits.

        dataset_root: Data path to the preprocessed sparse nuScenes data (for training)
        split: [train/val/test]
        future_frame_skip: Specify to skip how many future frames
        voxel_size: The lattice resolution. Should be consistent with the preprocessed data
        area_extents: The area extents of the processed LiDAR data. Should be consistent with the preprocessed data
        category_num: The number of object categories (including the background)
        cache_size: The cache size for storing parts of data in the memory (for reducing the IO cost)
        """
        if split is None:
            self.split = config.split
        else:
            self.split = split
        self.voxel_size = config.voxel_size
        self.area_extents = config.area_extents
        self.category_num = config.category_num
        self.future_frame_skip = config.future_frame_skip
        self.pred_len = config.pred_len
        self.box_code_size = config.box_code_size
        self.anchor_size = config.anchor_size
        self.val = val
        self.only_det = config.only_det
        self.binary = config.binary
        self.config = config
        self.use_vis = config.use_vis
        # dataset_root = dataset_root + '/'+split
        if dataset_root is None:
            raise ValueError(
                "The {} dataset root is None. Should specify its value.".format(
                    self.split
                )
            )
        self.dataset_root = dataset_root
        seq_dirs = [
            os.path.join(self.dataset_root, d)
            for d in os.listdir(self.dataset_root)
            if os.path.isdir(os.path.join(self.dataset_root, d))
        ]
        seq_dirs = sorted(seq_dirs)
        self.seq_files = [
            os.path.join(seq_dir, f)
            for seq_dir in seq_dirs
            for f in os.listdir(seq_dir)
            if os.path.isfile(os.path.join(seq_dir, f))
        ]

        self.num_sample_seqs = len(self.seq_files)
        print("The number of {} sequences: {}".format(self.split, self.num_sample_seqs))

        """
        # For training, the size of dataset should be 17065 * 2; for validation: 1623; for testing: 4309
        if split == 'train' and self.num_sample_seqs != 17065 * 2:
            warnings.warn(">> The size of training dataset is not 17065 * 2.\n")
        elif split == 'val' and self.num_sample_seqs != 1623:
            warnings.warn(">> The size of validation dataset is not 1719.\n")
        elif split == 'test' and self.num_sample_seqs != 4309:
            warnings.warn('>> The size of test dataset is not 4309.\n')
        """

        # object information
        self.anchors_map = init_anchors_no_check(
            self.area_extents, self.voxel_size, self.box_code_size, self.anchor_size
        )
        self.map_dims = [
            int(
                (self.area_extents[0][1] - self.area_extents[0][0]) / self.voxel_size[0]
            ),
            int(
                (self.area_extents[1][1] - self.area_extents[1][0]) / self.voxel_size[1]
            ),
        ]
        self.reg_target_shape = (
            self.map_dims[0],
            self.map_dims[1],
            len(self.anchor_size),
            self.pred_len,
            self.box_code_size,
        )
        self.label_shape = (self.map_dims[0], self.map_dims[1], len(self.anchor_size))
        self.label_one_hot_shape = (
            self.map_dims[0],
            self.map_dims[1],
            len(self.anchor_size),
            self.category_num,
        )
        self.dims = config.map_dims
        self.num_past_pcs = config.num_past_pcs
        manager = Manager()
        self.cache = manager.dict()
        self.cache_size = cache_size if split == "train" else 0
        # self.cache_size = cache_size

    def __len__(self):
        return self.num_sample_seqs

    def get_one_hot(self, label, category_num):
        one_hot_label = np.zeros((label.shape[0], category_num))
        for i in range(label.shape[0]):
            one_hot_label[i][label[i]] = 1

        return one_hot_label

    def __getitem__(self, idx):
        if idx in self.cache:
            gt_dict = self.cache[idx]
        else:
            seq_file = self.seq_files[idx]
            gt_data_handle = np.load(seq_file, allow_pickle=True)
            gt_dict = gt_data_handle.item()
            if len(self.cache) < self.cache_size:
                self.cache[idx] = gt_dict

        allocation_mask = gt_dict["allocation_mask"].astype(bool)
        reg_loss_mask = gt_dict["reg_loss_mask"].astype(bool)
        gt_max_iou = gt_dict["gt_max_iou"]
        motion_one_hot = np.zeros(5)
        motion_mask = np.zeros(5)

        # load regression target
        reg_target_sparse = gt_dict["reg_target_sparse"]
        # need to be modified Yiqi , only use reg_target and allocation_map
        reg_target = np.zeros(self.reg_target_shape).astype(reg_target_sparse.dtype)

        reg_target[allocation_mask] = reg_target_sparse
        reg_target[np.bitwise_not(reg_loss_mask)] = 0
        label_sparse = gt_dict["label_sparse"]

        one_hot_label_sparse = self.get_one_hot(label_sparse, self.category_num)
        label_one_hot = np.zeros(self.label_one_hot_shape)
        label_one_hot[:, :, :, 0] = 1
        label_one_hot[allocation_mask] = one_hot_label_sparse

        if self.config.motion_state:
            motion_sparse = gt_dict["motion_state"]
            motion_one_hot_label_sparse = self.get_one_hot(motion_sparse, 3)
            motion_one_hot = np.zeros(self.label_one_hot_shape[:-1] + (3,))
            motion_one_hot[:, :, :, 0] = 1
            motion_one_hot[allocation_mask] = motion_one_hot_label_sparse
            motion_mask = motion_one_hot[:, :, :, 2] == 1

        if self.only_det:
            reg_target = reg_target[:, :, :, :1]
            reg_loss_mask = reg_loss_mask[:, :, :, :1]

        # only center for pred

        elif self.config.pred_type in ["motion", "center"]:
            reg_loss_mask = np.expand_dims(reg_loss_mask, axis=-1)
            reg_loss_mask = np.repeat(reg_loss_mask, self.box_code_size, axis=-1)
            reg_loss_mask[:, :, :, 1:, 2:] = False

        if self.config.use_map:
            if ("map_allocation_0" in gt_dict.keys()) or (
                "map_allocation" in gt_dict.keys()
            ):
                semantic_maps = []
                for m_id in range(self.config.map_channel):
                    map_alloc = gt_dict["map_allocation_" + str(m_id)]
                    map_sparse = gt_dict["map_sparse_" + str(m_id)]
                    recover = np.zeros(tuple(self.config.map_dims[:2]))
                    recover[map_alloc] = map_sparse
                    recover = np.rot90(recover, 3)
                    # recover_map = cv2.resize(recover,(self.config.map_dims[0],self.config.map_dims[1]))
                    semantic_maps.append(recover)
                semantic_maps = np.asarray(semantic_maps)
        else:
            semantic_maps = np.zeros(0)
        """
        if self.binary:
            reg_target = np.concatenate([reg_target[:,:,:2],reg_target[:,:,5:]],axis=2)
            reg_loss_mask = np.concatenate([reg_loss_mask[:,:,:2],reg_loss_mask[:,:,5:]],axis=2)
            label_one_hot = np.concatenate([label_one_hot[:,:,:2],label_one_hot[:,:,5:]],axis=2)

        """
        padded_voxel_points = list()

        for i in range(self.num_past_pcs):
            indices = gt_dict["voxel_indices_" + str(i)]
            curr_voxels = np.zeros(self.dims, dtype=bool)
            curr_voxels[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
            curr_voxels = np.rot90(curr_voxels, 3)
            padded_voxel_points.append(curr_voxels)
        padded_voxel_points = np.stack(padded_voxel_points, 0).astype(np.float32)
        anchors_map = self.anchors_map
        """
        if self.binary:
            anchors_map = np.concatenate([anchors_map[:,:,:2],anchors_map[:,:,5:]],axis=2)
        """
        if self.config.use_vis:
            vis_maps = np.zeros(
                (
                    self.num_past_pcs,
                    self.config.map_dims[-1],
                    self.config.map_dims[0],
                    self.config.map_dims[1],
                )
            )
            vis_free_indices = gt_dict["vis_free_indices"]
            vis_occupy_indices = gt_dict["vis_occupy_indices"]
            vis_maps[
                vis_occupy_indices[0, :],
                vis_occupy_indices[1, :],
                vis_occupy_indices[2, :],
                vis_occupy_indices[3, :],
            ] = math.log(0.7 / (1 - 0.7))
            vis_maps[
                vis_free_indices[0, :],
                vis_free_indices[1, :],
                vis_free_indices[2, :],
                vis_free_indices[3, :],
            ] = math.log(0.4 / (1 - 0.4))
            vis_maps = np.swapaxes(vis_maps, 2, 3)
            vis_maps = np.transpose(vis_maps, (0, 2, 3, 1))
            for v_id in range(vis_maps.shape[0]):
                vis_maps[v_id] = np.rot90(vis_maps[v_id], 3)
            vis_maps = vis_maps[-1]

        else:
            vis_maps = np.zeros(0)

        padded_voxel_points = padded_voxel_points.astype(np.float32)
        label_one_hot = label_one_hot.astype(np.float32)
        reg_target = reg_target.astype(np.float32)
        anchors_map = anchors_map.astype(np.float32)
        motion_one_hot = motion_one_hot.astype(np.float32)
        semantic_maps = semantic_maps.astype(np.float32)
        vis_maps = vis_maps.astype(np.float32)

        if self.val:
            return (
                padded_voxel_points,
                label_one_hot,
                reg_target,
                reg_loss_mask,
                anchors_map,
                motion_one_hot,
                motion_mask,
                vis_maps,
                [{"gt_box": gt_max_iou}],
                [seq_file],
            )
        else:
            return (
                padded_voxel_points,
                label_one_hot,
                reg_target,
                reg_loss_mask,
                anchors_map,
                motion_one_hot,
                motion_mask,
                vis_maps,
            )
