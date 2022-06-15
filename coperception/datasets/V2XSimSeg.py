import os
from multiprocessing import Manager

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset


class V2XSimSeg(Dataset):
    def __init__(
        self,
        dataset_roots=None,
        config=None,
        split=None,
        cache_size=1000,
        val=False,
        com=False,
        bound=None,
        kd_flag=False,
        rsu=False,
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
        self.pred_len = config.pred_len
        self.val = val
        self.config = config
        self.use_vis = config.use_vis
        self.com = com
        self.bound = bound
        self.kd_flag = kd_flag
        self.rsu = rsu

        if dataset_roots is None:
            raise ValueError(
                "The {} dataset root is None. Should specify its value.".format(
                    self.split
                )
            )
        self.dataset_roots = dataset_roots
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

        self.num_agent = len(self.dataset_roots)

        self.num_sample_seqs = len(self.seq_files[0])
        print("The number of {} sequences: {}".format(self.split, self.num_sample_seqs))
        # object information
        self.dims = config.map_dims
        self.num_past_pcs = config.num_past_pcs
        manager = Manager()
        self.cache = [manager.dict() for i in range(self.num_agent)]
        self.cache_size = cache_size if split == "train" else 0

        self.transform = Transform(self.split)

    def __len__(self):
        return self.num_sample_seqs

    def get_one_hot(self, label, category_num):
        one_hot_label = np.zeros((label.shape[0], category_num))
        for i in range(label.shape[0]):
            one_hot_label[i][label[i]] = 1

        return one_hot_label

    def get_seginfo_from_single_agent(self, agent_id, idx):
        empty_flag = False
        if idx in self.cache[agent_id]:
            gt_dict = self.cache[agent_id][idx]
        else:
            seq_file = self.seq_files[agent_id][idx]
            gt_data_handle = np.load(seq_file, allow_pickle=True)
            if gt_data_handle == 0:
                empty_flag = True
                if self.com != 'lowerbound' and self.com != 'upperbound':
                    return (
                        torch.zeros((256, 256, 13)).bool(),
                        torch.zeros((256, 256, 13)).bool(),
                        torch.zeros((256, 256)).int(),
                        torch.zeros((self.num_agent, 4, 4)),
                        0,
                        0,
                    )
                else:
                    return (
                        torch.zeros((256, 256, 13)).bool(),
                        torch.zeros((256, 256, 13)).bool(),
                        torch.zeros((256, 256)).int(),
                    )
            else:
                gt_dict = gt_data_handle.item()
                if len(self.cache[agent_id]) < self.cache_size:
                    self.cache[agent_id][idx] = gt_dict

        if not empty_flag:
            bev_seg = gt_dict["bev_seg"].astype(np.int32)

            padded_voxel_points = list()

            # if self.bound == 'lowerbound':
            for i in range(self.num_past_pcs):
                indices = gt_dict["voxel_indices_" + str(i)]
                curr_voxels = np.zeros(self.dims, dtype=bool)
                curr_voxels[indices[:, 0], indices[:, 1], indices[:, 2]] = 1

                curr_voxels = np.rot90(curr_voxels, 3)
                # curr_voxels = np.rot90(np.fliplr(curr_voxels), 3)
                bev_seg = np.rot90(bev_seg, 1)  # to align with voxel

                padded_voxel_points.append(curr_voxels)
            padded_voxel_points = np.stack(padded_voxel_points, 0)
            padded_voxel_points = np.squeeze(padded_voxel_points, 0)

            padded_voxel_points_teacher = list()
            # if self.bound == 'upperbound' or self.kd_flag:
            if self.rsu:
                indices_teacher = gt_dict["voxel_indices_teacher"]
            else:
                indices_teacher = gt_dict["voxel_indices_teacher_no_cross_road"]

            curr_voxels_teacher = np.zeros(self.dims, dtype=bool)
            curr_voxels_teacher[
                indices_teacher[:, 0], indices_teacher[:, 1], indices_teacher[:, 2]
            ] = 1
            curr_voxels_teacher = np.rot90(curr_voxels_teacher, 3)
            padded_voxel_points_teacher.append(curr_voxels_teacher)
            padded_voxel_points_teacher = np.stack(padded_voxel_points_teacher, 0)
            padded_voxel_points_teacher = np.squeeze(padded_voxel_points_teacher, 0)

            if self.com != 'lowerbound' and self.com != 'upperbound':
                if self.rsu:
                    trans_matrices = gt_dict["trans_matrices"]
                else:
                    trans_matrices = gt_dict["trans_matrices_no_cross_road"]

                target_agent_id = gt_dict["target_agent_id"]
                num_sensor = gt_dict["num_sensor"]

                return (
                    torch.from_numpy(padded_voxel_points),
                    torch.from_numpy(padded_voxel_points_teacher),
                    torch.from_numpy(bev_seg.copy()),
                    torch.from_numpy(trans_matrices.copy()),
                    target_agent_id,
                    num_sensor,
                )
            else:
                return (
                    torch.from_numpy(padded_voxel_points),
                    torch.from_numpy(padded_voxel_points_teacher),
                    torch.from_numpy(bev_seg.copy()),
                )

    def __getitem__(self, idx):
        res = []
        for i in range(self.num_agent):
            res.append(self.get_seginfo_from_single_agent(i, idx))
        return res


class Transform:
    def __init__(self, split):
        self.totensor = transforms.ToTensor()
        self.resize = transforms.Resize((256, 256))
        self.split = split

    def __call__(self, img, label):
        img = self.totensor(img.copy())
        label = self.totensor(label.copy())

        if self.split != "train":
            return img.permute(1, 2, 0).float(), label.squeeze(0).int()

        crop = transforms.RandomResizedCrop(256)
        params = crop.get_params(img, scale=(0.08, 1.0), ratio=(0.75, 1.33))
        img = TF.crop(img, *params)
        label = TF.crop(label, *params)

        if np.random.random() > 0.5:
            img = TF.hflip(img)
            label = TF.hflip(label)

        if np.random.random() > 0.5:
            img = TF.vflip(img)
            label = TF.vflip(label)

        img = self.resize(img)
        label = cv2.resize(
            label.squeeze(0).numpy(), dsize=(256, 256), interpolation=cv2.INTER_NEAREST
        )  # Resize provided by pytorch will have some random noise
        # return img.permute(1, 2, 0).float(), label.squeeze(0).int()
        return img.permute(1, 2, 0).float(), label
