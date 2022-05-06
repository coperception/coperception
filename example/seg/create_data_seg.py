# Copyright (c) 2020 Mitsubishi Electric Research Laboratories (MERL). All rights reserved. The software, documentation and/or data in this file is provided on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications. MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for any particular purpose. In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages. As more fully described in the license agreement that was required in order to download this software, documentation and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, research and non-commercial purposes.
import argparse
import os

import cv2

from coperception.configs import Config, ConfigGlobal
from coperception.utils.nuscenes_pc_util import (
    from_file_multisweep_upperbound_sample_data,
    from_file_multisweep_warp2com_sample_data,
)
from coperception.utils.data_util import voxelize_occupy
from coperception.utils.obj_util import *

from nuscenes import NuScenes


def check_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name, exist_ok=True)
    return folder_name


# ---------------------- Extract the scenes, and then pre-process them into BEV maps ----------------------
def create_data(config, nusc, current_agent, config_global, scene_begin, scene_end):
    channel = "LIDAR_TOP_id_" + str(current_agent)
    channel_bev = "BEV_TOP_id_" + str(current_agent)
    total_sample = 0

    res_scenes = range(100)
    for scene_idx in res_scenes[scene_begin:scene_end]:
        curr_scene = nusc.scene[scene_idx]
        first_sample_token = curr_scene["first_sample_token"]
        curr_sample = nusc.get("sample", first_sample_token)

        # Iterate each sample data
        print("Processing scene {} of agent {} ...".format(scene_idx, current_agent))

        # Added by Yiming, make sure the agent_num equals maximum num (5)
        channel_flag = True
        if channel in curr_sample["data"]:
            curr_sample_data = nusc.get("sample_data", curr_sample["data"][channel])
            curr_sample_data_bev = nusc.get(
                "sample_data", curr_sample["data"][channel_bev]
            )
            save_seq_cnt = 0  # only used for save data file name
        else:
            channel_flag = False
            for num_sample in range(100):
                save_directory = check_folder(
                    os.path.join(
                        config.savepath, str(scene_idx) + "_" + str(num_sample)
                    )
                )
                save_file_name = os.path.join(save_directory, "0.npy")
                np.save(save_file_name, 0)
                print("  >> Finish sample: {}".format(num_sample))
        # -------------------------------------------------------------------------------------------------
        # ------------------------ We only calculate non-empty channels -----------------------------------
        # -------------------------------------------------------------------------------------------------
        while channel_flag:
            t = time.time()

            (
                all_pc_single_view,
                all_times,
                trans_matrices,
                trans_matrices_no_cross_road,
                target_agent_id,
                num_sensor,
            ) = from_file_multisweep_warp2com_sample_data(
                current_agent, nusc, curr_sample_data, return_trans_matrix=True
            )
            (all_pc_teacher, _,) = from_file_multisweep_upperbound_sample_data(
                nusc, curr_sample_data, return_trans_matrix=False
            )
            (
                all_pc_teacher_no_cross_road,
                _,
            ) = from_file_multisweep_upperbound_sample_data(
                nusc, curr_sample_data, return_trans_matrix=False, no_cross_road=True
            )

            # Store point cloud of each sweep
            pc_single = all_pc_single_view.points
            pc_teacher = all_pc_teacher.points
            pc_teacher_no_cross_road = all_pc_teacher_no_cross_road.points

            # Store semantics bev of each agent
            bev_data_path, _, _ = nusc.get_sample_data(curr_sample_data_bev["token"])
            bev_pix = np.load(bev_data_path, allow_pickle=True)

            # some data comes as .npz format
            if type(bev_pix) == np.lib.npyio.NpzFile:
                bev_pix = bev_pix["arr_0"]

            bev_pix = cv2.resize(
                bev_pix, dsize=(256, 256), interpolation=cv2.INTER_NEAREST
            )
            # plt.imshow(bev_pix)
            # plt.show()

            # bev_pic = bev_pic.resize((256, 256), Image.ANTIALIAS)
            # bev_pix = np.array(bev_pic)[:,:,0]
            new_bev_pix = np.zeros(bev_pix.shape)

            for key, value in config.classes_remap.items():
                new_bev_pix[np.where(bev_pix == key)] = value

            # print(np.unique(bev_pix))

            # Prepare data dictionary for the next step (ie, generating BEV maps)
            save_data_dict = dict()
            save_data_dict["pc_single"] = pc_single
            save_data_dict["pc_teacher"] = pc_teacher
            save_data_dict["pc_teacher_no_cross_road"] = pc_teacher_no_cross_road
            save_data_dict["bev_seg"] = new_bev_pix.astype(np.uint8)
            # Now we generate dense and sparse BEV maps
            seq_idx = 0
            dense_bev_data_single = convert_to_dense_bev(
                save_data_dict, config, "pc_single"
            )
            dense_bev_data_teacher = convert_to_dense_bev(
                save_data_dict, config, "pc_teacher"
            )
            dense_bev_data_teacher_no_cross_road = convert_to_dense_bev(
                save_data_dict, config, "pc_teacher_no_cross_road"
            )

            sparse_bev_data = convert_to_sparse_bev(config, dense_bev_data_single)
            convert_to_sparse_bev_teacher(
                config, dense_bev_data_teacher, sparse_bev_data, "voxel_indices_teacher"
            )
            convert_to_sparse_bev_teacher(
                config,
                dense_bev_data_teacher_no_cross_road,
                sparse_bev_data,
                "voxel_indices_teacher_no_cross_road",
            )

            sparse_bev_data["trans_matrices"] = trans_matrices
            sparse_bev_data[
                "trans_matrices_no_cross_road"
            ] = trans_matrices_no_cross_road
            sparse_bev_data["target_agent_id"] = target_agent_id
            sparse_bev_data["num_sensor"] = num_sensor

            # save the data
            save_directory = check_folder(
                os.path.join(config.savepath, str(scene_idx) + "_" + str(save_seq_cnt))
            )
            save_file_name = os.path.join(save_directory, str(seq_idx) + ".npy")
            np.save(save_file_name, arr=sparse_bev_data)
            total_sample += 1
            print(
                "  >> Finish sample: {}, sequence {} takes {} s".format(
                    save_seq_cnt, seq_idx, time.time() - t
                )
            )

            save_seq_cnt += 1
            # Skip some keyframes if necessary
            flag = False
            for _ in range(config.num_keyframe_skipped + 1):
                if curr_sample["next"] != "":
                    curr_sample = nusc.get("sample", curr_sample["next"])
                else:
                    flag = True
                    break

            if flag:  # No more keyframes
                break
            else:
                curr_sample_data = nusc.get("sample_data", curr_sample["data"][channel])
                curr_sample_data_bev = nusc.get(
                    "sample_data", curr_sample["data"][channel_bev]
                )


# ----------------------------------------------------------------------------------------
# ---------------------- Convert the raw data into (dense) BEV maps ----------------------
# ----------------------------------------------------------------------------------------
def convert_to_dense_bev(seq_data_dict, config, pc_category):
    data_dict = seq_data_dict
    pc_all = data_dict[pc_category]
    pc_all = pc_all.T

    bev_seg = data_dict["bev_seg"]

    # Discretize the input point clouds, and compute the ground-truth displacement vectors
    # The following two variables contain the information for the
    # compact representation of binary voxels, as described in the paper
    voxel_indices_list = list()
    padded_voxel_points_list = list()
    res, voxel_indices = voxelize_occupy(
        pc_all,
        voxel_size=config.voxel_size,
        extents=config.area_extents,
        return_indices=True,
    )
    voxel_indices_list.append(voxel_indices)
    padded_voxel_points_list.append(res)

    # Compile the batch of voxels, so that they can be fed into the network.
    # Note that, the padded_voxel_points in this script will only be used for sanity check.
    padded_voxel_points = np.stack(padded_voxel_points_list, axis=0).astype(bool)

    return voxel_indices_list, padded_voxel_points, bev_seg


# ---------------------- Convert the dense BEV data into sparse format ----------------------
# This will significantly reduce the space used for data storage
def convert_to_sparse_bev(config, dense_bev_data):
    save_voxel_indices_list, save_voxel_points, bev_seg = dense_bev_data
    save_voxel_dims = save_voxel_points.shape[1:]
    save_data_dict = dict()

    save_data_dict["bev_seg"] = bev_seg

    for i in range(len(save_voxel_indices_list)):
        save_data_dict["voxel_indices_" + str(i)] = save_voxel_indices_list[i].astype(
            np.int32
        )

    # -------------------------------- Sanity Check --------------------------------
    for i in range(len(save_voxel_indices_list)):
        indices = save_data_dict["voxel_indices_" + str(i)]
        curr_voxels = np.zeros(save_voxel_dims, dtype=bool)
        curr_voxels[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
        assert np.all(curr_voxels == save_voxel_points[i]), "Error: Mismatch"

    return save_data_dict


def convert_to_sparse_bev_teacher(config, dense_bev_data, save_data_dict, key_name):
    save_voxel_indices_list, save_voxel_points, bev_seg = dense_bev_data
    save_voxel_dims = save_voxel_points.shape[1:]

    for i in range(len(save_voxel_indices_list)):
        save_data_dict[key_name] = save_voxel_indices_list[i].astype(np.int32)

    # -------------------------------- Sanity Check --------------------------------
    for i in range(len(save_voxel_indices_list)):
        indices = save_data_dict[key_name]
        curr_voxels = np.zeros(save_voxel_dims, dtype=bool)
        curr_voxels[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
        assert np.all(curr_voxels == save_voxel_points[i]), "Error: Mismatch"

    return save_data_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument('-r', '--root', default='/data1/yimingli/carScene-mini', type=str, help='Root path to nuScenes dataset')
    parser.add_argument(
        "-r",
        "--root",
        default="/home/dekunma/CS/ai4ce/V2X-Sim-2",
        type=str,
        help="Root path to nuScenes dataset",
    )
    parser.add_argument(
        "-s",
        "--split",
        default="train",
        type=str,
        help="The data split [train/val/test]",
    )
    parser.add_argument(
        "-b", "--scene_begin", default=0, type=int, help="Index of begining scene"
    )
    parser.add_argument(
        "-e", "--scene_end", default=1, type=int, help="Index of end scene + 1"
    )
    parser.add_argument(
        "-p",
        "--savepath",
        default="/home/dekunma/CS/ai4ce/V2X-seg-2",
        type=str,
        help="Directory for saving the generated data",
    )
    parser.add_argument(
        "--from_agent", default=0, type=int, help="start from which agent"
    )
    parser.add_argument(
        "--to_agent", default=6, type=int, help="until which agent (index + 1)"
    )

    # parser.add_argument('-m', '--mode', default='upperbound', type=str, choices=['upperbound', 'lowerbound'])
    args = parser.parse_args()

    nusc = NuScenes(version="v1.0-mini", dataroot=args.root, verbose=True)
    print("Total number of scenes:", len(nusc.scene))
    scene_begin = args.scene_begin
    scene_end = args.scene_end

    root = os.path.join(args.savepath, args.split)
    for current_agent in range(args.from_agent, args.to_agent):
        savepath = check_folder(os.path.join(root, "agent" + str(current_agent)))
        config = Config(
            args.split, True, savepath=savepath, is_cross_road=current_agent == 0
        )
        config_global = ConfigGlobal(args.split, True, savepath=savepath)
        create_data(config, nusc, current_agent, config_global, scene_begin, scene_end)
