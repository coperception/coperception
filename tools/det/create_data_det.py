# Copyright (c) 2020 Mitsubishi Electric Research Laboratories (MERL). All rights reserved. The software, documentation and/or data in this file is provided on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications. MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for any particular purpose. In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages. As more fully described in the license agreement that was required in order to download this software, documentation and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, research and non-commercial purposes.
import argparse
import os

from coperception.utils import mapping
from coperception.configs import Config
from coperception.utils.data_util import voxelize_occupy
from coperception.utils.obj_util import *
from coperception.utils.nuscenes_pc_util import (
    from_file_multisweep_upperbound_sample_data,
    from_file_multisweep_warp2com_sample_data,
    get_instance_boxes_multisweep_sample_data
)
from coperception.utils.v2x_sim_scene_split.parser import parse_scene_files
from nuscenes import NuScenes as CoPerceptionDataset

def check_folder(folder_name):
    os.makedirs(folder_name, exist_ok=True)
    return folder_name


# ---------------------- Extract the scenes, and then pre-process them into BEV maps ----------------------
def create_data(coperception_dataset, current_agent, scene_begin, scene_end, scene_splits, args_savepath):
    channel = "LIDAR_TOP_id_" + str(current_agent)
    total_sample = 0
    res_scenes = range(100)

    config = Config(
        'SPLIT', True, is_cross_road=current_agent == 0
    )

    for scene_idx in res_scenes[scene_begin:scene_end]:
        curr_scene = coperception_dataset.scene[scene_idx]
        split = None
        
        for s in ['train', 'val', 'test']:
            if scene_idx in scene_splits[s]:
                split = s
                config.split = s
        
        if not split:
            raise Exception(f'There is no scene {scene_idx} in the dataset.')

        savepath = os.path.join(args_savepath, split, "agent" + str(current_agent))
        os.makedirs(savepath, exist_ok=True)

        first_sample_token = curr_scene["first_sample_token"]
        curr_sample = coperception_dataset.get("sample", first_sample_token)

        # Iterate each sample data
        print("Processing scene {} of agent {} ...".format(scene_idx, current_agent))

        # Added by Yiming, make sure the agent_num equals maximum num (5)
        channel_flag = True
        if channel in curr_sample["data"]:
            curr_sample_data = coperception_dataset.get("sample_data", curr_sample["data"][channel])
            # for storing consecutive sequences; the data consists of timestamps, points, etc
            save_data_dict_list = list()

            # for storing box annotations in consecutive sequences
            save_box_dict_list = list()
            save_instance_token_list = list()

            adj_seq_cnt = 0
            save_seq_cnt = 0  # only used for save data file name
        else:
            channel_flag = False
            for num_sample in range(100):
                save_directory = check_folder(
                    os.path.join(
                        savepath, str(scene_idx) + "_" + str(num_sample)
                    )
                )
                save_file_name = os.path.join(save_directory, "0.npy")
                np.save(save_file_name, 0)
                print("  >> Finish sample: {}".format(num_sample))

        # -------------------------------------------------------------------------------------------------
        # ------------------------ We only calculate non-empty channels -----------------------------------
        # -------------------------------------------------------------------------------------------------
        while channel_flag:

            (all_pc_teacher, _,) = from_file_multisweep_upperbound_sample_data(
                coperception_dataset, curr_sample_data, return_trans_matrix=False
            )

            (
                all_pc_teacher_no_cross_road,
                _,
            ) = from_file_multisweep_upperbound_sample_data(
                coperception_dataset, curr_sample_data, return_trans_matrix=False, no_cross_road=True
            )

            # Get the synchronized point clouds
            (
                all_pc,
                all_times,
                trans_matrices,
                trans_matrices_no_cross_road,
                target_agent_id,
                num_sensor,
            ) = from_file_multisweep_warp2com_sample_data(
                current_agent, coperception_dataset, curr_sample_data, return_trans_matrix=True
            )

            assert (
                target_agent_id == current_agent
            ), "The target_agent_id mismatches the input agent id."

            # we only store one agent info in when2com
            save_data_dict_all_agents_list = list()

            # Store point cloud of each sweep
            pc = all_pc.points

            _, sort_idx = np.unique(all_times, return_index=True)

            # Preserve the item order in unique_times
            unique_times = all_times[np.sort(sort_idx)]
            num_sweeps = len(unique_times)

            # Prepare data dictionary for the next step (ie, generating BEV maps)
            save_data_dict = dict()

            # for remapping the instance ids, according to class_map
            box_data_dict = dict()
            curr_token_list = list()

            for tid in range(num_sweeps):
                _time = unique_times[tid]
                points_idx = np.where(all_times == _time)[0]
                _pc = pc[:, points_idx]
                save_data_dict["pc_" + str(tid)] = _pc

            save_data_dict["times"] = unique_times
            save_data_dict["num_sweeps"] = num_sweeps
            save_data_dict["trans_matrices"] = trans_matrices
            save_data_dict[
                "trans_matrices_no_cross_road"
            ] = trans_matrices_no_cross_road
            save_data_dict["num_sensor"] = num_sensor
            save_data_dict["target_agent_id"] = target_agent_id
            save_data_dict["all_pc_teacher"] = all_pc_teacher.points
            save_data_dict[
                "all_pc_teacher_no_cross_road"
            ] = all_pc_teacher_no_cross_road.points

            save_data_dict_all_agents_list.append(save_data_dict)
            # -------------------------------------------------------------------------------------------------
            # ------------------------ Now we get the synchronized bounding boxes -----------------------------
            # -------------------------------------------------------------------------------------------------
            # First, we need to iterate all the instances, and then retrieve their corresponding bounding boxes
            num_instances = 0  # The number of instances within this sample
            corresponding_sample_token = curr_sample_data["sample_token"]
            corresponding_sample_rec = coperception_dataset.get("sample", corresponding_sample_token)

            for ann_token in corresponding_sample_rec["anns"]:
                ann_rec = coperception_dataset.get("sample_annotation", ann_token)
                category_name = ann_rec["category_name"]
                instance_token = ann_rec["instance_token"]

                flag = False
                for c, v in config.class_map.items():
                    if category_name.startswith(c):
                        box_data_dict["category_" + instance_token] = v
                        flag = True
                        break
                if not flag:
                    box_data_dict["category_" + instance_token] = 4  # Other category

                (
                    instance_boxes,
                    instance_all_times,
                    _,
                    _,
                ) = get_instance_boxes_multisweep_sample_data(
                    coperception_dataset,
                    curr_sample_data,
                    instance_token,
                    nsweeps_back=config.nsweeps_back,
                    nsweeps_forward=config.nsweeps_forward,
                )

                assert np.array_equal(
                    unique_times, instance_all_times
                ), "The sweep and instance times are inconsistent!"
                assert num_sweeps == len(
                    instance_boxes
                ), "The number of instance boxes does not match that of sweeps!"

                # Each row corresponds to a box annotation; the column consists of box center, box size, and quaternion
                box_data = np.zeros((len(instance_boxes), 3 + 3 + 4), dtype=np.float32)
                box_data.fill(np.nan)
                for r, box in enumerate(instance_boxes):
                    if box is not None:
                        row = np.concatenate(
                            [box.center, box.wlh, box.orientation.elements]
                        )

                        box_data[r] = row[:]

                # Save the box data for current instance
                box_data_dict["instance_boxes_" + instance_token] = box_data
                num_instances += 1

                curr_token_list.append(instance_token)

            save_data_dict_list.append(save_data_dict_all_agents_list)

            box_data_dict["num_instances"] = num_instances
            save_box_dict_list.append(box_data_dict)
            save_instance_token_list.append(curr_token_list)

            # -------------------------------------------------------------------------------------------------
            # ------------------- Now we generate sparse BEV map and reorganize gt_box ------------------------
            # -------------------------------------------------------------------------------------------------
            # Update the counter and save the data if desired (But here we do not want to
            # save the data to disk since it would cost about 2TB space)
            adj_seq_cnt += 1
            if adj_seq_cnt == config.num_adj_seqs:

                # First, we need to reorganize the instance tokens (ids)
                if config.binary:
                    flag = False

                # local box retrieving
                curr_save_box_dict = save_box_dict_list[0]
                gt_box_dict = dict()
                for index, token in enumerate(save_instance_token_list[0]):
                    box_info = curr_save_box_dict["instance_boxes_" + token]
                    box_cat = curr_save_box_dict["category_" + token]

                    gt_box_dict["instance_boxes_" + str(index)] = box_info
                    gt_box_dict["category_" + str(index)] = box_cat

                gt_box_dict["num_instances"] = curr_save_box_dict["num_instances"]

                save_data_dict_list[0].append(gt_box_dict)

                # Now we generate dense and sparse BEV maps
                for seq_idx, seq_data_dict in enumerate(save_data_dict_list):
                    t = time.time()
                    dense_bev_data = convert_to_dense_bev(seq_data_dict, config)
                    if config.binary and dense_bev_data is None:
                        continue
                    sparse_bev_data = convert_to_sparse_bev(
                        config, dense_bev_data, config.motion_state
                    )

                    # save the data
                    save_directory = check_folder(
                        os.path.join(
                            savepath, str(scene_idx) + "_" + str(save_seq_cnt)
                        )
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
                adj_seq_cnt = 0
                save_data_dict_list = list()
                save_box_dict_list = list()
                save_instance_token_list = list()

                # Skip some keyframes if necessary
                flag = False
                for _ in range(config.num_keyframe_skipped + 1):
                    if curr_sample["next"] != "":
                        curr_sample = coperception_dataset.get("sample", curr_sample["next"])
                    else:
                        flag = True
                        break

                if flag:  # No more keyframes
                    break
                else:
                    curr_sample_data = coperception_dataset.get(
                        "sample_data", curr_sample["data"][channel]
                    )
            else:
                flag = False
                for _ in range(config.skip_frame + 1):
                    if curr_sample_data["next"] != "":
                        curr_sample_data = coperception_dataset.get(
                            "sample_data", curr_sample_data["next"]
                        )
                    else:
                        flag = True
                        break

                if flag:  # No more sample frames
                    break

            if flag:  # No more sample frames
                break


# ----------------------------------------------------------------------------------------
# ---------------------- Convert the raw data into (dense) BEV maps ----------------------
# ----------------------------------------------------------------------------------------
def convert_to_dense_bev(seq_data_dict, config):
    gt_dict = seq_data_dict[-1]
    data_dict = seq_data_dict[0]
    num_sensor = data_dict["num_sensor"]
    target_agent_id = data_dict["target_agent_id"]
    num_sweeps = data_dict["num_sweeps"]
    trans_matrices = data_dict["trans_matrices"]
    trans_matrices_warping = data_dict["trans_matrices"]
    trans_matrices_no_cross_road = data_dict["trans_matrices_no_cross_road"]

    assert num_sweeps == 1, "Currently we only consider single sweep."

    if config.binary:
        flag = False
        num_instances = gt_dict["num_instances"]
        for i in range(num_instances):
            category = gt_dict["category_" + str(i)]
            if config.binary:
                if category == 1:
                    flag = True
                    break
        if not flag:
            return None

    # Load point cloud
    pc_list = []
    for i in range(num_sweeps):
        pc = data_dict["pc_" + str(i)]
        pc_list.append(pc.T)

    # Discretize the input point clouds
    voxel_indices_list = list()
    padded_voxel_points_list = list()

    for i in range(num_sweeps):
        res, voxel_indices = voxelize_occupy(
            pc_list[i],
            voxel_size=config.voxel_size,
            extents=config.area_extents,
            return_indices=True,
        )
        voxel_indices_list.append(voxel_indices)
        padded_voxel_points_list.append(res)

    trans_matrices = trans_matrices[target_agent_id]
    origins = np.zeros((1, 4))
    origins[:, 3] = 1.0
    pc_range = [
        config.area_extents[0, 0],
        config.area_extents[1, 0],
        config.area_extents[2, 0],
        config.area_extents[0, 1],
        config.area_extents[1, 1],
        config.area_extents[2, 1],
    ]

    visibility_maps = []

    if config.is_cross_road:
        visibility_maps.append(np.zeros([256, 256, 13]))
    else:
        extents = config.area_extents
        for idx in range(origins.shape[0]):
            pts = pc_list[idx]

            filter_idx = np.where(
                (extents[0, 0] < pts[:, 0])
                & (pts[:, 0] < extents[0, 1])
                & (extents[1, 0] < pts[:, 1])
                & (pts[:, 1] < extents[1, 1])
                & (extents[2, 0] < pts[:, 2])
                & (pts[:, 2] < extents[2, 1])
            )[0]
            pts = pts[filter_idx]
            origins[idx] = trans_matrices[idx].dot(origins[idx].T).T
            visibility_maps.append(
                mapping.compute_logodds_dp(
                    pts,
                    origins[[idx], :3],
                    pc_range,
                    range(pts.shape[0]),
                    min(config.voxel_size),
                )
            )  # , lo_occupied, lo_free

    # Compile the batch of voxels, so that they can be fed into the network.
    # Note that, the padded_voxel_points in this script will only be used for sanity check.
    padded_voxel_points = np.stack(padded_voxel_points_list, axis=0).astype(bool)

    # -----------------------------------------------------------------
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # # Get the teacher's input
    pc_list = []
    for i in range(num_sweeps):
        pc = data_dict["all_pc_teacher"]
        pc_list.append(pc.T)

    voxel_indices_list_teacher = list()
    padded_voxel_points_list_teacher = list()

    for i in range(num_sweeps):
        res_teacher, voxel_indices_teacher = voxelize_occupy(
            pc_list[i],
            voxel_size=config.voxel_size,
            extents=config.area_extents,
            return_indices=True,
        )
        voxel_indices_list_teacher.append(voxel_indices_teacher)
        padded_voxel_points_list_teacher.append(res_teacher)

    # Get teacher's input (without data from cross road)
    pc_list = []
    for i in range(num_sweeps):
        pc = data_dict["all_pc_teacher_no_cross_road"]
        pc_list.append(pc.T)

    voxel_indices_list_teacher_no_cross_road = list()

    for i in range(num_sweeps):
        res_teacher, voxel_indices_teacher_no_cross_road = voxelize_occupy(
            pc_list[i],
            voxel_size=config.voxel_size,
            extents=config.area_extents,
            return_indices=True,
        )
        voxel_indices_list_teacher_no_cross_road.append(
            voxel_indices_teacher_no_cross_road
        )

    # Finally, generate the ground-truth displacement field
    (
        label,
        reg_target,
        allocation_map,
        gt_max_iou,
        reg_loss_mask,
        motion_state,
    ) = generate_object_detection_gt(
        gt_dict,
        config.voxel_size,
        config.area_extents,
        config.anchor_size,
        config.map_dims,
        config.pred_len,
        config.nsweeps_back,
        config.box_code_size,
        config.category_threshold,
        config,
    )

    if label is None:
        print("label is none")
        return None

    else:
        return (
            voxel_indices_list,
            voxel_indices_list_teacher,
            voxel_indices_list_teacher_no_cross_road,
            padded_voxel_points,
            trans_matrices_warping,
            trans_matrices_no_cross_road,
            label,
            reg_target,
            allocation_map,
            gt_max_iou,
            reg_loss_mask,
            motion_state,
            visibility_maps,
            target_agent_id,
            num_sensor,
        )


# ---------------------- Convert the dense BEV data into sparse format ----------------------
# This will significantly reduce the space used for data storage
def convert_to_sparse_bev(config, dense_bev_data, use_motion_state=False):
    (
        save_voxel_indices_list,
        voxel_indices_list_teacher,
        voxel_indices_list_teacher_no_cross_road,
        save_voxel_points,
        save_trans_matrices_warp,
        save_trans_matrices_no_cross_road,
        save_label,
        save_reg_target,
        save_allocation_mask,
        gt_max_iou,
        reg_loss_mask,
        motion_state,
        visibility_maps,
        target_agent_id,
        num_sensor,
    ) = dense_bev_data

    save_voxel_dims = save_voxel_points.shape[1:]
    save_label_sparse = save_label[save_allocation_mask]
    save_reg_target_sparse = save_reg_target[save_allocation_mask]

    if use_motion_state:
        save_motion_label_sparse = motion_state[save_allocation_mask]

    save_data_dict = dict()
    for i in range(len(save_voxel_indices_list)):
        save_data_dict["voxel_indices_" + str(i)] = save_voxel_indices_list[i].astype(
            np.int32
        )

    save_data_dict["reg_target_sparse"] = save_reg_target_sparse
    save_data_dict["label_sparse"] = save_label_sparse
    save_data_dict["allocation_mask"] = save_allocation_mask

    save_data_dict["gt_max_iou"] = gt_max_iou
    save_data_dict["reg_loss_mask"] = reg_loss_mask
    if use_motion_state:
        save_data_dict["motion_state"] = save_motion_label_sparse

    visibility_maps = np.asarray(visibility_maps)
    visibility_maps = visibility_maps.reshape(
        -1, config.map_dims[2], config.map_dims[0], config.map_dims[1]
    )

    vis_occupy_indices = np.asarray(np.where(visibility_maps > 0)).astype(np.uint8)
    vis_free_indices = np.asarray(np.where(visibility_maps < 0)).astype(np.uint8)

    save_data_dict["vis_occupy_indices"] = vis_occupy_indices
    save_data_dict["vis_free_indices"] = vis_free_indices
    save_data_dict["target_agent_id"] = target_agent_id
    save_data_dict["num_sensor"] = num_sensor
    save_data_dict["trans_matrices"] = save_trans_matrices_warp
    save_data_dict["trans_matrices_no_cross_road"] = save_trans_matrices_no_cross_road

    save_data_dict["voxel_indices_teacher"] = voxel_indices_list_teacher[0].astype(
        np.int32
    )
    save_data_dict[
        "voxel_indices_teacher_no_cross_road"
    ] = voxel_indices_list_teacher_no_cross_road[0].astype(np.int32)

    # -------------------------------- Sanity Check --------------------------------
    for i in range(len(save_voxel_indices_list)):
        indices = save_data_dict["voxel_indices_" + str(i)]
        curr_voxels = np.zeros(save_voxel_dims, dtype=bool)
        curr_voxels[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
        assert np.all(curr_voxels == save_voxel_points[i]), "Error: Mismatch"

    recover_label = np.zeros((save_label.shape)).astype(save_label.dtype)
    recover_label[save_allocation_mask] = save_label_sparse
    assert np.all(recover_label == save_label), "Error: Mismatch"

    recover_reg_target = np.zeros((save_reg_target.shape)).astype(save_reg_target.dtype)
    recover_reg_target[save_allocation_mask] = save_reg_target_sparse
    assert np.all(recover_reg_target == save_reg_target), "Error: Mismatch"

    if use_motion_state:
        recover_motion_state = np.zeros((motion_state.shape)).astype(motion_state.dtype)
        recover_motion_state[save_allocation_mask] = save_motion_label_sparse
        assert np.all(recover_motion_state == motion_state), "Error: Mismatch"

    recover = np.zeros_like(visibility_maps)
    recover[
        vis_occupy_indices[0, :],
        vis_occupy_indices[1, :],
        vis_occupy_indices[2, :],
        vis_occupy_indices[3, :],
    ] = math.log(0.7 / (1 - 0.7))
    recover[
        vis_free_indices[0, :],
        vis_free_indices[1, :],
        vis_free_indices[2, :],
        vis_free_indices[3, :],
    ] = math.log(0.4 / (1 - 0.4))
    assert np.all(recover == visibility_maps), "Visibility Sanity check fails"

    return save_data_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--root",
        default="/mnt/NAS/home/yiming/NeurIPS2021-DiscoNet/V2X-Sim-1.0-raw",
        type=str,
        help="Root path to CoPerception dataset",
    )
    parser.add_argument(
        "-c", "--current_agent", default=0, type=int, help="current_agent"
    )
    parser.add_argument("-b", "--scene_begin", default=0, type=int, help="scene_begin")
    parser.add_argument("-e", "--scene_end", default=100, type=int, help="scene_end")
    parser.add_argument(
        "-p",
        "--savepath",
        default="/mnt/NAS/home/yiming/NeurIPS2021-DiscoNet/V2X-Sim-1.0-trainval/",
        type=str,
        help="Directory for saving the generated data",
    )
    parser.add_argument(
        "--from_agent", default=0, type=int, help="start from which agent"
    )
    parser.add_argument(
        "--to_agent", default=6, type=int, help="until which agent (index + 1)"
    )
    args = parser.parse_args()

    coperception_dataset = CoPerceptionDataset(version="v1.0-mini", dataroot=args.root, verbose=True)
    print("Total number of scenes:", len(coperception_dataset.scene))
    scene_begin = args.scene_begin
    scene_end = args.scene_end

    scene_files_loc = '../../coperception/utils/v2x_sim_scene_split'
    scene_splits = parse_scene_files(scene_files_loc)

    print(f'Parsed files will be saved to {args.savepath}')

    for current_agent in range(args.from_agent, args.to_agent):
        create_data(coperception_dataset, current_agent, scene_begin, scene_end, scene_splits, args.savepath)
