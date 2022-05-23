from nuscenes.nuscenes import NuScenes
import os
import numpy as np
import torch
import argparse

from coperception.utils.obj_util import *
from coperception.configs import Config, ConfigGlobal
from coperception.datasets import V2XSimDet
from coperception.utils.nuscenes_pc_util import get_instance_boxes_multisweep_sample_data


def get_gt_corners(config, data):
    voxel_size = config.voxel_size
    area_extents = config.area_extents
    pred_len = 1

    anchors_map = data["anchors_map"]
    reg_targets = data["reg_targets"]

    gt_max_iou_idx = data["gt_max_iou"]

    gt_corners = []
    for p in range(pred_len):

        for k in range(len(gt_max_iou_idx)):

            anchor = anchors_map[tuple(gt_max_iou_idx[k][:-1])]

            encode_box = reg_targets[tuple(gt_max_iou_idx[k][:-1]) + (p,)]
            if config.code_type[0] == "f":
                decode_box = bev_box_decode_np(encode_box, anchor)
                decode_corner = center_to_corner_box2d(
                    np.asarray([decode_box[:2]]),
                    np.asarray([decode_box[2:4]]),
                    np.asarray([decode_box[4:]]),
                )[0]
            elif config.code_type[0] == "c":
                # decoded_corner = (encode_box + anchor).reshape(-1, 4, 2)
                pass

            corner = coor_to_vis(
                decode_corner, area_extents=area_extents, voxel_size=voxel_size
            )
            # print(corner)
            gt_corners.append(
                (
                    min(corner[:, 0]),
                    255 - min(corner[:, 1]),
                    max(corner[:, 0]) - min(corner[:, 0]),
                    max(corner[:, 1] - min(corner[:, 1])),
                )
            )

    return gt_corners


def check_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name, exist_ok=True)
    return folder_name


# ---------------------- Extract the scenes, and then pre-process them into BEV maps ----------------------
def create_data(
    config, nusc, current_agent, config_global, scene_begin, scene_end, data_root
):
    # if current_agent != 1:
    # return
    print("current_agent", current_agent)
    channel = "LIDAR_TOP_id_" + str(current_agent)
    res_scenes = range(100)

    valdataset = V2XSimDet(
        dataset_roots=[f"{data_root}/agent{i}" for i in range(args.num_agent)],
        config=config,
        config_global=config_global,
        split="val",
        val=True,
    )
    file2id = {}
    for idx, file in enumerate(valdataset.seq_files[current_agent]):
        file2id[file] = idx

    for scene_idx in res_scenes[scene_begin:scene_end]:
        curr_scene = nusc.scene[scene_idx]

        first_sample_token = curr_scene["first_sample_token"]
        curr_sample = nusc.get("sample", first_sample_token)

        instance2id = {}

        save_dir = f"./TrackEval/data/gt/mot_challenge/V2X-{config.split}{current_agent}/{scene_idx}"
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "seqinfo.ini"), "w") as f:
            f.write(
                "\n".join(
                    [
                        "[Sequence]",
                        f"name={scene_idx}",
                        "imDir=img1",
                        "frameRate=5",
                        "seqLength=100",
                        "imWidth=255",
                        "imHeight=255",
                        "imExt=.jpg",
                    ]
                )
            )

        gt_dir = os.path.join(save_dir, "gt")
        os.makedirs(gt_dir, exist_ok=True)
        f = open(os.path.join(gt_dir, "gt.txt"), "w")
        frame = 0
        while True:
            if channel in curr_sample["data"]:
                sample_data = nusc.get("sample_data", curr_sample["data"][channel])
            else:
                break
            anns = curr_sample["anns"]
            target_path = f"{data_root}/agent{current_agent}/{scene_idx}_{frame}/0.npy"
            print("processing:", target_path)
            (
                _,
                _,
                _,
                reg_target_list,
                _,
                anchors_map_list,
                _,
                gt_max_iou,
                _,
                _,
                _,
                _,
            ) = zip(*valdataset[file2id[target_path]])
            reg_target_list = [torch.Tensor(x) for x in reg_target_list]
            anchors_map_list = [torch.Tensor(x) for x in anchors_map_list]
            reg_target = torch.stack(tuple(reg_target_list), 0)
            anchors_map = torch.stack(tuple(anchors_map_list), 0)
            tmp = [[]]
            for k in range(len(gt_max_iou[0])):
                tmp[0].append(
                    {"gt_box": torch.Tensor(gt_max_iou[0][k]["gt_box"]).unsqueeze(0)}
                )
            data = {
                "reg_targets": torch.unsqueeze(reg_target[current_agent, ...], 0)
                .detach()
                .numpy()[0],
                "anchors_map": torch.unsqueeze(anchors_map[current_agent, ...], 0)
                .detach()
                .numpy()[0],
                "gt_max_iou": torch.Tensor(
                    gt_max_iou[current_agent][0]["gt_box"]
                ).int(),
            }

            gt = get_gt_corners(config, data)
            num_instance = 0
            for ann in anns:
                ann = nusc.get("sample_annotation", ann)
                instance_token = ann["instance_token"]

                # ====== valid category ======
                categ = False
                for c, v in config.class_map.items():
                    if ann["category_name"].startswith(c):
                        categ = v
                        break
                if categ != 1:  # only track car
                    continue

                (
                    instance_boxes,
                    _,
                    _,
                    _,
                ) = get_instance_boxes_multisweep_sample_data(
                    nusc,
                    sample_data,
                    instance_token,
                    nsweeps_back=config.nsweeps_back,
                    nsweeps_forward=config.nsweeps_forward,
                )

                box_data = np.zeros((len(instance_boxes), 3 + 3 + 4), dtype=np.float32)
                box_data.fill(np.nan)
                for r, box in enumerate(instance_boxes):
                    if box is not None:
                        row = np.concatenate(
                            [box.center, box.wlh, box.orientation.elements]
                        )
                        box_data[r] = row[:]
                if np.max(np.abs(box_data[0, :2])) > (
                    np.max(config.area_extents[:, 1])
                    + (np.max(config.anchor_size[:, :2] / 2.0))
                ):
                    continue
                # ============================

                if instance_token not in instance2id:
                    instance2id[instance_token] = len(instance2id)

                bbox = gt[num_instance]
                f.write(
                    ",".join(
                        [  # align to MOT benchmark
                            str(frame + 1),  # frame id
                            str(instance2id[instance_token]),  # instance id
                            str(bbox[0]),  # instance bbox
                            str(bbox[1]),
                            str(bbox[2]),
                            str(bbox[3]),
                            "-1",
                            "-1",
                            "-1",
                            "-1",  # <conf> <x, y, z>
                        ]
                    )
                    + "\n"
                )
                f.flush()
                num_instance += 1

                if num_instance >= len(gt):
                    break

            frame += 1
            assert (
                len(gt) == num_instance
            ), f"len gt: {len(gt)}, num instance: {num_instance}"
            if curr_sample["next"] == "":
                break
            curr_sample = nusc.get("sample", curr_sample["next"])
        f.close()
    print("total instance:", len(instance2id))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root", type=str, help="Root path to nuScenes dataset")
    parser.add_argument("-d", "--data", type=str)
    parser.add_argument(
        "-s", "--split", type=str, help="The data split [train/val/test]"
    )
    parser.add_argument("-b", "--scene_begin", type=int, help="scene_begin")
    parser.add_argument("-e", "--scene_end", type=int, help="scene_end")
    parser.add_argument(
        "-p",
        "--savepath",
        default="./dataset/",
        type=str,
        help="Directory for saving the generated data",
    )
    parser.add_argument(
        "--num_agent", default=6, type=int, help="The total number of agents"
    )
    parser.add_argument("--current_agent", type=int, help="Index of current agent")
    args = parser.parse_args()

    nusc = NuScenes(version="v1.0-mini", dataroot=args.root, verbose=True)
    print("Total number of scenes:", len(nusc.scene))
    scene_begin = args.scene_begin
    scene_end = args.scene_end

    current_agent = args.current_agent
    savepath = check_folder(args.savepath + args.split + "/agent" + str(current_agent))
    config = Config(
        args.split, True, savepath=savepath, is_cross_road=current_agent == 0
    )
    config_global = ConfigGlobal(args.split, True, savepath=savepath)
    create_data(
        config, nusc, current_agent, config_global, scene_begin, scene_end, args.data
    )
