# Copyright (c) 2020 Mitsubishi Electric Research Laboratories (MERL). All rights reserved. The software, documentation and/or data in this file is provided on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications. MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for any particular purpose. In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages. As more fully described in the license agreement that was required in order to download this software, documentation and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, research and non-commercial purposes.
import argparse
import os

import cv2
import torch.optim as optim
from tqdm import tqdm

from coperception.datasets import V2XSimSeg
from coperception.configs import Config
from coperception.utils.SegMetrics import ComputeIoU
from coperception.utils.SegModule import *
from coperception.utils.loss import *
from coperception.models.seg import *
from torch.utils.data import DataLoader
from coperception.utils.data_util import apply_pose_noise


def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path


@torch.no_grad()
def main(config, args):
    config.nepoch = args.nepoch
    batch_size = args.batch
    num_workers = args.nworker
    logpath = args.logpath
    pose_noise = args.pose_noise
    compress_level = args.compress_level
    only_v2i = args.only_v2i

    # Specify gpu device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = torch.cuda.device_count()
    print("device number", device_num)

    if args.bound == "upperbound":
        flag = "upperbound"
    else:
        if args.com == "when2com":
            flag = "when2com"
            if args.warp_flag:
                flag += "_warp"
            if args.inference == "argmax_test":
                flag = flag.replace("when2com", "who2com")
        elif args.com == "v2v":
            flag = "v2v"
        elif args.com == "mean":
            flag = "mean"
        elif args.com == "max":
            flag = "max"
        elif args.com == "sum":
            flag = "sum"
        elif args.com == "cat":
            flag = "cat"
        elif args.com == "agent":
            flag = "agent"
        elif args.com == "disco":
            flag = "disco"
        else:
            flag = "lowerbound"

    num_agent = args.num_agent
    agent_idx_range = range(1, num_agent) if args.no_cross_road else range(num_agent)
    # TODO: kd_flag
    valset = V2XSimSeg(
        dataset_roots=[args.data + "/agent%d" % i for i in agent_idx_range],
        config=config,
        split="val",
        val=True,
        com=args.com,
        bound=args.bound,
        no_cross_road=args.no_cross_road,
    )
    valloader = DataLoader(
        valset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    print("Validation dataset size:", len(valset))

    checkpoint = torch.load(args.resume)

    config.flag = flag
    config.com = args.com
    config.inference = args.inference
    config.split = "test"
    # build model
    if args.no_cross_road:
        num_agent -= 1
    if args.com.startswith("when2com") or args.com.startswith("who2com"):
        model = When2Com_UNet(
            config,
            in_channels=config.in_channels,
            n_classes=config.num_class,
            warp_flag=args.warp_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif args.com == "v2v":
        model = V2VNet(
            config.in_channels,
            config.num_class,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif args.com == "mean":
        model = MeanFusion(
            config.in_channels,
            config.num_class,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif args.com == "max":
        model = MaxFusion(
            config.in_channels,
            config.num_class,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif args.com == "sum":
        model = SumFusion(
            config.in_channels,
            config.num_class,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif args.com == "cat":
        model = CatFusion(
            config.in_channels,
            config.num_class,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif args.com == "agent":
        model = AgentWiseWeightedFusion(
            config.in_channels,
            config.num_class,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif args.com == "disco":
        model = DiscoNet(
            config.in_channels,
            config.num_class,
            num_agent=num_agent,
            kd_flag=False,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    else:
        model = UNet(
            config.in_channels,
            config.num_class,
            num_agent=num_agent,
            compress_level=compress_level,
        )
    # model = nn.DataParallel(model)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    segmodule = SegModule(model, model, config, optimizer, False)
    segmodule.model.load_state_dict(checkpoint["model_state_dict"])
    # ==== eval ====
    segmodule.model.eval()
    compute_iou = ComputeIoU(num_class=config.num_class)  # num_class
    os.makedirs(logpath, exist_ok=True)
    logpath = os.path.join(logpath, f"{flag}_eval")
    os.makedirs(logpath, exist_ok=True)
    logpath = os.path.join(logpath, "no_cross" if args.no_cross_road else "with_cross")
    os.makedirs(logpath, exist_ok=True)
    print("log path:", logpath)

    for idx, sample in enumerate(tqdm(valloader)):

        if args.com:
            (
                padded_voxel_points_list,
                padded_voxel_points_teacher_list,
                label_one_hot_list,
                trans_matrices,
                target_agent,
                num_sensor,
            ) = list(zip(*sample))
        else:
            (
                padded_voxel_points_list,
                padded_voxel_points_teacher_list,
                label_one_hot_list,
            ) = list(zip(*sample))

        if flag == "upperbound":
            padded_voxel_points = torch.cat(tuple(padded_voxel_points_teacher_list), 0)
        else:
            padded_voxel_points = torch.cat(tuple(padded_voxel_points_list), 0)

        label_one_hot = torch.cat(tuple(label_one_hot_list), 0)
        # print('voxel', padded_voxel_points.size())  # batch*agent seq h w z
        # print('label', label_one_hot.size())

        data = {}
        data["bev_seq"] = padded_voxel_points.to(device).float()
        data["labels"] = label_one_hot.to(device)
        if args.com:
            trans_matrices = torch.stack(trans_matrices, 1)

            # add pose noise
            if pose_noise > 0:
                apply_pose_noise(pose_noise, trans_matrices)

            target_agent = torch.stack(target_agent, 1)
            num_sensor = torch.stack(num_sensor, 1)

            if args.no_cross_road:
                num_sensor -= 1

            data["trans_matrices"] = trans_matrices.to(device)
            data["target_agent"] = target_agent.to(device)
            data["num_sensor"] = num_sensor.to(device)

        pred, labels = segmodule.step(data, num_agent, batch_size, loss=False)

        # If has RSU, do not count RSU's output into evaluation
        if not args.no_cross_road:
            pred = pred[1:, :, :, :]
            labels = labels[1:, :, :]

        labels = labels.detach().cpu().numpy().astype(np.int32)

        # late fusion
        if args.apply_late_fusion:
            pred = torch.flip(pred, (2,))
            size = (1, *pred[0].shape)

            for ii in range(num_sensor[0, 0]):
                for jj in range(num_sensor[0, 0]):
                    if ii == jj:
                        continue

                    nb_agent = torch.unsqueeze(pred[jj], 0)
                    tfm_ji = trans_matrices[0, jj, ii]
                    M = (
                        torch.hstack((tfm_ji[:2, :2], -tfm_ji[:2, 3:4]))
                        .float()
                        .unsqueeze(0)
                    )  # [1,2,3]

                    mask = torch.tensor(
                        [[[1, 1, 4 / 128], [1, 1, 4 / 128]]], device=M.device
                    )

                    M *= mask
                    grid = F.affine_grid(M, size=torch.Size(size)).to(device)
                    warp_feat = F.grid_sample(nb_agent, grid).squeeze()
                    pred[ii] += warp_feat

            pred = torch.flip(pred, (2,))
        # ============

        pred = torch.argmax(F.softmax(pred, dim=1), dim=1)
        compute_iou(pred, labels)

        if args.vis and idx % 50 == 0:  # render segmatic map
            plt.clf()
            pred_map = np.zeros((256, 256, 3))
            gt_map = np.zeros((256, 256, 3))

            for k, v in config.class_to_rgb.items():
                pred_map[np.where(pred.cpu().numpy()[0] == k)] = v
                gt_map[np.where(label_one_hot.numpy()[0] == k)] = v

            plt.imsave(
                f"{logpath}/{idx}_voxel_points.png",
                np.asarray(
                    np.max(padded_voxel_points.cpu().numpy()[0], axis=2), dtype=np.uint8
                ),
            )
            cv2.imwrite(f"{logpath}/{idx}_pred.png", pred_map[:, :, ::-1])
            cv2.imwrite(f"{logpath}/{idx}_gt.png", gt_map[:, :, ::-1])

    print("iou:", compute_iou.get_ious())
    print("miou:", compute_iou.get_miou(ignore=0))
    log_file = open(f"{logpath}/log.txt", "w")
    log_file.write(f"iou: {compute_iou.get_ious()}\n")
    log_file.write(f"miou: {compute_iou.get_miou(ignore=0)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data",
        default="./dataset/train",
        type=str,
        help="The path to the preprocessed sparse BEV training data",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="The path to the saved model that is loaded to resume training",
    )
    parser.add_argument("--model_only", action="store_true", help="only load model")
    parser.add_argument("--batch", default=1, type=int, help="Batch size")
    parser.add_argument("--nepoch", default=10, type=int, help="Number of epochs")
    parser.add_argument("--nworker", default=2, type=int, help="Number of workers")
    parser.add_argument("--lr", default=0.001, type=float, help="Initial learning rate")
    parser.add_argument("--com", default="", type=str, help="Whether to communicate")
    parser.add_argument("--inference", default="activated")
    parser.add_argument("--warp_flag", action="store_true")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--logpath", default="", help="The path to the output log file")
    parser.add_argument(
        "--no_cross_road", action="store_true", help="Do not load data of cross roads"
    )
    parser.add_argument(
        "--num_agent", default=6, type=int, help="The total number of agents"
    )
    parser.add_argument(
        "--pose_noise",
        default=0,
        type=float,
        help="draw noise from normal distribution with given mean (in meters), apply to transformation matrix.",
    )
    parser.add_argument(
        "--apply_late_fusion",
        default=0,
        type=int,
        help="1: apply late fusion. 0: no late fusion",
    )
    parser.add_argument(
        "--compress_level",
        default=0,
        type=int,
        help="Compress the communication layer channels by 2**x times in encoder",
    )
    parser.add_argument(
        "--only_v2i",
        default=0,
        type=int,
        help="1: only v2i, 0: v2v and v2i",
    )

    parser.add_argument("--bound", default="lowerbound", type=str)
    torch.multiprocessing.set_sharing_strategy("file_system")

    args = parser.parse_args()
    print(args)
    config = Config("train")
    main(config, args)
