# Copyright (c) 2020 Mitsubishi Electric Research Laboratories (MERL). All rights reserved. The software, documentation and/or data in this file is provided on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications. MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for any particular purpose. In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages. As more fully described in the license agreement that was required in order to download this software, documentation and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, research and non-commercial purposes.
import argparse
from genericpath import exists

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from coperception.datasets import V2XSimSeg
from coperception.configs import Config
from coperception.utils.SegModule import *
from coperception.utils.loss import *
from coperception.models.seg import *
from coperception.utils.AverageMeter import AverageMeter
from coperception.utils.data_util import apply_pose_noise
import glob
import os


def main(config, args):
    config.nepoch = args.nepoch
    num_epochs = args.nepoch
    need_log = args.log
    batch_size = args.batch_size
    num_workers = args.nworker
    compress_level = args.compress_level
    start_epoch = 1
    pose_noise = args.pose_noise
    only_v2i = args.only_v2i
    kd_flag = args.kd_flag

    # Specify gpu device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = torch.cuda.device_count()
    print("device number", device_num)

    if args.com == "upperbound":
        flag = "upperbound"
    elif args.com == "when2com" and args.warp_flag:
        flag = "when2com_warp"
    elif args.com in [
        "lowerbound",
        "v2v",
        "disco",
        "sum",
        "mean",
        "max",
        "cat",
        "agent",
        "when2com",
    ]:
        flag = args.com
    else:
        raise ValueError(f"com: {args.com} is not supported")

    config.flag = flag

    num_agent = args.num_agent
    # agent0 is the RSU
    agent_idx_range = range(num_agent) if args.rsu else range(1, num_agent)
    trainset = V2XSimSeg(
        dataset_roots=[args.data + "/agent%d" % i for i in agent_idx_range],
        config=config,
        split="train",
        com=args.com,
        bound="upperbound" if args.com == "upperbound" else "lowerbound",
        kd_flag=kd_flag,
        rsu=args.rsu,
    )
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    print("Training dataset size:", len(trainset))

    logger_root = args.logpath if args.logpath != "" else "logs"
    model_save_path = os.path.join(logger_root, flag)

    rsu_path = "with_rsu" if args.rsu else "no_rsu"
    model_save_path = os.path.join(model_save_path, rsu_path)
    os.makedirs(model_save_path, exist_ok=True)

    # build model
    if not args.rsu:
        num_agent -= 1

    if flag == "when2com" or flag == "when2com_warp":
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
    elif args.com == "agent":
        model = AgentWiseWeightedFusion(
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
    elif args.com == "disco":
        model = DiscoNet(
            config.in_channels,
            config.num_class,
            num_agent=num_agent,
            kd_flag=True,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif args.com == "lowerbound" or args.com == "upperbound":
        model = UNet(
            config.in_channels,
            config.num_class,
            num_agent=num_agent,
            compress_level=compress_level,
        )
    # model = nn.DataParallel(model)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    config.com = '' if (args.com == 'upperbound' or args.com == 'lowerbound') else args.com

    if kd_flag:
        teacher = UNet(
            config.in_channels, config.num_class, num_agent=num_agent, kd_flag=True
        )
        teacher = teacher.to(device)
        seg_module = SegModule(model, teacher, config, optimizer, kd_flag)
        checkpoint_teacher = torch.load(args.resume_teacher)
        start_epoch_teacher = checkpoint_teacher["epoch"]
        seg_module.teacher.load_state_dict(checkpoint_teacher["model_state_dict"])
        print(
            "Load teacher model from {}, at epoch {}".format(
                args.resume_teacher, start_epoch_teacher
            )
        )
        seg_module.teacher.eval()
    else:
        seg_module = SegModule(model, None, config, optimizer, kd_flag)

    if args.resume is None and (
        args.auto_resume_path == ""
        or "epoch_1.pth"
        not in os.listdir(os.path.join(args.auto_resume_path, f"{flag}/{rsu_path}"))
    ):
        log_file_name = os.path.join(model_save_path, "log.txt")
        saver = open(log_file_name, "w")
        saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
        saver.flush()

        # Logging the details for this experiment
        saver.write("command line: {}\n".format(" ".join(sys.argv[0:])))
        saver.write(args.__repr__() + "\n\n")
        saver.flush()

    else:
        if args.auto_resume_path != "":
            model_save_path = os.path.join(
                args.auto_resume_path, f"{flag}/{rsu_path}"
            )
        else:
            model_save_path = args.resume[: args.resume.rfind("/")]

        print(f"model save path: {model_save_path}")

        log_file_name = os.path.join(model_save_path, "log.txt")
        if os.path.exists(log_file_name):
            saver = open(log_file_name, "a")
        else:
            os.makedirs(model_save_path)
            saver = open(log_file_name, "w")

        saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
        saver.flush()

        # Logging the details for this experiment
        saver.write("command line: {}\n".format(" ".join(sys.argv[1:])))
        saver.write(args.__repr__() + "\n\n")
        saver.flush()

        if args.auto_resume_path != "":
            list_of_files = glob.glob(f"{model_save_path}/*.pth")
            latest_pth = max(list_of_files, key=os.path.getctime)
            checkpoint = torch.load(latest_pth)
        else:
            checkpoint = torch.load(args.resume)

        seg_module.model.load_state_dict(checkpoint["model_state_dict"])
        seg_module.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        seg_module.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

        print(
            "Load model from {}, at epoch {}".format(
                args.resume if args.resume is not None else args.auto_resume_path,
                start_epoch - 1,
            )
        )

    for epoch in range(start_epoch, num_epochs + 1):
        lr = seg_module.optimizer.param_groups[0]["lr"]
        print("Epoch {}, learning rate {}".format(epoch, lr))

        if need_log:
            saver.write("epoch: {}, lr: {}\t".format(epoch, lr))
            saver.flush()

        running_loss_disp = AverageMeter(
            "Total loss", ":.6f"
        )  # for motion prediction error
        seg_module.model.train()

        t = time.time()
        for idx, sample in enumerate(tqdm(trainloader)):
            if flag != 'lowerbound' and flag != 'upperbound':
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
                padded_voxel_points = torch.cat(
                    tuple(padded_voxel_points_teacher_list), 0
                )
            else:
                padded_voxel_points = torch.cat(tuple(padded_voxel_points_list), 0)

            label_one_hot = torch.cat(tuple(label_one_hot_list), 0)
            # print('voxel', padded_voxel_points.size())  # batch*agent seq h w z
            # print('label', label_one_hot.size())

            data = {}
            data["bev_seq"] = padded_voxel_points.to(device).float()
            data["labels"] = label_one_hot.to(device)
            if flag != 'lowerbound' and flag != 'upperbound':
                trans_matrices = torch.stack(trans_matrices, 1)
                # add pose noise
                if pose_noise > 0:
                    apply_pose_noise(pose_noise, trans_matrices)
                target_agent = torch.stack(target_agent, 1)
                num_sensor = torch.stack(num_sensor, 1)
                data["trans_matrices"] = trans_matrices.to(device)
                data["target_agent"] = target_agent

                if not args.rsu:
                    num_sensor -= 1

                data["num_sensor"] = num_sensor

            if kd_flag:
                padded_voxel_points_teacher = torch.cat(
                    tuple(padded_voxel_points_teacher_list), 0
                )
                data["bev_seq_teacher"] = padded_voxel_points_teacher.to(device)
                data["kd_weight"] = args.kd_weight

            pred, loss = seg_module.step(data, num_agent, batch_size)

            running_loss_disp.update(loss)
        print("\nEpoch {}".format(epoch))
        print("Running total loss: {}".format(running_loss_disp.avg))
        seg_module.scheduler.step()
        print("{}\t Takes {} s\n".format(running_loss_disp, str(time.time() - t)))

        if need_log:
            saver.write("{}\n".format(running_loss_disp))
            saver.flush()

        # save model
        if need_log:
            save_dict = {
                "epoch": epoch,
                "model_state_dict": seg_module.model.state_dict(),
                "optimizer_state_dict": seg_module.optimizer.state_dict(),
                "scheduler_state_dict": seg_module.scheduler.state_dict(),
                "loss": running_loss_disp.avg,
            }
            print(model_save_path)
            torch.save(
                save_dict, os.path.join(model_save_path, "epoch_" + str(epoch) + ".pth")
            )
    if need_log:
        saver.close()


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
        default=None,
        type=str,
        help="The path to the saved model that is loaded to resume training",
    )
    parser.add_argument("--model_only", action="store_true", help="only load model")
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size")
    parser.add_argument("--warp_flag", default=0, type=int, help="Whether to use pose info for When2com")
    parser.add_argument(
        "--kd_flag",
        default=0,
        type=int,
        help="Whether to enable distillation (only DiscNet is 1 )",
    )
    parser.add_argument(
        "--augmentation", default=False, help="Whether to use data augmentation"
    )
    parser.add_argument("--nepoch", default=100, type=int, help="Number of epochs")
    parser.add_argument("--nworker", default=2, type=int, help="Number of workers")
    parser.add_argument("--lr", default=0.001, type=float, help="Initial learning rate")
    parser.add_argument("--log", action="store_true", help="Whether to log")
    parser.add_argument("--logpath", default="", help="The path to the output log file")
    parser.add_argument("--com", default="", type=str, help="lowerbound/upperbound/disco/when2com/v2v/sum/mean/max/cat/agent")
    parser.add_argument("--rsu", default=0, type=int, help="0: no RSU, 1: RSU")
    parser.add_argument(
        "--num_agent", default=6, type=int, help="The total number of agents"
    )
    parser.add_argument(
        "--resume_teacher",
        default="",
        type=str,
        help="The path to the saved teacher model that is loaded to resume training",
    )
    parser.add_argument("--kd_weight", default=100, type=int, help="KD loss weight")
    parser.add_argument(
        "--auto_resume_path",
        default="",
        type=str,
        help="The path to automatically reload the latest pth",
    )
    parser.add_argument(
        "--compress_level",
        default=0,
        type=int,
        help="Compress the communication layer channels by 2**x times in encoder",
    )
    parser.add_argument(
        "--pose_noise",
        default=0,
        type=float,
        help="draw noise from normal distribution with given mean (in meters), apply to transformation matrix.",
    )
    parser.add_argument(
        "--only_v2i",
        default=0,
        type=int,
        help="1: only v2i, 0: v2v and v2i",
    )
    torch.multiprocessing.set_sharing_strategy("file_system")

    args = parser.parse_args()
    print(args)
    config = Config("train")
    main(config, args)
