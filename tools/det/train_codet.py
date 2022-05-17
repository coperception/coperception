import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from coperception.datasets import V2XSimDet
from coperception.configs import Config, ConfigGlobal
from coperception.utils.CoDetModule import *
from coperception.utils.loss import *
from coperception.models.det import *
from coperception.utils import AverageMeter

import glob
import os


def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path


def main(args):
    config = Config("train", binary=True, only_det=True)
    config_global = ConfigGlobal("train", binary=True, only_det=True)

    num_epochs = args.nepoch
    need_log = args.log
    num_workers = args.nworker
    start_epoch = 1
    batch_size = args.batch
    compress_level = args.compress_level

    # Specify gpu device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = torch.cuda.device_count()
    print("device number", device_num)

    if args.bound == "upperbound":
        flag = "upperbound"
    elif args.bound == "lowerbound":
        if args.com == "when2com" and args.warp_flag:
            flag = "when2com_warp"
        elif args.com in {
            "v2v",
            "disco",
            "sum",
            "mean",
            "max",
            "cat",
            "agent",
            "when2com",
        }:
            flag = args.com
        else:
            flag = "lowerbound"
    else:
        raise ValueError("not implement")

    config.flag = flag

    num_agent = args.num_agent
    # agent0 is the cross road
    agent_idx_range = range(1, num_agent) if args.no_cross_road else range(num_agent)
    training_dataset = V2XSimDet(
        dataset_roots=[f"{args.data}/agent{i}" for i in agent_idx_range],
        config=config,
        config_global=config_global,
        split="train",
        bound=args.bound,
        kd_flag=args.kd_flag,
        no_cross_road=args.no_cross_road,
    )
    training_data_loader = DataLoader(
        training_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers
    )
    print("Training dataset size:", len(training_dataset))

    logger_root = args.logpath if args.logpath != "" else "logs"

    if args.no_cross_road:
        num_agent -= 1

    if args.com == "":
        model = FaFNet(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
        )
    elif args.com == "when2com":
        model = When2com(
            config,
            layer=args.layer,
            warp_flag=args.warp_flag,
            num_agent=num_agent,
            compress_level=compress_level,
        )
    elif args.com == "v2v":
        model = V2VNet(
            config,
            gnn_iter_times=args.gnn_iter_times,
            layer=args.layer,
            layer_channel=256,
            num_agent=num_agent,
            compress_level=compress_level,
        )
    elif args.com == "disco":
        model = DiscoNet(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
        )
    elif args.com == "sum":
        model = SumFusion(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
        )
    elif args.com == "mean":
        model = MeanFusion(
            config, layer=args.layer, kd_flag=args.kd_flag, num_agent=num_agent, compress_level=compress_level
        )
    elif args.com == "max":
        model = MaxFusion(
            config, layer=args.layer, kd_flag=args.kd_flag, num_agent=num_agent, compress_level=compress_level
        )
    elif args.com == "cat":
        model = CatFusion(
            config, layer=args.layer, kd_flag=args.kd_flag, num_agent=num_agent, compress_level=compress_level
        )
    elif args.com == "agent":
        model = AgentWiseWeightedFusion(
            config, layer=args.layer, kd_flag=args.kd_flag, num_agent=num_agent, compress_level=compress_level
        )
    else:
        raise NotImplementedError("Invalid argument com:" + args.com)

    model = nn.DataParallel(model)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = {
        "cls": SoftmaxFocalClassificationLoss(),
        "loc": WeightedSmoothL1LocalizationLoss(),
    }

    if args.kd_flag == 1:
        teacher = TeacherNet(config)
        teacher = nn.DataParallel(teacher)
        teacher = teacher.to(device)
        faf_module = FaFModule(
            model, teacher, config, optimizer, criterion, args.kd_flag
        )
        checkpoint_teacher = torch.load(args.resume_teacher)
        start_epoch_teacher = checkpoint_teacher["epoch"]
        faf_module.teacher.load_state_dict(checkpoint_teacher["model_state_dict"])
        print(
            "Load teacher model from {}, at epoch {}".format(
                args.resume_teacher, start_epoch_teacher
            )
        )
        faf_module.teacher.eval()
    else:
        faf_module = FaFModule(model, model, config, optimizer, criterion, args.kd_flag)

    cross_path = "no_cross" if args.no_cross_road else "with_cross"
    model_save_path = check_folder(logger_root)
    model_save_path = check_folder(os.path.join(model_save_path, flag))

    if args.no_cross_road:
        model_save_path = check_folder(os.path.join(model_save_path, "no_cross"))
    else:
        model_save_path = check_folder(os.path.join(model_save_path, "with_cross"))

    if args.resume == "" and (
        args.auto_resume_path == ""
        or "epoch_1.pth"
        not in os.listdir(os.path.join(args.auto_resume_path, f"{flag}/{cross_path}"))
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
                args.auto_resume_path, f"{flag}/{cross_path}"
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

        start_epoch = checkpoint["epoch"] + 1
        faf_module.model.load_state_dict(checkpoint["model_state_dict"])
        faf_module.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        faf_module.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print("Load model from {}, at epoch {}".format(args.resume, start_epoch - 1))

    for epoch in range(start_epoch, num_epochs + 1):
        lr = faf_module.optimizer.param_groups[0]["lr"]
        print("Epoch {}, learning rate {}".format(epoch, lr))

        if need_log:
            saver.write("epoch: {}, lr: {}\t".format(epoch, lr))
            saver.flush()

        running_loss_disp = AverageMeter("Total loss", ":.6f")
        running_loss_class = AverageMeter(
            "classification Loss", ":.6f"
        )  # for cell classification error
        running_loss_loc = AverageMeter(
            "Localization Loss", ":.6f"
        )  # for state estimation error

        faf_module.model.train()

        t = tqdm(training_data_loader)
        for sample in t:
            (
                padded_voxel_point_list,
                padded_voxel_points_teacher_list,
                label_one_hot_list,
                reg_target_list,
                reg_loss_mask_list,
                anchors_map_list,
                vis_maps_list,
                target_agent_id_list,
                num_agent_list,
                trans_matrices_list,
            ) = zip(*sample)

            trans_matrices = torch.stack(tuple(trans_matrices_list), 1)
            target_agent_id = torch.stack(tuple(target_agent_id_list), 1)
            num_all_agents = torch.stack(tuple(num_agent_list), 1)

            if args.no_cross_road:
                num_all_agents -= 1

            if flag == "upperbound":
                padded_voxel_point = torch.cat(
                    tuple(padded_voxel_points_teacher_list), 0
                )
            else:
                padded_voxel_point = torch.cat(tuple(padded_voxel_point_list), 0)

            label_one_hot = torch.cat(tuple(label_one_hot_list), 0)
            reg_target = torch.cat(tuple(reg_target_list), 0)
            reg_loss_mask = torch.cat(tuple(reg_loss_mask_list), 0)
            anchors_map = torch.cat(tuple(anchors_map_list), 0)
            vis_maps = torch.cat(tuple(vis_maps_list), 0)

            data = {
                "bev_seq": padded_voxel_point.to(device),
                "labels": label_one_hot.to(device),
                "reg_targets": reg_target.to(device),
                "anchors": anchors_map.to(device),
                "reg_loss_mask": reg_loss_mask.to(device).type(dtype=torch.bool),
                "vis_maps": vis_maps.to(device),
                "target_agent_ids": target_agent_id.to(device),
                "num_agent": num_all_agents.to(device),
                "trans_matrices": trans_matrices,
            }

            if args.kd_flag == 1:
                padded_voxel_points_teacher = torch.cat(
                    tuple(padded_voxel_points_teacher_list), 0
                )
                data["bev_seq_teacher"] = padded_voxel_points_teacher.to(device)
                data["kd_weight"] = args.kd_weight

            loss, cls_loss, loc_loss = faf_module.step(
                data, batch_size, num_agent=num_agent
            )
            running_loss_disp.update(loss)
            running_loss_class.update(cls_loss)
            running_loss_loc.update(loc_loss)

            if np.isnan(loss) or np.isnan(cls_loss) or np.isnan(loc_loss):
                print(f"Epoch {epoch}, loss is nan: {loss}, {cls_loss} {loc_loss}")
                sys.exit()

            t.set_description("Epoch {},     lr {}".format(epoch, lr))
            t.set_postfix(
                cls_loss=running_loss_class.avg, loc_loss=running_loss_loc.avg
            )

        faf_module.scheduler.step()

        # save model
        if need_log:
            saver.write(
                "{}\t{}\t{}\n".format(
                    running_loss_disp, running_loss_class, running_loss_loc
                )
            )
            saver.flush()
            if config.MGDA:
                save_dict = {
                    "epoch": epoch,
                    "encoder_state_dict": faf_module.encoder.state_dict(),
                    "optimizer_encoder_state_dict": faf_module.optimizer_encoder.state_dict(),
                    "scheduler_encoder_state_dict": faf_module.scheduler_encoder.state_dict(),
                    "head_state_dict": faf_module.head.state_dict(),
                    "optimizer_head_state_dict": faf_module.optimizer_head.state_dict(),
                    "scheduler_head_state_dict": faf_module.scheduler_head.state_dict(),
                    "loss": running_loss_disp.avg,
                }
            else:
                save_dict = {
                    "epoch": epoch,
                    "model_state_dict": faf_module.model.state_dict(),
                    "optimizer_state_dict": faf_module.optimizer.state_dict(),
                    "scheduler_state_dict": faf_module.scheduler.state_dict(),
                    "loss": running_loss_disp.avg,
                }
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
        default=None,
        type=str,
        help="The path to the preprocessed sparse BEV training data",
    )
    parser.add_argument("--batch", default=4, type=int, help="Batch size")
    parser.add_argument("--nepoch", default=100, type=int, help="Number of epochs")
    parser.add_argument("--nworker", default=2, type=int, help="Number of workers")
    parser.add_argument("--lr", default=0.001, type=float, help="Initial learning rate")
    parser.add_argument("--log", action="store_true", help="Whether to log")
    parser.add_argument("--logpath", default="", help="The path to the output log file")
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        help="The path to the saved model that is loaded to resume training",
    )
    parser.add_argument(
        "--resume_teacher",
        default="",
        type=str,
        help="The path to the saved teacher model that is loaded to resume training",
    )
    parser.add_argument(
        "--layer",
        default=3,
        type=int,
        help="Communicate which layer in the single layer com mode",
    )
    parser.add_argument(
        "--warp_flag", action="store_true", help="Whether to use pose info for Ｗhen2com"
    )
    parser.add_argument(
        "--kd_flag",
        default=0,
        type=int,
        help="Whether to enable distillation (only DiscNet is 1 )",
    )
    parser.add_argument("--kd_weight", default=100000, type=int, help="KD loss weight")
    parser.add_argument(
        "--gnn_iter_times",
        default=3,
        type=int,
        help="Number of message passing for V2VNet",
    )
    parser.add_argument(
        "--visualization", default=True, help="Visualize validation result"
    )
    parser.add_argument(
        "--com", default="", type=str, help="disco/when2com/v2v/sum/mean/max/cat/agent"
    )
    parser.add_argument(
        "--bound",
        type=str,
        help="The input setting: lowerbound -> single-view or upperbound -> multi-view",
    )
    parser.add_argument(
        "--no_cross_road", action="store_true", help="Do not load data of cross roads"
    )
    parser.add_argument(
        "--num_agent", default=6, type=int, help="The total number of agents"
    )
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

    torch.multiprocessing.set_sharing_strategy("file_system")
    args = parser.parse_args()
    print(args)
    main(args)
