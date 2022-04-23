import torch.nn.functional as F
import torch.nn as nn
import torch
from coperception.utils.detection_util import *


class SegModule(object):
    def __init__(self, model, teacher, config, optimizer, kd_flag):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.nepoch
        )
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150, 200], gamma=0.5)
        self.criterion = nn.CrossEntropyLoss()
        self.teacher = teacher
        if kd_flag:
            for k, v in self.teacher.named_parameters():
                v.requires_grad = False  # fix parameters

        self.kd_flag = kd_flag

        self.com = config.com

    def resume(self, path):
        def map_func(storage, location):
            return storage.cuda()

        if os.path.isfile(path):
            if rank == 0:
                print("=> loading checkpoint '{}'".format(path))

            checkpoint = torch.load(path, map_location=map_func)
            self.model.load_state_dict(checkpoint["state_dict"], strict=False)

            ckpt_keys = set(checkpoint["state_dict"].keys())
            own_keys = set(model.state_dict().keys())
            missing_keys = own_keys - ckpt_keys
            for k in missing_keys:
                print("caution: missing keys from checkpoint {}: {}".format(path, k))
        else:
            print("=> no checkpoint found at '{}'".format(path))

    def step(self, data, num_agent, batch_size, loss=True):
        bev = data["bev_seq"]
        labels = data["labels"]
        self.optimizer.zero_grad()
        bev = bev.permute(0, 3, 1, 2).contiguous()

        if not self.com:
            filtered_bev = []
            filtered_label = []
            for i in range(bev.size(0)):
                if torch.sum(bev[i]) > 1e-4:
                    filtered_bev.append(bev[i])
                    filtered_label.append(labels[i])
            bev = torch.stack(filtered_bev, 0)
            labels = torch.stack(filtered_label, 0)

        if self.kd_flag:
            data["bev_seq_teacher"] = (
                data["bev_seq_teacher"].permute(0, 3, 1, 2).contiguous()
            )

        if self.com:
            if self.kd_flag:
                pred, x9, x8, x7, x6, x5, fused_layer = self.model(
                    bev, data["trans_matrices"], data["num_sensor"]
                )
            elif self.config.flag.startswith("when2com") or self.config.flag.startswith(
                "who2com"
            ):
                if self.config.split == "train":
                    pred = self.model(
                        bev, data["trans_matrices"], data["num_sensor"], training=True
                    )
                else:
                    pred = self.model(
                        bev,
                        data["trans_matrices"],
                        data["num_sensor"],
                        inference=self.config.inference,
                        training=False,
                    )
            else:
                pred = self.model(bev, data["trans_matrices"], data["num_sensor"])
        else:
            pred = self.model(bev)

        if self.com:
            filtered_pred = []
            filtered_label = []
            for i in range(bev.size(0)):
                if torch.sum(bev[i]) > 1e-4:
                    filtered_pred.append(pred[i])
                    filtered_label.append(labels[i])
            pred = torch.stack(filtered_pred, 0)
            labels = torch.stack(filtered_label, 0)
        if not loss:
            return pred, labels

        kd_loss = (
            self.get_kd_loss(batch_size, data, fused_layer, num_agent, x5, x6, x7)
            if self.kd_flag
            else 0
        )
        loss = self.criterion(pred, labels.long()) + kd_loss

        if isinstance(self.criterion, nn.DataParallel):
            loss = loss.mean()

        loss_data = loss.data.item()
        if np.isnan(loss_data):
            raise ValueError("loss is nan while training")

        loss.backward()
        self.optimizer.step()

        return pred, loss_data

    def get_kd_loss(self, batch_size, data, fused_layer, num_agent, x5, x6, x7):
        if not self.kd_flag:
            return 0

        bev_seq_teacher = data["bev_seq_teacher"].type(torch.cuda.FloatTensor)
        kd_weight = data["kd_weight"]
        (
            logit_teacher,
            x9_teacher,
            x8_teacher,
            x7_teacher,
            x6_teacher,
            x5_teacher,
            x4_teacher,
        ) = self.teacher(bev_seq_teacher)
        kl_loss_mean = nn.KLDivLoss(size_average=True, reduce=True)

        target_x5 = x5_teacher.permute(0, 2, 3, 1).reshape(
            num_agent * batch_size * 16 * 16, -1
        )
        student_x5 = x5.permute(0, 2, 3, 1).reshape(
            num_agent * batch_size * 16 * 16, -1
        )
        kd_loss_x5 = kl_loss_mean(
            F.log_softmax(student_x5, dim=1), F.softmax(target_x5, dim=1)
        )

        target_x6 = x6_teacher.permute(0, 2, 3, 1).reshape(
            num_agent * batch_size * 32 * 32, -1
        )
        student_x6 = x6.permute(0, 2, 3, 1).reshape(
            num_agent * batch_size * 32 * 32, -1
        )
        kd_loss_x6 = kl_loss_mean(
            F.log_softmax(student_x6, dim=1), F.softmax(target_x6, dim=1)
        )

        target_x7 = x7_teacher.permute(0, 2, 3, 1).reshape(
            num_agent * batch_size * 64 * 64, -1
        )
        student_x7 = x7.permute(0, 2, 3, 1).reshape(
            num_agent * batch_size * 64 * 64, -1
        )
        kd_loss_x7 = kl_loss_mean(
            F.log_softmax(student_x7, dim=1), F.softmax(target_x7, dim=1)
        )

        target_x4 = x4_teacher.permute(0, 2, 3, 1).reshape(
            num_agent * batch_size * 32 * 32, -1
        )
        student_x4 = fused_layer.permute(0, 2, 3, 1).reshape(
            num_agent * batch_size * 32 * 32, -1
        )
        kd_loss_fused_layer = kl_loss_mean(
            F.log_softmax(student_x4, dim=1), F.softmax(target_x4, dim=1)
        )

        return kd_weight * (kd_loss_x5 + kd_loss_x6 + kd_loss_x7 + kd_loss_fused_layer)
