import matplotlib

import torch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from coperception.utils.postprocess import *
import torch.nn.functional as F
from coperception.utils.obj_util import *
from torch import stack as tstack

matplotlib.use("Agg")


def apply_box_global_transform(
    trans_matrices_map,
    batch_box_preds,
    batch_cls_preds,
    anchors,
    code_type,
    config,
    batch_motion=None,
):
    """
    Transform the predicted boxes into the global scene for global mAP evaluation.
    2020.10.11 Yiming Li
    """

    batch_anchors = anchors.view(
        batch_box_preds.shape[0], -1, batch_box_preds.shape[-1]
    )
    assert (
        len(batch_box_preds.shape) == 6
    ), "bbox must have shape [N ,W , H , num_per_loc, T, box_code]"

    batch_id = 0
    for box_preds, cls_preds, anchors in zip(
        batch_box_preds, batch_cls_preds, batch_anchors
    ):
        # N  * (W X H) * T * decoded_loc_dim(6)
        global_cls_preds = cls_preds
        if config.motion_state:
            total_motion = F.softmax(batch_motion[batch_id], dim=-1)[..., 1:]
            total_motion = total_motion.cpu().detach().numpy()
            total_motion = np.argmax(total_motion, axis=-1)
            batch_id += 1

        boxes_for_nms = box_preds.view(
            -1, box_preds.shape[-2], box_preds.shape[-1]
        )  # [N,pred_len,code_size]
        box_corners = np.random.rand(
            boxes_for_nms.shape[0], boxes_for_nms.shape[1], 4, 2
        )  # [N, pred_len,4 ,2]

        if config.pred_type == "motion":
            cur_det = None
        for i in range(boxes_for_nms.shape[1]):
            if code_type[0] == "f":

                if config.pred_type == "motion":
                    # motion _a
                    """
                    if i==0:
                        decoded_boxes = bev_box_decode_torch(boxes_for_nms[:,i],anchors).cpu().detach().numpy()
                        cur_det = decoded_boxes
                    else:
                        decoded_boxes = cur_det
                        decoded_boxes[:,:2] += boxes_for_nms[:,i,:2].cpu().detach().numpy()
                    #print(decoded_boxes.shape)
                    """
                    # motion _b
                    if i == 0:
                        decoded_boxes = (
                            bev_box_decode_torch(boxes_for_nms[:, i], anchors)
                            .cpu()
                            .detach()
                            .numpy()
                        )
                        cur_det = decoded_boxes.copy()
                    else:
                        decoded_boxes = cur_det
                        if config.motion_state:
                            moving_idx = total_motion == 1
                            moving_box = (
                                boxes_for_nms.cpu().detach().numpy()[moving_idx, i]
                            )
                            decoded_boxes[moving_idx, :2] += moving_box[:, :2]
                        else:
                            decoded_boxes[:, :2] += (
                                boxes_for_nms[:, i, :2].cpu().detach().numpy()
                            )
                        cur_det = decoded_boxes.copy()
                else:
                    decoded_boxes = (
                        bev_box_decode_torch(boxes_for_nms[:, i], anchors)
                        .cpu()
                        .detach()
                        .numpy()
                    )
                # print(w_id,decoded_boxes[w_id])

                box_pred_corners = center_to_corner_box2d(
                    decoded_boxes[:, :2], decoded_boxes[:, 2:4], decoded_boxes[:, 4:]
                )  # corners: [N, 4, 2]
            elif code_type[0] == "c":
                box_pred_corners = (
                    bev_box_decode_corner(boxes_for_nms[:, i], anchors)
                    .cpu()
                    .detach()
                    .numpy()
                )
                box_pred_corners = box_pred_corners.reshape(
                    -1, 4, 2
                )  # corners: [N, 4, 2]

            # print(box_preds_corners[w_id])
            # exit()
            box_corners[:, i] = box_pred_corners

        temp = box_corners.reshape(-1, 2)
        local_points = temp.T
        local_points[0, :] = -local_points[0, :]
        trans_matrices_map = torch.squeeze(trans_matrices_map)
        trans_matrices_map = trans_matrices_map.cpu().detach().numpy()
        global_points = np.dot(
            trans_matrices_map,
            np.vstack(
                (
                    local_points,
                    np.zeros(local_points.shape[1]),
                    np.ones(local_points.shape[1]),
                )
            ),
        )[:2, :]
        # print(global_points.shape)
        global_points[0, :] = -global_points[0, :]
        global_points = global_points.T
        global_points = global_points.reshape(-1, 1, 4, 2)

        # print(batch_box_preds.shape, batch_cls_preds.shape, global_points.shape, global_cls_preds.shape)

    return global_points, global_cls_preds


def apply_box_global_transform_af_localnms(
    trans_matrices_map, class_selected, box_scores_pred_cls
):
    """
    Transform the predicted boxes into the global scene after local nms.
    2021.4.2 Yiming Li
    """

    box_corners = class_selected[0][0]["pred"]
    box_scores_af_localnms = box_scores_pred_cls

    temp = box_corners.reshape(-1, 2)
    local_points = temp.T
    local_points[0, :] = -local_points[0, :]
    trans_matrices_map = torch.squeeze(trans_matrices_map)
    trans_matrices_map = trans_matrices_map.cpu().detach().numpy()
    global_points = np.dot(
        trans_matrices_map,
        np.vstack(
            (
                local_points,
                np.zeros(local_points.shape[1]),
                np.ones(local_points.shape[1]),
            )
        ),
    )[:2, :]
    # print(global_points.shape)
    global_points[0, :] = -global_points[0, :]
    global_points = global_points.T
    global_boxes_af_localnms = global_points.reshape(-1, 1, 4, 2)

    return global_boxes_af_localnms, box_scores_af_localnms


def apply_nms_global_scene(all_points_scene, cls_preds_scene):
    # print(all_points_scene.shape, cls_preds_scene.shape)

    predictions_dicts = []
    # N  * (W X H) * T * decoded_loc_dim(6)
    total_scores = F.softmax(cls_preds_scene, dim=-1)[..., 1:]
    # print(cls_preds_scene.shape, F.softmax(cls_preds_scene, dim=-1).shape, total_scores.shape)
    total_scores = total_scores.cpu().detach().numpy()

    class_selected = []
    for i in range(total_scores.shape[1]):
        selected_idx = non_max_suppression(
            all_points_scene[:, 0], total_scores[:, i], threshold=0.01
        )
        class_selected.append(
            {
                "pred": all_points_scene[selected_idx],
                "score": total_scores[selected_idx, i],
                "selected_idx": selected_idx,
            }
        )
    predictions_dicts.append(class_selected)

    return predictions_dicts


def apply_box_local_transform(class_selected_global, trans_matrices_map):

    predictions_dicts = []
    class_selected = []

    global_corners_af_NMS = class_selected_global[0][0]["pred"]
    global_scores_af_NMS = class_selected_global[0][0]["score"]

    temp = global_corners_af_NMS.reshape(-1, 2)
    global_points = temp.T
    global_points[0, :] = -global_points[0, :]

    trans_matrices_g2l = torch.inverse(
        torch.squeeze(trans_matrices_map)
    )  # transformation from global2local
    trans_matrices_g2l = trans_matrices_g2l.cpu().detach().numpy()

    local_points = np.dot(
        trans_matrices_g2l,
        np.vstack(
            (
                global_points,
                np.zeros(global_points.shape[1]),
                np.ones(global_points.shape[1]),
            )
        ),
    )[:2, :]

    local_points[0, :] = -local_points[0, :]
    local_points = local_points.T

    local_points = local_points.reshape(-1, 1, 4, 2)

    local_index = []
    for i in range(local_points.shape[0]):
        x_c = np.mean(local_points[i, 0, :, 0])
        y_c = np.mean(local_points[i, 0, :, 1])
        if np.abs(x_c) <= 32 and np.abs(y_c) <= 32:
            local_index.append(i)

    local_boxes = local_points[local_index]
    local_scores = global_scores_af_NMS[local_index]

    class_selected.append(
        {"pred": local_boxes, "score": local_scores, "selected_idx": local_index}
    )
    predictions_dicts.append(class_selected)

    return predictions_dicts, len(local_index)


def apply_nms_det(
    batch_box_preds, batch_cls_preds, anchors, code_type, config, batch_motion=None
):

    predictions_dicts = []
    batch_anchors = anchors.view(
        batch_box_preds.shape[0], -1, batch_box_preds.shape[-1]
    )

    assert (
        len(batch_box_preds.shape) == 6
    ), "bbox must have shape [N ,W , H , num_per_loc, T, box_code]"

    batch_id = 0
    for box_preds, cls_preds, anchors in zip(
        batch_box_preds, batch_cls_preds, batch_anchors
    ):

        # N  * (W X H) * T * decoded_loc_dim(6)
        total_scores = F.softmax(cls_preds, dim=-1)[..., 1:]
        total_scores = total_scores.cpu().detach().numpy()

        if config.motion_state:
            total_motion = F.softmax(batch_motion[batch_id], dim=-1)[..., 1:]
            total_motion = total_motion.cpu().detach().numpy()
            total_motion = np.argmax(total_motion, axis=-1)
            batch_id += 1

        boxes_for_nms = box_preds.view(
            -1, box_preds.shape[-2], box_preds.shape[-1]
        )  # [N,pred_len,code_size]
        box_corners = np.random.rand(
            boxes_for_nms.shape[0], boxes_for_nms.shape[1], 4, 2
        )  # [N, pred_len,4 ,2]

        if config.pred_type == "motion":
            cur_det = None
        for i in range(boxes_for_nms.shape[1]):
            if code_type[0] == "f":

                if config.pred_type == "motion":
                    # motion _a
                    """
                    if i==0:
                        decoded_boxes = bev_box_decode_torch(boxes_for_nms[:,i],anchors).cpu().detach().numpy()
                        cur_det = decoded_boxes
                    else:
                        decoded_boxes = cur_det
                        decoded_boxes[:,:2] += boxes_for_nms[:,i,:2].cpu().detach().numpy()
                    #print(decoded_boxes.shape)
                    """
                    # motion _b
                    if i == 0:
                        decoded_boxes = (
                            bev_box_decode_torch(boxes_for_nms[:, i], anchors)
                            .cpu()
                            .detach()
                            .numpy()
                        )
                        cur_det = decoded_boxes.copy()
                    else:
                        decoded_boxes = cur_det
                        if config.motion_state:
                            moving_idx = total_motion == 1
                            moving_box = (
                                boxes_for_nms.cpu().detach().numpy()[moving_idx, i]
                            )
                            decoded_boxes[moving_idx, :2] += moving_box[:, :2]
                        else:
                            decoded_boxes[:, :2] += (
                                boxes_for_nms[:, i, :2].cpu().detach().numpy()
                            )
                        cur_det = decoded_boxes.copy()
                else:
                    decoded_boxes = (
                        bev_box_decode_torch(boxes_for_nms[:, i], anchors)
                        .cpu()
                        .detach()
                        .numpy()
                    )
                # print(w_id,decoded_boxes[w_id])

                box_pred_corners = center_to_corner_box2d(
                    decoded_boxes[:, :2], decoded_boxes[:, 2:4], decoded_boxes[:, 4:]
                )  # corners: [N, 4, 2]
            elif code_type[0] == "c":
                box_pred_corners = (
                    bev_box_decode_corner(boxes_for_nms[:, i], anchors)
                    .cpu()
                    .detach()
                    .numpy()
                )
                box_pred_corners = box_pred_corners.reshape(
                    -1, 4, 2
                )  # corners: [N, 4, 2]

            box_corners[:, i] = box_pred_corners

        class_selected = []
        # print(box_preds.shape, cls_preds.shape, total_scores.shape)
        for i in range(total_scores.shape[1]):
            selected_idx = non_max_suppression(
                box_corners[:, 0], total_scores[:, i], threshold=0.01
            )
            class_selected.append(
                {
                    "pred": box_corners[selected_idx],
                    "score": total_scores[selected_idx, i],
                    "selected_idx": selected_idx,
                }
            )

            cls_pred_first_nms = cls_preds[selected_idx, :]
            # break

        predictions_dicts.append(class_selected)

    return predictions_dicts, cls_pred_first_nms


def bev_box_decode_torch(
    box_encodings, anchors, encode_angle_to_vector=False, smooth_dim=False
):
    """box decode for VoxelNet in lidar
    Args:
        boxes ([N, 6] Tensor): normal boxes: x, y, w, l, sin, cos
        anchors ([N, 6] Tensor): anchors
    """

    xa, ya, wa, ha, sina, cosa = torch.split(anchors, 1, dim=-1)
    xp, yp, wp, hp, sinp, cosp = torch.split(box_encodings, 1, dim=-1)

    # xt, yt, zt, wt, lt, ht, rt = torch.split(box_encodings, 1, dim=-1)
    h = ha / torch.exp(hp)
    w = wa / torch.exp(wp)

    x = xa - w * xp
    y = ya - h * yp

    sin = sina * cosp + cosa * sinp
    cos = cosa * cosp - sina * sinp

    box_decoding = torch.cat([x, y, w, h, sin, cos], dim=-1)

    return box_decoding


def center_to_corner_box2d_torch(centers, dims, angles=None, origin=0.5):
    """convert kitti locations, dimensions and angles to corners

    Args:
        centers (float array, shape=[N, 2]): locations in kitti label file.
        dims (float array, shape=[N, 2]): dimensions in kitti label file.
        angles (float array, shape=[N,2]): rotation_y in kitti label file.

    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # xyz(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd_torch(dims, origin=origin)
    # corners: [N, 4, 2]
    if angles is not None:
        corners = rotation_2d_torch(corners, angles)
    corners += centers.view(-1, 1, 2)
    return corners


def corners_nd_torch(dims, origin=0.5):
    """generate relative box corners based on length per dim and
    origin point.

    Args:
        dims (float array, shape=[N, ndim]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.
        dtype (output dtype, optional): Defaults to np.float32

    Returns:
        float array, shape=[N, 2 ** ndim, ndim]: returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1
    """
    ndim = int(dims.shape[1])
    dtype = np.float32
    if isinstance(origin, float):
        origin = [origin] * ndim
    corners_norm = np.stack(
        np.unravel_index(np.arange(2**ndim), [2] * ndim), axis=1
    ).astype(dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start from minimum point
    # for 3d boxes, please draw them by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dtype)
    corners_norm = torch.from_numpy(corners_norm).type_as(dims)
    corners = dims.view(-1, 1, ndim) * corners_norm.view(1, 2**ndim, ndim)
    corners = torch.cat(
        (
            corners[:, [1], :],
            corners[:, [2], :],
            corners[:, [3], :],
            corners[:, [0], :],
        ),
        dim=1,
    )
    return corners


def rotation_2d_torch(points, angles):
    """rotation 2d points based on origin point clockwise when angle positive.

    Args:
        points (float array, shape=[N, point_size, 2]): points to be rotated.
        angles (float array, shape=[N]): rotation angle.

    Returns:
        float array: same shape as points
    """
    rot_sin = angles[:, 0]
    rot_cos = angles[:, 1]
    rot_mat_T = torch.stack([tstack([rot_cos, -rot_sin]), tstack([rot_sin, rot_cos])])
    return torch.einsum("aij,jka->aik", (points, rot_mat_T))


def bev_box_decode_corner(
    box_encodings, anchors, encode_angle_to_vector=False, smooth_dim=False
):
    """box decode for VoxelNet in lidar
    Args:
        boxes ([N, 8] Tensor): normal boxes: x, y, w, l, sin, cos
        anchors ([N, 8] Tensor): anchors
    """

    # box_encodings = box_decoding + anchors

    # return box_decoding

    raise NotImplementedError(
        "bev_box_decode_corner in detection_util is not implemented"
    )


def cal_local_mAP(config, data, det_results, annotations):
    # voxel_size = config.voxel_size
    # area_extents = config.area_extents
    # anchor_size = config.anchor_size
    pred_len = 1
    # box_code_size = 6  # (x,y,w,h,sin,cos)

    # voxel map visualization
    # voxel = data["bev_seq"]

    # maps = np.max(voxel, axis=-1)

    anchors_map = data["anchors_map"]
    reg_targets = data["reg_targets"]
    pred_selected = data["result"]
    gt_max_iou_idx = data["gt_max_iou"]

    # empty
    if len(gt_max_iou_idx) == 0:
        return det_results, annotations

    # if anchors_map.shape[2] < 7:#binary classification only has 4 anchors
    #    anchors_map = np.concatenate([anchors_map[:,:,:2],np.zeros_like(anchors_map[:,:,:3]),anchors_map[:,:,2:]],axis=2)
    #    reg_targets = np.concatenate([reg_targets[:,:,:2],np.zeros_like(reg_targets[:,:,:3]),reg_targets[:,:,2:]],axis=2)
    plt.clf()
    for p in range(pred_len):
        gt_corners = []
        pred_corners = []
        cls_score = []
        det_results_multiclass = []

        for k in range(len(gt_max_iou_idx)):

            anchor = anchors_map[tuple(gt_max_iou_idx[k][:-1])]

            encode_box = reg_targets[tuple(gt_max_iou_idx[k][:-1]) + (p,)]

            decode_box = bev_box_decode_np(encode_box, anchor)
            # print(decode_box)
            decode_corner = center_to_corner_box2d(
                np.asarray([decode_box[:2]]),
                np.asarray([decode_box[2:4]]),
                np.asarray([decode_box[4:]]),
            )[0]
            gt_corners.append(decode_corner)

        # print(pred_selected[0])
        for k in range(len(pred_selected)):
            cls_pred_corners = pred_selected[k]["pred"][:, p]
            cls_pred_scores = pred_selected[k]["score"]
            # cls_pred_idx = pred_selected[k]['selected_idx']
            pred_corners = cls_pred_corners
            cls_score = cls_pred_scores

        ## iou calculation
        gt_corners = np.asarray(gt_corners)
        #        if init_flag == False:
        #            ious_gt_corners = gt_corners
        #            ious_pred_corners = pred_corners
        #            init_flag = True
        #        else:
        #            ious_gt_corners = np.concatenate((ious_gt_corners,gt_corners),axis=0)
        #            ious_pred_corners = np.concatenate((ious_pred_corners,pred_corners),axis=0)

        for k in range(gt_corners.shape[0]):
            gt_box = np.array(
                [
                    gt_corners[k, 0, 0],
                    gt_corners[k, 0, 1],
                    gt_corners[k, 1, 0],
                    gt_corners[k, 1, 1],
                    gt_corners[k, 2, 0],
                    gt_corners[k, 2, 1],
                    gt_corners[k, 3, 0],
                    gt_corners[k, 3, 1],
                ]
            )
            if k == 0:
                gt_boxes_frame = np.array(
                    [
                        gt_corners[k, 0, 0],
                        gt_corners[k, 0, 1],
                        gt_corners[k, 1, 0],
                        gt_corners[k, 1, 1],
                        gt_corners[k, 2, 0],
                        gt_corners[k, 2, 1],
                        gt_corners[k, 3, 0],
                        gt_corners[k, 3, 1],
                    ]
                )
            else:
                gt_boxes_frame = np.vstack((gt_boxes_frame, gt_box))

        # make the dimension match
        if gt_corners.shape[0] == 1:
            gt_boxes_frame = np.array([gt_boxes_frame])

        annotation_frame = {
            "bboxes": gt_boxes_frame,
            "labels": np.zeros(gt_corners.shape[0]),
        }

        for k in range(pred_corners.shape[0]):

            detection_result = np.array(
                [
                    pred_corners[k, 0, 0],
                    pred_corners[k, 0, 1],
                    pred_corners[k, 1, 0],
                    pred_corners[k, 1, 1],
                    pred_corners[k, 2, 0],
                    pred_corners[k, 2, 1],
                    pred_corners[k, 3, 0],
                    pred_corners[k, 3, 1],
                    cls_score[k],
                ]
            )
            if k == 0:
                det_results_frame = np.array(
                    [
                        pred_corners[k, 0, 0],
                        pred_corners[k, 0, 1],
                        pred_corners[k, 1, 0],
                        pred_corners[k, 1, 1],
                        pred_corners[k, 2, 0],
                        pred_corners[k, 2, 1],
                        pred_corners[k, 3, 0],
                        pred_corners[k, 3, 1],
                        cls_score[k],
                    ]
                )
            else:
                det_results_frame = np.vstack((det_results_frame, detection_result))

        # make the dimension match
        if pred_corners.shape[0] == 1:
            det_results_frame = np.array([det_results_frame])

        if pred_corners.shape[0] > 0:
            det_results_multiclass.append(det_results_frame)
            det_results.append(det_results_multiclass)
            annotations.append(annotation_frame)

    return det_results, annotations


def cal_global_mAP(config, data, det_results, annotations):
    # voxel_size = config.voxel_size
    # area_extents = config.area_extents
    # anchor_size = config.anchor_size
    pred_len = 1
    # box_code_size = 6  # (x,y,w,h,sin,cos)
    anchors_map = data["anchors_map"]
    reg_targets = data["reg_targets"]
    pred_selected = data["result"]
    gt_max_iou_idx = data["gt_max_iou"]

    plt.clf()
    for p in range(pred_len):
        gt_corners = []
        pred_corners = []
        cls_score = []
        det_results_multiclass = []

        for k in range(len(gt_max_iou_idx)):

            anchor = anchors_map[tuple(gt_max_iou_idx[k][:-1])]

            encode_box = reg_targets[tuple(gt_max_iou_idx[k][:-1]) + (p,)]

            decode_box = bev_box_decode_np(encode_box, anchor)
            # print(decode_box)
            decode_corner = center_to_corner_box2d(
                np.asarray([decode_box[:2]]),
                np.asarray([decode_box[2:4]]),
                np.asarray([decode_box[4:]]),
            )[0]
            gt_corners.append(decode_corner)

        for k in range(len(pred_selected)):
            cls_pred_corners = pred_selected[k]["pred"][:, p]
            cls_pred_scores = pred_selected[k]["score"]
            # cls_pred_idx = pred_selected[k]['selected_idx']
            pred_corners = cls_pred_corners
            cls_score = cls_pred_scores

        ## iou calculation
        gt_corners = np.asarray(gt_corners)

        for k in range(gt_corners.shape[0]):
            gt_box = np.array(
                [
                    gt_corners[k, 0, 0],
                    gt_corners[k, 0, 1],
                    gt_corners[k, 1, 0],
                    gt_corners[k, 1, 1],
                    gt_corners[k, 2, 0],
                    gt_corners[k, 2, 1],
                    gt_corners[k, 3, 0],
                    gt_corners[k, 3, 1],
                ]
            )
            if k == 0:
                gt_boxes_frame = np.array(
                    [
                        gt_corners[k, 0, 0],
                        gt_corners[k, 0, 1],
                        gt_corners[k, 1, 0],
                        gt_corners[k, 1, 1],
                        gt_corners[k, 2, 0],
                        gt_corners[k, 2, 1],
                        gt_corners[k, 3, 0],
                        gt_corners[k, 3, 1],
                    ]
                )
            else:
                gt_boxes_frame = np.vstack((gt_boxes_frame, gt_box))

        # make the dimension match
        if gt_corners.shape[0] == 1:
            gt_boxes_frame = np.array([gt_boxes_frame])
        annotation_frame = {
            "bboxes": gt_boxes_frame,
            "labels": np.zeros(gt_corners.shape[0]),
        }

        for k in range(pred_corners.shape[0]):
            detection_result = np.array(
                [
                    pred_corners[k, 0, 0],
                    pred_corners[k, 0, 1],
                    pred_corners[k, 1, 0],
                    pred_corners[k, 1, 1],
                    pred_corners[k, 2, 0],
                    pred_corners[k, 2, 1],
                    pred_corners[k, 3, 0],
                    pred_corners[k, 3, 1],
                    cls_score[k],
                ]
            )
            if k == 0:
                det_results_frame = np.array(
                    [
                        pred_corners[k, 0, 0],
                        pred_corners[k, 0, 1],
                        pred_corners[k, 1, 0],
                        pred_corners[k, 1, 1],
                        pred_corners[k, 2, 0],
                        pred_corners[k, 2, 1],
                        pred_corners[k, 3, 0],
                        pred_corners[k, 3, 1],
                        cls_score[k],
                    ]
                )
            else:
                det_results_frame = np.vstack((det_results_frame, detection_result))

        # make the dimension match
        if pred_corners.shape[0] == 1:
            det_results_frame = np.array([det_results_frame])
        if pred_corners.shape[0] > 0:
            det_results_multiclass.append(det_results_frame)
            det_results.append(det_results_multiclass)
            annotations.append(annotation_frame)

    return det_results, annotations


def get_gt_corners(config, data):
    voxel_size = config.voxel_size
    area_extents = config.area_extents
    # anchor_size = config.anchor_size
    # map_dims = config.map_dims
    pred_len = 1
    # box_code_size = 6  # (x,y,w,h,sin,cos)

    # voxel map visualization
    # voxel = data["bev_seq"]
    # maps = np.max(voxel, axis=-1)

    anchors_map = data["anchors_map"]
    reg_targets = data["reg_targets"]
    pred_selected = data["result"]
    gt_max_iou_idx = data["gt_max_iou"]

    gt_corners, det_corners = [], []
    for p in range(pred_len):
        # p=0
        for k in range(len(pred_selected)):

            cls_pred_corners = pred_selected[k]["pred"][:, p]
            # cls_pred_scores = pred_selected[k]["score"]
            # cls_pred_idx = pred_selected[k]['selected_idx']
            if config.motion_state:
                # cls_pred_state = pred_selected[k]["motion"]
                pass

            for corner_id in range(cls_pred_corners.shape[0]):
                corner_box = cls_pred_corners[corner_id]

                corner = coor_to_vis(
                    corner_box, area_extents=area_extents, voxel_size=voxel_size
                )
                det_corners.append(
                    (
                        min(corner[:, 0]),
                        255 - min(corner[:, 1]),
                        max(corner[:, 0]) - min(corner[:, 0]),
                        max(corner[:, 1] - min(corner[:, 1])),
                    )
                )

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
            gt_corners.append(
                (
                    min(corner[:, 0]),
                    min(corner[:, 1]),
                    max(corner[:, 0]) - min(corner[:, 0]),
                    max(corner[:, 1]) - min(corner[:, 1]),
                )
            )
    return gt_corners, det_corners


# def get_det_corners(config, det_result):
#     voxel_size = config.voxel_size
#     area_extents = config.area_extents
#     anchor_size = config.anchor_size
#     map_dims = config.map_dims
#     pred_len=1
#     box_code_size = 6 #(x,y,w,h,sin,cos)

#     pred_selected = det_result

#     det_corners = []
#     for p in range(pred_len):
#         #p=0
#         for k in range(len(pred_selected)):

#             cls_pred_corners = pred_selected[k]['pred'][:, p]
#             cls_pred_scores = pred_selected[k]['score']
#             #cls_pred_idx = pred_selected[k]['selected_idx']
#             if config.motion_state:
#                 cls_pred_state = pred_selected[k]['motion']

#             for corner_id in range(cls_pred_corners.shape[0]):
#                 corner_box = cls_pred_corners[corner_id]

#                 corner = coor_to_vis(corner_box,area_extents = area_extents,voxel_size = voxel_size)
#                 # det_corners.append((min(corner[:, 0]), 255-min(corner[:, 1]), max(corner[:, 0])-min(corner[:, 0]), max(corner[:, 1]-min(corner[:, 1]))))
#                 det_corners.append((min(corner[:, 0]), min(corner[:, 1]), max(corner[:, 0])-min(corner[:, 0]), max(corner[:, 1])-min(corner[:,1])))

#     return det_corners


def get_det_corners(config, data, savename=None):
    voxel_size = config.voxel_size
    area_extents = config.area_extents
    # anchor_size = config.anchor_size
    # map_dims = config.map_dims
    pred_len = 1
    # box_code_size = 6  # (x,y,w,h,sin,cos)

    # voxel map visualization
    # voxel = data["bev_seq"]
    # maps = np.max(voxel, axis=-1)

    # anchors_map = data["anchors_map"]
    # reg_targets = data["reg_targets"]

    pred_selected = data["result"]
    # gt_max_iou_idx = data["gt_max_iou"]

    det_corners = []
    for p in range(pred_len):
        # p=0
        for k in range(len(pred_selected)):

            cls_pred_corners = pred_selected[k]["pred"][:, p]
            # cls_pred_scores = pred_selected[k]["score"]
            # cls_pred_idx = pred_selected[k]['selected_idx']
            if config.motion_state:
                # cls_pred_state = pred_selected[k]["motion"]
                pass

            for corner_id in range(cls_pred_corners.shape[0]):
                corner_box = cls_pred_corners[corner_id]

                corner = coor_to_vis(
                    corner_box, area_extents=area_extents, voxel_size=voxel_size
                )
                det_corners.append(
                    (
                        min(corner[:, 0]),
                        255 - min(corner[:, 1]),
                        max(corner[:, 0]) - min(corner[:, 0]),
                        max(corner[:, 1]) - min(corner[:, 1]),
                    )
                )

    return det_corners


def late_fusion(ego_agent, num_agent, result, trans_matrices, box_color_map):
    box_colors = np.array(
        [box_color_map[ego_agent] for _ in result[ego_agent][0][0][0]["pred"]]
    )

    for j in range(num_agent):
        if j == ego_agent or len(result[ego_agent]) == 0 or len(result[j]) == 0:
            continue

        trans_mat_j2ego = trans_matrices[0, ego_agent, j]

        # remove z-axis
        trans_mat_j2ego = np.delete(trans_mat_j2ego, 2, axis=1)
        trans_mat_j2ego = np.delete(trans_mat_j2ego, 2, axis=0)

        boxes_j = np.array(result[j][0][0][0]["pred"])  # [0][0][0] ==> squeeze
        points = boxes_j.reshape(-1, 2).T
        points[0, :] = -points[0, :]
        points = np.dot(trans_mat_j2ego, np.vstack((points, np.ones(points.shape[1]))))[
            :2, :
        ]
        points[0, :] = -points[0, :]
        points = points.T.reshape(-1, 1, 4, 2)

        result[ego_agent][0][0][0]["pred"] = np.vstack(
            (result[ego_agent][0][0][0]["pred"], points)
        )

        result[ego_agent][0][0][0]["score"] = np.append(
            result[ego_agent][0][0][0]["score"], result[j][0][0][0]["score"]
        )
        result[ego_agent][0][0][0]["selected_idx"] = np.append(
            result[ego_agent][0][0][0]["selected_idx"],
            result[j][0][0][0]["selected_idx"],
        )
        box_colors = np.append(box_colors, [box_color_map[j] for _ in points])

    # nms
    if len(result[ego_agent]) > 0:
        boxes = np.squeeze(result[ego_agent][0][0][0]["pred"])
        pick = non_max_suppression(boxes, result[ego_agent][0][0][0]["score"], 0.01)
        result[ego_agent][0][0][0]["pred"] = np.take(
            result[ego_agent][0][0][0]["pred"], pick, axis=0
        )
        box_colors = np.take(box_colors, pick, axis=0)

    return box_colors


def visualization(config, data, box_colors, box_color_map, apply_late_fusion, savename=None):
    voxel_size = config.voxel_size
    area_extents = config.area_extents
    # anchor_size = config.anchor_size
    # map_dims = config.map_dims
    pred_len = 1
    # box_code_size = 6  # (x,y,w,h,sin,cos)

    # voxel map visualization
    voxel = data["bev_seq"]
    maps = np.max(voxel, axis=-1)

    anchors_map = data["anchors_map"]
    # print(anchors_map.shape)
    # anchor_corners_list = get_anchor_corners_list(anchors_map,box_code_size)
    # anchor_corners_map = anchor_corners_list.reshape(map_dims[0],map_dims[1],len(anchor_size),4,2)
    reg_targets = data["reg_targets"]

    # 'pred':box_corners[selected_idx],'score': cls_preds[selected_idx,i]},'selected_idx': selected_idx
    pred_selected = data["result"]
    gt_max_iou_idx = data["gt_max_iou"]
    # if anchors_map.shape[2] < 7:#binary classification only has 4 anchors
    #    anchors_map = np.concatenate([anchors_map[:,:,:2],np.zeros_like(anchors_map[:,:,:3]),anchors_map[:,:,2:]],axis=2)

    #    reg_targets = np.concatenate([reg_targets[:,:,:2],np.zeros_like(reg_targets[:,:,:3]),reg_targets[:,:,2:]],axis=2)

    # plt.clf()
    if config.pred_type == "motion":
        cur_det = []
    for p in range(pred_len):
        # p=0
        for k in range(len(pred_selected)):

            cls_pred_corners = pred_selected[k]["pred"][:, p]
            # cls_pred_scores = pred_selected[k]["score"]
            # cls_pred_idx = pred_selected[k]['selected_idx']
            if config.motion_state:
                cls_pred_state = pred_selected[k]["motion"]

            for corner_id in range(cls_pred_corners.shape[0]):
                corner_box = cls_pred_corners[corner_id]

                corners = coor_to_vis(
                    corner_box, area_extents=area_extents, voxel_size=voxel_size
                )
                c_x, c_y = np.mean(corners, axis=0)
                corners = np.concatenate([corners, corners[[0]]])

                color = box_colors[corner_id] if apply_late_fusion == 1 else "red"

                if apply_late_fusion == 1:
                    legends = [mpatches.Patch(color=ii, label=f'Agent {idx}') for (idx, ii) in enumerate(box_color_map)]
                    plt.legend(handles=legends, loc='center left', bbox_to_anchor=(1, 0.5))

                if p == 0:
                    plt.plot(
                        corners[:, 0], corners[:, 1], c=color, linewidth=0.8, zorder=15
                    )
                    plt.scatter(c_x, c_y, s=3, c=color, zorder=15)
                    plt.plot(
                        [c_x, (corners[-2][0] + corners[0][0]) / 2.0],
                        [c_y, (corners[-2][1] + corners[0][1]) / 2.0],
                        linewidth=0.8,
                        c=color,
                        zorder=15,
                    )
                else:
                    if config.motion_state:
                        if cls_pred_state[corner_id] == 0:
                            continue
                    plt.scatter(c_x, c_y, s=3, c=color, zorder=15)

        for k in range(len(gt_max_iou_idx)):

            anchor = anchors_map[tuple(gt_max_iou_idx[k][:-1])]

            encode_box = reg_targets[tuple(gt_max_iou_idx[k][:-1]) + (p,)]
            if config.code_type[0] == "f":
                if config.pred_type == "motion":

                    # motion a
                    """
                    if p ==0:
                        decode_box = bev_box_decode_np(encode_box,anchor)
                        cur_det.append(decode_box)
                    else:
                        decode_box = cur_det[k].copy()
                        decode_box[:2] += encode_box[:2]
                    """

                    # motion b
                    if p == 0:
                        decode_box = bev_box_decode_np(encode_box, anchor)
                        cur_det.append(decode_box)
                    else:
                        decode_box = cur_det[k].copy()
                        decode_box[:2] += encode_box[:2]
                        cur_det[k] = decode_box.copy()

                else:
                    decode_box = bev_box_decode_np(encode_box, anchor)
                # print(decode_box)
                decode_corner = center_to_corner_box2d(
                    np.asarray([decode_box[:2]]),
                    np.asarray([decode_box[2:4]]),
                    np.asarray([decode_box[4:]]),
                )[0]
            # print(decode_corner)
            # exit()
            # decode_corner = center_to_corner_box2d(np.asarray([anchor[:2]]),np.asarray([anchor[2:4]]),np.asarray([anchor[4:]]))[0]
            elif config.code_type[0] == "c":
                # decoded_corner = (encode_box + anchor).reshape(-1, 4, 2)
                pass

            corners = coor_to_vis(
                decode_corner, area_extents=area_extents, voxel_size=voxel_size
            )
            c_x, c_y = np.mean(corners, axis=0)
            corners = np.concatenate([corners, corners[[0]]])

            if p == 0:

                plt.plot(corners[:, 0], corners[:, 1], c="g", linewidth=2, zorder=5)
                plt.scatter(c_x, c_y, s=5, c="g", zorder=5)
                # plt.scatter(corners[0,0], corners[0,1], s=10,c = 'r')
                plt.plot(
                    [c_x, (corners[-2][0] + corners[0][0]) / 2.0],
                    [c_y, (corners[-2][1] + corners[0][1]) / 2.0],
                    linewidth=2,
                    c="g",
                    zorder=5,
                )
            else:
                plt.scatter(c_x, c_y, s=5, linewidth=2, c="g", zorder=5)

    m = np.stack([maps, maps, maps], axis=-1)
    m[m == 0] = 0.99

    m1 = m[:, :, 0]
    m2 = m[:, :, 1]
    m3 = m[:, :, 2]

    m1[m1 == 1] = 78 / 255
    m2[m2 == 1] = 52 / 255
    m3[m3 == 1] = 112 / 255

    m = np.stack([m1, m2, m3], axis=-1)

    maps = (m * 255).astype(np.uint8)
    plt.imshow(maps, zorder=0)
    plt.xticks([])
    plt.yticks([])
    if savename is not None:
        plt.savefig(savename, dpi=500)
        plt.close(1)
    else:
        # plt.show()
        plt.pause(1)


def torch_to_np_dtype(ttype):
    type_map = {
        torch.float16: np.dtype(np.float16),
        torch.float32: np.dtype(np.float32),
        torch.float16: np.dtype(np.float64),
        torch.int32: np.dtype(np.int32),
        torch.int64: np.dtype(np.int64),
        torch.uint8: np.dtype(np.uint8),
    }
    return type_map[ttype]
