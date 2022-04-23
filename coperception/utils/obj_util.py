import numpy as np
from shapely.geometry import Polygon
from matplotlib import pyplot as plt
import sys
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
import math
import time


"""
redundent functions in postprocess.py
"""


def convert_format(boxes_array):
    """
    :param array: an array of shape [# bboxs, 4, 2]
    :return: a shapely.geometry.Polygon object
    """

    polygons = [
        Polygon([(box[i, 0], box[i, 1]) for i in range(4)]) for box in boxes_array
    ]

    return np.array(polygons)


def compute_overlaps_gen_gt(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: a np array of boxes
    For better performance, pass the largest set first and the smaller second.
    :return: a matrix of overlaps [boxes1 count, boxes2 count]
    """
    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.

    box1_center = np.mean(boxes1, axis=1)
    box2_center = np.mean(boxes2, axis=1)
    boxes2_poly = convert_format(boxes2)

    overlaps = np.zeros((len(boxes1), len(boxes2)))
    for i in range(overlaps.shape[1]):
        box_gt_poly = boxes2_poly[i]
        box_gt = boxes2[i]
        h = max(
            np.linalg.norm(box_gt[0] - box_gt[1]), np.linalg.norm(box_gt[1] - box_gt[2])
        )
        dis = np.linalg.norm(box1_center - box2_center[i], axis=1)
        # print('gt center: ',box2_center[i])
        idx = dis < max(1.0, h / 5.0)
        box_filter = convert_format(boxes1[idx])
        overlaps[idx, i] = compute_iou(box_gt_poly, box_filter)

        # debug
        """
        if i == 7:
            boxes = boxes1[overlaps[:,i]>0.4]
            for box in boxes:
                box = coor_to_vis(box)
                c=np.random.rand(3,)
                plt.plot(box[:,0], box[:,1], linewidth=1.0,c=c)
        """
        # print('qualified num: ',np.sum(overlaps[:,i]>0.4))
    return overlaps


def compute_iou(box, boxes):
    """Calculates IoU of the given box with the array of the given boxes.
    box: a polygon
    boxes: a vector of polygons
    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas

    iou = [box.intersection(b).area / box.union(b).area for b in boxes]

    return np.array(iou, dtype=np.float32)


#########################################################

"""
redundent functions in deteciton_util.py
"""


def bev_box_decode_np(box_encoding, anchor_info):
    """box decode for VoxelNet in lidar
    Args:
        boxes ([N, 6] Tensor): normal boxes: x, y, w, l, sin, cos
        anchors ([N, 6] Tensor): anchors
    """

    xa, ya, wa, ha, sina, cosa = anchor_info
    xp, yp, wp, hp, sinp, cosp = box_encoding

    h = ha / math.exp(hp)
    w = wa / math.exp(wp)

    x = xa - w * xp
    y = ya - h * yp

    sin = sina * cosp + cosa * sinp
    cos = cosa * cosp - sina * sinp

    decode_box = np.asarray([x, y, w, h, sin, cos])

    return decode_box


def bev_box_corner_decode_np(box_encoding, anchor_info):
    """box decode for VoxelNet in lidar
    Args:
        boxes ([N, 8] Tensor): normal boxes: x1,y1,x2,y2,x3,y3,x4,y4
        anchors ([N, 8] Tensor): anchors \delta{x1},\delta{y1},...
    """
    decode_box = box_encoding + anchor_info
    return decode_box


def bev_box_decode_np_corner_3(box_encoding, anchor_info):
    """box decode for VoxelNet in lidar
    Args:
        boxes ([N, 6] Tensor): normal boxes: x, y, w, l, sin, cos
        anchors ([N, 6] Tensor): anchors
    """

    xa, ya, x0, y0, x1, y1, x2, y2, x3, y3 = anchor_info
    dx, dy, dx0, dy0, dx1, dy1, dx2, dy2, dx3, dy3 = box_encoding

    x = xa + dx
    y = ya + dy

    x0 = x0 + dx0
    y0 = y0 + dy0
    x1 = x1 + dx1
    y1 = y1 + dy1
    x2 = x2 + dx2
    y2 = y2 + dy2
    x3 = x3 + dx3
    y3 = y3 + dy3

    decode_box = np.asarray([x, y, x0, y0, x1, y1, x2, y2, x3, y3])

    return decode_box.reshape(..., 5, 2)


def corners_nd(dims, origin=0.5):
    """generate relative box corners based on length per dim and
    origin point.

    Args:
        dims (float array, shape=[N, ndim]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.

    Returns:
        float array, shape=[N, 2 ** ndim, ndim]: returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1
    """
    ndim = int(dims.shape[1])
    corners_norm = np.stack(
        np.unravel_index(np.arange(2**ndim), [2] * ndim), axis=1
    ).astype(dims.dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start with minimum point
    # for 3d boxes, please draw lines by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape([1, 2**ndim, ndim])

    # Transfer to nuscenes format: (2d) 1st corner is the upper left corner
    # x0y1,x1y1,x1y0,x0y0

    corners = np.concatenate(
        [
            corners[:, [1], :],
            corners[:, [2], :],
            corners[:, [3], :],
            corners[:, [0], :],
        ],
        axis=1,
    )

    # exit()

    return corners


def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
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

    corners = corners_nd(dims, origin=origin)
    # corners: [N, 4, 2]
    if angles is not None:
        corners = rotation_2d(corners, angles)
    corners += centers.reshape(-1, 1, 2)

    return corners


def rotation_2d(points, angles):
    """rotation 2d points based on origin point clockwise when angle positive.

    Args:
        points (float array, shape=[N, point_size, 2]): points to be rotated.
        angles (float array, shape=[N, 2]): rotation angle. sin, cos

    Returns:
        float array: same shape as points
    """

    rot_sin = angles[:, 0]
    rot_cos = angles[:, 1]
    rot_mat_T = np.stack([np.stack([rot_cos, -rot_sin]), np.stack([rot_sin, rot_cos])])

    return np.einsum("aij,jka->aik", points, rot_mat_T)


#####################################################################
def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    See https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    a = 2.0 * (q[0] * q[3] + q[1] * q[2])
    b = 1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2)

    return np.arctan2(a, b)


def generate_gts(
    instance_map,
    instance_cat_list,
    instance_box_list,
    anchors_map,
    anchor_size,
    nsweeps_back,
    pred_len,
    map_dims,
    category_threshold,
    box_code_size,
    config,
):
    interval = 5
    offset_thre = config.static_thre  # 0.2/s
    instance_box_list = np.asarray(instance_box_list)

    instance_box_list = np.concatenate(
        [
            instance_box_list[:, [0]],
            instance_box_list[:, nsweeps_back : (nsweeps_back + (pred_len) * interval)],
        ],
        axis=1,
    )
    instance_box_list = instance_box_list[:, ::interval]

    label = np.zeros((map_dims[0], map_dims[1], len(anchor_size))).astype(np.int8)

    motion_labels = np.zeros((map_dims[0], map_dims[1], len(anchor_size))).astype(
        np.int8
    )

    reg_loss_mask = np.ones(
        (map_dims[0], map_dims[1], len(anchor_size), pred_len)
    ).astype(np.bool)
    reg_target = np.zeros(
        (map_dims[0], map_dims[1], len(anchor_size), pred_len, box_code_size)
    )
    for i in range(map_dims[0]):
        for j in range(map_dims[1]):
            for k in range(len(anchor_size)):

                # Binary classification:
                if config.binary:
                    check = (
                        instance_map[i, j, k] >= 0
                        and instance_cat_list[instance_map[i, j, k]] == 1
                    )
                else:
                    check = instance_map[i, j, k] >= 0

                if check:
                    instance_id = instance_map[i, j, k]

                    label[i, j, k] = instance_cat_list[instance_id]

                    # generate reg_target

                    instance_box = instance_box_list[instance_id]

                    false_frame = []
                    for box in range(1, pred_len):
                        if (
                            np.sum(np.isnan(instance_box[box])) > 0
                            or np.sum((instance_box[box])) == 0
                            or (
                                np.max(np.abs(instance_box[box][:2]))
                                > (32 + np.max(anchor_size[:, :2] / 2.0))
                            )
                        ):

                            instance_box[box] = np.zeros(instance_box[0].shape)
                            reg_loss_mask[i][j][k][box] = False
                            false_frame.append(box)
                    anchor_box = anchors_map[i][j][k]

                    sin = math.sin(anchor_size[k][2])
                    cos = math.cos(anchor_size[k][2])

                    x_list = (-instance_box[:pred_len, 0] + anchor_box[0]) / (
                        instance_box[:pred_len, 3] + 1e-6
                    )
                    y_list = (-instance_box[:pred_len, 1] + anchor_box[1]) / (
                        instance_box[:pred_len, 4] + 1e-6
                    )

                    if config.motion_state:
                        # print(instance_box[:pred_len][reg_loss_mask[i][j][k][1:],:2].shape)
                        center_collect = instance_box[1:pred_len][
                            reg_loss_mask[i][j][k][1:], :2
                        ]
                        if center_collect.shape[0] < 1:
                            motion_labels[i, j, k] = 0  # dont care
                        else:
                            max_idx = np.argmax(
                                np.linalg.norm(
                                    (center_collect - instance_box[0, :2]), axis=-1
                                )
                            )
                            offset = np.max(
                                np.linalg.norm(
                                    (center_collect - instance_box[0, :2]), axis=-1
                                )
                            )
                            velocity = offset / ((max_idx + 1.0) / pred_len)

                            if velocity > offset_thre:
                                motion_labels[i, j, k] = 2  # moving
                            else:
                                motion_labels[i, j, k] = 1  # static

                    if config.pred_type == "motion":
                        """
                        #motion_a
                        x_list[1:] = instance_box[1:pred_len,1]-instance_box[0,1]
                        y_list[1:] = instance_box[1:pred_len,0]-instance_box[0,0]
                        """
                        # motion_b

                        for p_id in range(1, pred_len):
                            x_list[p_id] = (
                                instance_box[p_id, 0] - instance_box[p_id - 1, 0]
                            )
                            y_list[p_id] = (
                                instance_box[p_id, 1] - instance_box[p_id - 1, 1]
                            )

                    w_list = np.log(anchor_box[2] / (instance_box[:pred_len, 3] + 1e-6))
                    h_list = np.log(anchor_box[3] / (instance_box[:pred_len, 4] + 1e-6))
                    sin_list = []
                    cos_list = []
                    # degree_list = []
                    for idx in range(pred_len):
                        box = instance_box[idx]
                        orientation = Quaternion(box[6:])

                        rad = quaternion_yaw(orientation) - math.pi / 2.0
                        # degree_list.append(orientation.degrees)
                        sin_list.append(math.sin(rad) * (-1.0))
                        cos_list.append(math.cos(rad))

                    reg_target[i, j, k, :, 0] = x_list
                    reg_target[i, j, k, :, 1] = y_list
                    reg_target[i, j, k, :, 2] = w_list
                    reg_target[i, j, k, :, 3] = h_list
                    reg_target[i, j, k, :, 4] = cos * np.asarray(
                        sin_list
                    ) - sin * np.asarray(cos_list)
                    reg_target[i, j, k, :, 5] = cos * np.asarray(
                        cos_list
                    ) + sin * np.asarray(sin_list)

                    for idx in false_frame:
                        reg_target[i, j, k, idx] = 0.0
                else:
                    reg_loss_mask[i, j, k] = False
    return label, reg_target, reg_loss_mask, motion_labels


def get_center(shape):
    centers = np.zeros((shape[0], shape[1], 2))
    for w in range(shape[0]):
        for h in range(shape[1]):
            centers[w][h][0] = w * voxel_size[0] + voxel_size[0] / 2.0
            centers[w][h][1] = h * voxel_size[1] + voxel_size[1] / 2.0

    return centers


def encode_anchor_by_corner(corner, anchor_size):
    # corner (anchors_size,4,2)
    anchors = np.zeros(len(anchor_size), 8)
    anchors[:, :2] = corner[:, 0]
    anchors[:, 2:4] = corner[:, 1]
    anchors[:, 4:6] = corner[:, 2]
    anchors[:, 6:] = corner[:, 3]

    return anchors


def encode_anchor_by_center(center, area_extents, anchor_size):
    anchors = []

    for size in anchor_size:
        w = min(area_extents[0][1] - center[0] + size[0] / 2.0, size[0])
        h = min(area_extents[1][1] - center[1] + size[1] / 2.0, size[1])
        # print(w,h)
        anchors.append(
            np.asarray(
                [center[0], center[1], w, h, math.sin(size[2]), math.cos(size[2])]
            )
        )
    # exit()

    return np.asarray(anchors)


def coor_to_vis(coor, area_extents, voxel_size):

    if len(coor) == 6:  # box code
        x, y, w, h, sin, cos = coor

        x = (x - area_extents[0][0]) / voxel_size[0]
        y = (y - area_extents[1][0]) / voxel_size[1]
        w = w / voxel_size[0]
        h = h / voxel_size[1]
        out = np.asarray([x, y, w, h, sin, cos])
    elif len(coor) < 6:  # corners or polygons
        coor = np.asarray(coor)
        coor[:, 0] = (coor[:, 0] - area_extents[0][0]) / voxel_size[0]
        coor[:, 1] = (coor[:, 1] - area_extents[1][0]) / voxel_size[1]
        out = coor

    return out


def init_anchors(area_extents, voxel_size, box_code_size, anchor_size):
    # Output shape [H,W,num_per_loc,code_size]

    w_range = math.ceil((area_extents[0][1] - area_extents[0][0]) / voxel_size[0])
    h_range = math.ceil((area_extents[1][1] - area_extents[1][0]) / voxel_size[1])
    anchor_maps = np.zeros((h_range, w_range, len(anchor_size), box_code_size))

    for i in range(h_range):
        for j in range(w_range):
            center = [
                j * voxel_size[0] + area_extents[0][0] + voxel_size[0] / 2.0,
                i * voxel_size[0] + area_extents[0][0] + voxel_size[1] / 2.0,
            ]
            anchor_maps[i][j] = encode_anchor_by_center(center)

    return anchor_maps


def init_anchors_no_check(area_extents, voxel_size, box_code_size, anchor_size):
    # Output shape [H,W,num_per_loc,code_size]

    w_range = math.ceil((area_extents[0][1] - area_extents[0][0]) / voxel_size[0])
    h_range = math.ceil((area_extents[1][1] - area_extents[1][0]) / voxel_size[1])
    anchor_maps = np.zeros((h_range, w_range, len(anchor_size), box_code_size))

    anchor_maps[:, :, :, 2:4] = anchor_size[:, :2]
    anchor_maps[:, :, :, 4] = np.sin(anchor_size[:, 2])
    anchor_maps[:, :, :, 5] = np.cos(anchor_size[:, 2])
    for i in range(h_range):
        for j in range(w_range):
            center = np.asarray(
                [
                    j * voxel_size[0] + area_extents[0][0] + voxel_size[0] / 2.0,
                    i * voxel_size[0] + area_extents[0][0] + voxel_size[1] / 2.0,
                ]
            )
            anchor_maps[i, j, :, :2] = np.asarray(
                [center for _ in range(len(anchor_size))]
            )

    return anchor_maps


def get_anchor_corners_list(anchors_map, box_code_size):
    anchors_list = anchors_map.reshape(-1, box_code_size)
    corner_list = center_to_corner_box2d(
        anchors_list[:, :2], anchors_list[:, 2:4], anchors_list[:, 4:]
    )

    return corner_list


def init_category_gt(shape):
    # Output shape [H,W,num_per_loc]

    # cat_gt = np.zeros((shape[0], shape[1], idx_len, len(anchor_size)))
    # return cat_gt
    raise NotImplementedError("method init_category_gt in obj_util is not implemented")


def init_reg_gt(shape):
    # reg_gt = np.zeros(
    #     (shape[0], shape[1], pred_len, idx_len, len(anchor_size), box_code_size)
    # )
    # return reg_gt
    raise NotImplementedError("method init_reg_gt in obj_util is not implemented")


def get_gt_corners(gt_box):
    orientation = Quaternion(gt_box[6:])
    tmp_box = Box(center=gt_box[:3], size=gt_box[3:6], orientation=orientation)
    corners = tmp_box.corners()
    corners_2d = np.concatenate(
        [corners[:, 2:4], corners[:, [7]], corners[:, [6]]], axis=1
    )[:2]
    corners_2d = corners_2d.swapaxes(0, 1)

    """
    tmp = corners_2d[:,0].copy()
    corners_2d[:,0] = corners_2d[:,1]
    corners_2d[:,1] = tmp
    """

    return np.asarray(corners_2d)


def get_gt_corners_list(box_list):
    box_list = np.asarray(box_list)
    out = []
    if len(box_list.shape) > 2:  # (N,T,code)
        for i in range(len(box_list)):
            out.append(get_gt_corners(box_list[i][0]))

    elif len(box_list.shape) == 2:
        for i in range(len(box_list)):
            out.append(get_gt_corners(box_list[i]))

    return np.asarray(out)


if __name__ == "__main__":

    binary = True
    voxel_size = (0.25, 0.25, 0.4)
    area_extents = np.array([[-32.0, 32.0], [-32.0, 32.0], [-3.0, 2.0]])
    if binary:
        anchor_size = np.asarray(
            [
                [4.0, 2.0, 0],
                [4.0, 2.0, math.pi / 2.0],
                [4.0, 2.0, -math.pi / 4.0],
                [12.0, 3.0, 0],
                [12.0, 3.0, math.pi / 2.0],
                [12.0, 3.0, -math.pi / 4.0],
            ]
        )

    else:
        anchor_size = np.asarray(
            [
                [4.0, 2.0, 0],
                [4.0, 2.0, math.pi / 2.0],
                [1.0, 1.0, 0],
                [2.0, 1.0, 0.0],
                [2.0, 1.0, math.pi / 2.0],
                [12.0, 3.0, 0.0],
                [12.0, 3.0, math.pi / 2.0],
            ]
        )
    map_dims = [
        int((area_extents[0][1] - area_extents[0][0]) / voxel_size[0]),
        int((area_extents[1][1] - area_extents[1][0]) / voxel_size[1]),
    ]
    pred_len = 5
    code_type = "corner"
    if code_type == "corner":
        box_code_size = 8
    else:

        box_code_size = 6  # (x,y,w,h,sin,cos)

    data = np.load(sys.argv[1], allow_pickle=True).item()
    reg_mask = data["reg_loss_mask"].astype(np.bool)

    reg_loss_mask = data["reg_loss_mask"].astype(np.bool)

    # reg_mask = np.max(reg_mask,axis=-1)
    v = data["voxel_indices_4"]
    t = time.time()
    m = np.zeros((map_dims[0], map_dims[1], 13))
    m[v[:, 0], v[:, 1], v[:, 2]] = 1
    m = np.max(m, axis=2)
    anchors_map = init_anchors_no_check(
        area_extents, voxel_size, box_code_size, anchor_size
    )
    anchor_corners_list = get_anchor_corners_list(anchors_map, box_code_size)

    reg_target_sparse = data["reg_target_sparse"]
    label = data["label_sparse"]
    mask = data["allocation_mask"].astype(np.bool)
    gt_max_iou_idx = data["gt_max_iou"]
    # print(gt_max_iou_idx)

    reg_target = np.zeros((256, 256, len(anchor_size), 5, box_code_size))
    reg_target[mask] = reg_target_sparse
    reg_target[np.bitwise_not(reg_loss_mask)] = 0

    # print(np.sum(np.isnan(reg_target)))
    # exit()

    reg_mask = np.max(reg_mask[:, :, :, 4], axis=-1)

    # ======================GT Generation code==========================
    anchor_corners_map = anchor_corners_list.reshape(
        map_dims[0], map_dims[1], len(anchor_size), 4, 2
    )
    gt_corners = []
    reg_anchors = []
    pred_len = 1
    if code_type == "faf":
        for p in range(pred_len):

            for k in range(len(gt_max_iou_idx)):
                anchor = anchors_map[tuple(gt_max_iou_idx[k][:-1])]
                reg_anchor = anchor_corners_map[tuple(gt_max_iou_idx[k][:-1])]
                encode_box = reg_target[tuple(gt_max_iou_idx[k][:-1]) + (p,)]
                decode_box = bev_box_decode_np(encode_box, anchor)
                decode_corner = center_to_corner_box2d(
                    np.asarray([decode_box[:2]]),
                    np.asarray([decode_box[2:4]]),
                    np.asarray([decode_box[4:]]),
                )[0]
                gt_corners.append(decode_corner)
                reg_anchors.append(reg_anchor)

        gt_corners = np.asarray(gt_corners)
        reg_anchors = np.asarray(reg_anchors)
        reg_target_corners = reg_anchors - gt_corners
        reg_target_corners = reg_target_corners.reshape(reg_target_corners.shape[0], -1)

    # ===================Visualize code==========================
    # check reg target

    if code_type == "corner":

        for p in range(pred_len):
            # p = 0
            """
            for i in range(256):
                for j in range(256):
                    for k in range(len(anchor_size)):
                        if reg_loss_mask[i][j][k][p]:
                            anchor = anchors_map[i][j][k]
                            encode_box = reg_target[i][j][k][p]
                            decode_box = bev_box_decode_np(encode_box,anchor)
                            #print(decode_box)

                            decode_corner = center_to_corner_box2d(np.asarray([decode_box[:2]]),np.asarray([decode_box[2:4]]),np.asarray([decode_box[4:]]))[0]
                            #decode_corner = center_to_corner_box2d(np.asarray([anchor[:2]]),np.asarray([anchor[2:4]]),np.asarray([anchor[4:]]))[0]

                            corners = coor_to_vis(decode_corner,area_extents,voxel_size)
                            c_x,c_y = np.mean(corners,axis=0)
                            corners = np.concatenate([corners,corners[[0]]])
                            gt_color = np.asarray([0,106,128])/255.
                            plt.plot(corners[:,0], corners[:,1], c=gt_color,linewidth=1.0)
                            plt.scatter(c_x, c_y, s=10,c = [gt_color])
                            #plt.scatter(corners[0,0], corners[0,1], s=10,c = 'r')
                            plt.plot([c_x,(corners[-2][0]+corners[0][0])/2.],[c_y,(corners[-2][1]+corners[0][1])/2.],linewidth=1.0,c=gt_color)

            """
            for k in range(len(gt_max_iou_idx)):
                anchor_corner = anchor_corners_map[tuple(gt_max_iou_idx[k][:-1])]

                encode_box = reg_target[tuple(gt_max_iou_idx[k][:-1]) + (p,)].reshape(
                    4, 2
                )

                decode_corner = encode_box + anchor_corner

                corners = coor_to_vis(decode_corner, area_extents, voxel_size)
                c_x, c_y = np.mean(corners, axis=0)
                corners = np.concatenate([corners, corners[[0]]])
                gt_color = np.asarray([0, 106, 128]) / 255.0
                plt.plot(corners[:, 0], corners[:, 1], c=gt_color, linewidth=1.0)
                plt.scatter(c_x, c_y, s=10, c=[gt_color])
                # plt.scatter(corners[0,0], corners[0,1], s=10,c = 'r')
                plt.plot(
                    [c_x, (corners[-2][0] + corners[0][0]) / 2.0],
                    [c_y, (corners[-2][1] + corners[0][1]) / 2.0],
                    linewidth=1.0,
                    c=gt_color,
                )

            break

    else:
        for p in range(pred_len):
            # p = 0
            """
            for i in range(256):
                for j in range(256):
                    for k in range(len(anchor_size)):
                        if reg_loss_mask[i][j][k][p]:
                            anchor = anchors_map[i][j][k]
                            encode_box = reg_target[i][j][k][p]
                            decode_box = bev_box_decode_np(encode_box,anchor)
                            #print(decode_box)

                            decode_corner = center_to_corner_box2d(np.asarray([decode_box[:2]]),np.asarray([decode_box[2:4]]),np.asarray([decode_box[4:]]))[0]
                            #decode_corner = center_to_corner_box2d(np.asarray([anchor[:2]]),np.asarray([anchor[2:4]]),np.asarray([anchor[4:]]))[0]

                            corners = coor_to_vis(decode_corner,area_extents,voxel_size)
                            c_x,c_y = np.mean(corners,axis=0)
                            corners = np.concatenate([corners,corners[[0]]])
                            gt_color = np.asarray([0,106,128])/255.
                            plt.plot(corners[:,0], corners[:,1], c=gt_color,linewidth=1.0)
                            plt.scatter(c_x, c_y, s=10,c = [gt_color])
                            #plt.scatter(corners[0,0], corners[0,1], s=10,c = 'r')
                            plt.plot([c_x,(corners[-2][0]+corners[0][0])/2.],[c_y,(corners[-2][1]+corners[0][1])/2.],linewidth=1.0,c=gt_color)

            """
            for k in range(len(gt_max_iou_idx)):
                anchor = anchors_map[tuple(gt_max_iou_idx[k][:-1])]

                encode_box = reg_target[tuple(gt_max_iou_idx[k][:-1]) + (p,)]
                # print(encode_box)
                # exit()

                decode_box = bev_box_decode_np(encode_box, anchor)
                # print(decode_box)

                decode_corner = center_to_corner_box2d(
                    np.asarray([decode_box[:2]]),
                    np.asarray([decode_box[2:4]]),
                    np.asarray([decode_box[4:]]),
                )[0]
                # decode_corner = center_to_corner_box2d(np.asarray([anchor[:2]]),np.asarray([anchor[2:4]]),np.asarray([anchor[4:]]))[0]

                corners = coor_to_vis(decode_corner, area_extents, voxel_size)
                c_x, c_y = np.mean(corners, axis=0)
                corners = np.concatenate([corners, corners[[0]]])
                gt_color = np.asarray([0, 106, 128]) / 255.0
                plt.plot(corners[:, 0], corners[:, 1], c=gt_color, linewidth=1.0)
                plt.scatter(c_x, c_y, s=10, c=[gt_color])
                # plt.scatter(corners[0,0], corners[0,1], s=10,c = 'r')
                plt.plot(
                    [c_x, (corners[-2][0] + corners[0][0]) / 2.0],
                    [c_y, (corners[-2][1] + corners[0][1]) / 2.0],
                    linewidth=1.0,
                    c=gt_color,
                )
            break
    m = np.stack([m, m, m], axis=-1)
    m[m == 0] = 0.99
    m[m == 1] = 0.5
    maps = (m * 255).astype(np.uint8)
    # maps = m
    # print(maps[128])
    plt.imshow(maps)
    # plt.imshow(reg_mask)
    plt.show()
