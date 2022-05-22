# Copyright (c) 2020 Mitsubishi Electric Research Laboratories (MERL). All rights reserved. The software, documentation and/or data in this file is provided on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications. MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for any particular purpose. In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages. As more fully described in the license agreement that was required in order to download this software, documentation and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, research and non-commercial purposes.
import numpy as np
import os
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from numba import njit


def point_in_hull_slow(point, hull, tolerance=1e-12):
    """
    Check if a point lies in a convex hull. This implementation is slow.
    :param point: nd.array (1 x d); d: point dimension
    :param hull: The scipy ConvexHull object
    :param tolerance: Used to compare to a small positive constant because of issues of numerical precision
    (otherwise, you may find that a vertex of the convex hull is not in the convex hull)
    """
    return all((np.dot(eq[:-1], point) + eq[-1] <= tolerance) for eq in hull.equations)


def point_in_hull_fast(points: np.array, bounding_box: Box):
    """
    Check if a point lies in a bounding box. We first rotate the bounding box to align with axis. Meanwhile, we
    also rotate the whole point cloud. Finally, we just check the membership with the aid of aligned axis.
    This implementation is fast.
    :param points: nd.array (N x d); N: the number of points, d: point dimension
    :param bounding_box: the Box object
    return: The membership of points within the bounding box
    """
    # Make sure it is a unit quaternion
    bounding_box.orientation = bounding_box.orientation.normalised

    # Rotate the point clouds
    pc = bounding_box.orientation.inverse.rotation_matrix @ points.T
    pc = pc.T

    orientation_backup = Quaternion(bounding_box.orientation)  # Deep clone it
    bounding_box.rotate(bounding_box.orientation.inverse)

    corners = bounding_box.corners()

    # Test if the points are in the bounding box
    idx = np.where(
        (corners[0, 7] <= pc[:, 0])
        & (pc[:, 0] <= corners[0, 0])
        & (corners[1, 1] <= pc[:, 1])
        & (pc[:, 1] <= corners[1, 0])
        & (corners[2, 2] <= pc[:, 2])
        & (pc[:, 2] <= corners[2, 0])
    )[0]

    # recover
    bounding_box.rotate(orientation_backup)

    return idx


def calc_displace_vector(points: np.array, curr_box: Box, next_box: Box):
    """
    Calculate the displacement vectors for the input points.
    This is achieved by comparing the current and next bounding boxes. Specifically, we first rotate
    the input points according to the delta rotation angle, and then translate them. Finally we compute the
    displacement between the transformed points and the input points.
    :param points: The input points, (N x d). Note that these points should be inside the current bounding box.
    :param curr_box: Current bounding box.
    :param next_box: The future next bounding box in the temporal sequence.
    :return: Displacement vectors for the points.
    """
    assert points.shape[1] == 3, "The input points should have dimension 3."

    # Make sure the quaternions are normalized
    curr_box.orientation = curr_box.orientation.normalised
    next_box.orientation = next_box.orientation.normalised

    delta_rotation = curr_box.orientation.inverse * next_box.orientation
    rotated_pc = (delta_rotation.rotation_matrix @ points.T).T
    rotated_curr_center = np.dot(delta_rotation.rotation_matrix, curr_box.center)
    delta_center = next_box.center - rotated_curr_center

    rotated_tranlated_pc = rotated_pc + delta_center

    pc_displace_vectors = rotated_tranlated_pc - points

    return pc_displace_vectors


def get_static_and_moving_cells(batch_disp_field_gt, upper_thresh=0.1, frame_skip=3):
    """
    Get the indices/masks of static and moving cells. Ths speed of static cells is bounded by upper_thresh.
    In particular, for a given cell, if its displacement over the past 1 second (about 20 sample data) is in the
    range [0, upper_thresh], we consider it as static cell, otherwise as moving cell.
    :param batch_disp_field_gt: Batch of ground-truth displacement fields. numpy array, shape (seq len, h, w)
    :param upper_thresh: The speed upper bound
    :param frame_skip: The number of skipped frame in the sweep sequence. This is used for computing the upper bound
        for defining static objects
    """
    upper_bound = (frame_skip + 1) / 20 * upper_thresh
    disp_norm = np.linalg.norm(batch_disp_field_gt, ord=2, axis=-1)

    static_cell_mask = disp_norm <= upper_bound
    static_cell_mask = np.all(static_cell_mask, axis=0)  # along the sequence axis

    moving_cell_mask = np.logical_not(static_cell_mask)

    return static_cell_mask, moving_cell_mask


def voxelize(pts, voxel_size, extents=None, num_T=35, seed: float = None):
    """
    Voxelize the input point cloud. Code modified from https://github.com/Yc174/voxelnet
    Voxels are 3D grids that represent occupancy info.

    The input for the voxelization is expected to be a PointCloud
    with N points in 4 dimension (x,y,z,i). Voxel size is the quantization size for the voxel grid.

    voxel_size: I.e. if voxel size is 1 m, the voxel space will be
    divided up within 1m x 1m x 1m space. This space will be -1 if free/occluded and 1 otherwise.
    min_voxel_coord: coordinates of the minimum on each axis for the voxel grid
    max_voxel_coord: coordinates of the maximum on each axis for the voxel grid
    num_divisions: number of grids in each axis
    leaf_layout: the voxel grid of size (numDivisions) that contain -1 for free, 0 for occupied

    :param pts: Point cloud as N x [x, y, z, i]
    :param voxel_size: Quantization size for the grid, vd, vh, vw
    :param extents: Optional, specifies the full extents of the point cloud.
                    Used for creating same sized voxel grids. Shape (3, 2)
    :param num_T: Number of points in each voxel after sampling/padding
    :param seed: The random seed for fixing the data generation.
    """
    # Check if points are 3D, otherwise early exit
    if pts.shape[1] < 3 or pts.shape[1] > 4:
        raise ValueError("Points have the wrong shape: {}".format(pts.shape))

    if extents is not None:
        if extents.shape != (3, 2):
            raise ValueError("Extents are the wrong shape {}".format(extents.shape))

        filter_idx = np.where(
            (extents[0, 0] < pts[:, 0])
            & (pts[:, 0] < extents[0, 1])
            & (extents[1, 0] < pts[:, 1])
            & (pts[:, 1] < extents[1, 1])
            & (extents[2, 0] < pts[:, 2])
            & (pts[:, 2] < extents[2, 1])
        )[0]
        pts = pts[filter_idx]

    # Discretize voxel coordinates to given quantization size
    discrete_pts = np.floor(pts[:, :3] / voxel_size).astype(np.int32)

    # Use Lex Sort, sort by x, then y, then z
    x_col = discrete_pts[:, 0]
    y_col = discrete_pts[:, 1]
    z_col = discrete_pts[:, 2]
    sorted_order = np.lexsort((z_col, y_col, x_col))

    # Save original points in sorted order
    points = pts[sorted_order]
    discrete_pts = discrete_pts[sorted_order]

    # Format the array to c-contiguous array for unique function
    contiguous_array = np.ascontiguousarray(discrete_pts).view(
        np.dtype((np.void, discrete_pts.dtype.itemsize * discrete_pts.shape[1]))
    )

    # The new coordinates are the discretized array with its unique indexes
    _, unique_indices = np.unique(contiguous_array, return_index=True)

    # Sort unique indices to preserve order
    unique_indices.sort()

    voxel_coords = discrete_pts[unique_indices]

    # Number of points per voxel, last voxel calculated separately
    num_points_in_voxel = np.diff(unique_indices)
    num_points_in_voxel = np.append(
        num_points_in_voxel, discrete_pts.shape[0] - unique_indices[-1]
    )

    # Compute the minimum and maximum voxel coordinates
    if extents is not None:
        min_voxel_coord = np.floor(extents.T[0] / voxel_size)
        max_voxel_coord = np.ceil(extents.T[1] / voxel_size) - 1
    else:
        min_voxel_coord = np.amin(voxel_coords, axis=0)
        max_voxel_coord = np.amax(voxel_coords, axis=0)

    # Get the voxel grid dimensions
    num_divisions = ((max_voxel_coord - min_voxel_coord) + 1).astype(np.int32)

    # Bring the min voxel to the origin
    voxel_indices = (voxel_coords - min_voxel_coord).astype(int)

    # Padding the points within each voxel
    padded_voxel_points = np.zeros(
        [unique_indices.shape[0], num_T, pts.shape[1] + 3], dtype=np.float32
    )
    padded_voxel_points = padding_voxel(
        padded_voxel_points, unique_indices, num_points_in_voxel, points, num_T, seed
    )

    return padded_voxel_points, voxel_indices, num_divisions


@njit
def padding_voxel(
    padded_voxel_points, unique_indices, num_points_in_voxel, points, num_T, seed
):
    if seed is not None:
        np.random.seed(seed)
    for i, v in enumerate(zip(unique_indices, num_points_in_voxel)):
        if v[1] < num_T:
            padded_voxel_points[i, : v[1], :4] = points[v[0] : v[0] + v[1], :]
            middle_points_x = np.mean(points[v[0] : v[0] + v[1], 0])
            middle_points_y = np.mean(points[v[0] : v[0] + v[1], 1])
            middle_points_z = np.mean(points[v[0] : v[0] + v[1], 2])
            padded_voxel_points[i, : v[1], 4] = (
                padded_voxel_points[i, : v[1], 0] - middle_points_x
            )
            padded_voxel_points[i, : v[1], 5] = (
                padded_voxel_points[i, : v[1], 1] - middle_points_y
            )
            padded_voxel_points[i, : v[1], 6] = (
                padded_voxel_points[i, : v[1], 2] - middle_points_z
            )
        else:
            inds = np.random.choice(v[1], num_T)
            padded_voxel_points[i, :, :4] = points[v[0] + inds, :]
            middle_points_x = np.mean(points[v[0] + inds, 0])
            middle_points_y = np.mean(points[v[0] + inds, 1])
            middle_points_z = np.mean(points[v[0] + inds, 2])
            padded_voxel_points[i, :, 4] = (
                padded_voxel_points[i, :, 0] - middle_points_x
            )
            padded_voxel_points[i, :, 5] = (
                padded_voxel_points[i, :, 1] - middle_points_y
            )
            padded_voxel_points[i, :, 6] = (
                padded_voxel_points[i, :, 2] - middle_points_z
            )

    return padded_voxel_points


def gen_2d_grid_gt(
    data_dict: dict,
    grid_size: np.array,
    extents: np.array = None,
    frame_skip: int = 0,
    reordered: bool = False,
    proportion_thresh: float = 0.5,
    category_num: int = 5,
    one_hot_thresh: float = 0.8,
    h_flip: bool = False,
    min_point_num_per_voxel: int = -1,
    return_past_2d_disp_gt: bool = False,
    return_instance_map: bool = False,
):
    """
    Generate the 2d grid ground-truth for the input point cloud.
    The ground-truth is: the displacement vectors of the occupied pixels in BEV image.
    The displacement is computed w.r.t the current time and the future time
    :param data_dict: The dictionary containing the data information
    :param grid_size: The size of each pixel
    :param extents: The extents of the point cloud on the 2D xy plane. Shape (3, 2)
    :param frame_skip: The number of sample frames that need to be skipped
    :param reordered: Whether need to reorder the results, so that the first element corresponds to the oldest past
        record. This option is only effective when return_past_2d_disp_gt = True.
    :param proportion_thresh: Within a given pixel, only when the proportion of foreground points exceeds this threshold
        will we compute the displacement vector for this pixel.
    :param category_num: The number of categories for points.
    :param one_hot_thresh: When the proportion of the majority points within a cell exceeds this threshold, we
        compute the (hard) one-hot category vector for this cell, otherwise compute the soft category vector.
    :param h_flip: Flip the point clouds horizontally
    :param min_point_num_per_voxel: Minimum point number inside each voxel (cell). If smaller than this threshold, we
        do not compute the displacement vector for this voxel (cell), and set the displacement to zero.
    :param return_past_2d_disp_gt: Whether to compute the ground-truth displacement filed for the past sweeps.
    :param return_instance_map: Whether to return the instance id map.
    :return: The ground-truth displacement field. Shape (num_sweeps, image height, image width, 2).
    """
    num_sweeps = data_dict["num_sweeps"]
    times = data_dict["times"]
    num_past_sweeps = len(np.where(times >= 0)[0])
    num_future_sweeps = len(np.where(times < 0)[0])
    assert (
        num_past_sweeps + num_future_sweeps == num_sweeps
    ), "The number of sweeps is incorrect!"

    pc_list = []

    for i in range(num_sweeps):
        pc = data_dict["pc_" + str(i)]
        if h_flip:
            pc[0, :] = -pc[0, :]  # flip x coordinate
        pc_list.append(pc.T)

    # Retrieve the instance boxes
    num_instances = data_dict["num_instances"]
    instance_box_list = list()
    instance_cat_list = list()  # for instance categories

    for i in range(num_instances):
        instance = data_dict["instance_boxes_" + str(i)]
        category = data_dict["category_" + str(i)]
        instance_box_list.append(instance)
        instance_cat_list.append(category)

    # ----------------------------------------------------
    # Filter and sort the reference point cloud
    refer_pc = pc_list[0]
    refer_pc = refer_pc[:, 0:3]

    if extents is not None:
        if extents.shape != (3, 2):
            raise ValueError("Extents are the wrong shape {}".format(extents.shape))

        filter_idx = np.where(
            (extents[0, 0] < refer_pc[:, 0])
            & (refer_pc[:, 0] < extents[0, 1])
            & (extents[1, 0] < refer_pc[:, 1])
            & (refer_pc[:, 1] < extents[1, 1])
            & (extents[2, 0] < refer_pc[:, 2])
            & (refer_pc[:, 2] < extents[2, 1])
        )[0]
        refer_pc = refer_pc[filter_idx]

    # -- Discretize pixel coordinates to given quantization size
    discrete_pts = np.floor(refer_pc[:, 0:2] / grid_size).astype(np.int32)

    # -- Use Lex Sort, sort by x, then y
    x_col = discrete_pts[:, 0]
    y_col = discrete_pts[:, 1]
    sorted_order = np.lexsort((y_col, x_col))

    refer_pc = refer_pc[sorted_order]
    discrete_pts = discrete_pts[sorted_order]

    contiguous_array = np.ascontiguousarray(discrete_pts).view(
        np.dtype((np.void, discrete_pts.dtype.itemsize * discrete_pts.shape[1]))
    )

    # -- The new coordinates are the discretized array with its unique indexes
    _, unique_indices = np.unique(contiguous_array, return_index=True)

    # -- Sort unique indices to preserve order
    unique_indices.sort()
    pixel_coords = discrete_pts[unique_indices]

    # -- Number of points per voxel, last voxel calculated separately
    num_points_in_pixel = np.diff(unique_indices)
    num_points_in_pixel = np.append(
        num_points_in_pixel, discrete_pts.shape[0] - unique_indices[-1]
    )

    # -- Compute the minimum and maximum voxel coordinates
    if extents is not None:
        min_pixel_coord = np.floor(extents.T[0, 0:2] / grid_size)
        max_pixel_coord = np.ceil(extents.T[1, 0:2] / grid_size) - 1
    else:
        min_pixel_coord = np.amin(pixel_coords, axis=0)
        max_pixel_coord = np.amax(pixel_coords, axis=0)

    # -- Get the voxel grid dimensions
    num_divisions = ((max_pixel_coord - min_pixel_coord) + 1).astype(np.int32)

    # -- Bring the min voxel to the origin
    pixel_indices = (pixel_coords - min_pixel_coord).astype(int)
    # ----------------------------------------------------

    # ----------------------------------------------------
    # Get the point cloud subsets, which are inside different instance bounding boxes
    refer_box_list = list()
    refer_pc_idx_per_bbox = list()
    points_category = np.zeros(
        refer_pc.shape[0], dtype=np.int
    )  # store the point categories

    pixel_instance_id = np.zeros(pixel_indices.shape[0], dtype=np.uint8)
    points_instance_id = np.zeros(refer_pc.shape[0], dtype=np.int)

    for i in range(num_instances):
        instance_cat = instance_cat_list[i]
        instance_box = instance_box_list[i]
        instance_box_data = instance_box[0]
        assert not np.isnan(
            instance_box_data
        ).any(), "In the keyframe, there should not be NaN box annotation!"

        if h_flip:
            tmp_quad = instance_box_data[6:].copy()
            tmp_quad[2] *= -1  # y
            tmp_quad[3] *= -1  # z
            tmp_quad = Quaternion(tmp_quad)

            tmp_center = instance_box_data[0:3].copy()
            tmp_center[0] = -tmp_center[0]
            tmp_box = Box(
                center=tmp_center,
                size=instance_box_data[3:6],
                orientation=Quaternion(tmp_quad),
            )
        else:
            tmp_box = Box(
                center=instance_box_data[:3],
                size=instance_box_data[3:6],
                orientation=Quaternion(instance_box_data[6:]),
            )
        idx = point_in_hull_fast(refer_pc[:, 0:3], tmp_box)
        refer_pc_idx_per_bbox.append(idx)
        refer_box_list.append(tmp_box)

        points_category[idx] = instance_cat
        points_instance_id[idx] = i + 1  # object id starts from 1, background has id 0

    assert np.max(points_instance_id) <= 255, "The instance id exceeds uint8 max."

    if len(refer_pc_idx_per_bbox) > 0:
        refer_pc_idx_inside_box = np.concatenate(refer_pc_idx_per_bbox).tolist()
    else:
        refer_pc_idx_inside_box = []
    refer_pc_idx_outside_box = set(range(refer_pc.shape[0])) - set(
        refer_pc_idx_inside_box
    )
    refer_pc_idx_outside_box = list(refer_pc_idx_outside_box)

    # Compute pixel (cell) categories
    pixel_cat = np.zeros([unique_indices.shape[0], category_num], dtype=np.float32)
    most_freq_info = []

    for h, v in enumerate(zip(unique_indices, num_points_in_pixel)):
        pixel_elements_categories = points_category[v[0] : v[0] + v[1]]
        elements_freq = np.bincount(pixel_elements_categories, minlength=category_num)
        assert np.sum(elements_freq) == v[1], "The frequency count is incorrect."

        elements_freq = elements_freq / float(v[1])
        most_freq_cat, most_freq = np.argmax(elements_freq), np.max(elements_freq)
        most_freq_info.append([most_freq_cat, most_freq])

        most_freq_elements_idx = np.where(pixel_elements_categories == most_freq_cat)[0]
        pixel_elements_instance_ids = points_instance_id[v[0] : v[0] + v[1]]
        most_freq_instance_id = pixel_elements_instance_ids[most_freq_elements_idx[0]]

        if most_freq >= one_hot_thresh:
            one_hot_cat = np.zeros(category_num, dtype=np.float32)
            one_hot_cat[most_freq_cat] = 1.0
            pixel_cat[h] = one_hot_cat

            pixel_instance_id[h] = most_freq_instance_id
        else:
            pixel_cat[h] = elements_freq  # use soft category probability vector.

    pixel_cat_map = np.zeros(
        (num_divisions[0], num_divisions[1], category_num), dtype=np.float32
    )
    pixel_cat_map[pixel_indices[:, 0], pixel_indices[:, 1]] = pixel_cat[:]

    pixel_instance_map = np.zeros((num_divisions[0], num_divisions[1]), dtype=np.uint8)
    pixel_instance_map[pixel_indices[:, 0], pixel_indices[:, 1]] = pixel_instance_id[:]

    # Set the non-zero pixels to 1.0, which will be helpful for loss computation
    # Note that the non-zero pixels correspond to both the foreground and background objects
    non_empty_map = np.zeros((num_divisions[0], num_divisions[1]), dtype=np.float32)
    non_empty_map[pixel_indices[:, 0], pixel_indices[:, 1]] = 1.0

    # Ignore the voxel/pillar which contains number of points less than min_point_num_per_voxel; only for fg points
    cell_pts_num = np.zeros((num_divisions[0], num_divisions[1]), dtype=np.float32)
    cell_pts_num[pixel_indices[:, 0], pixel_indices[:, 1]] = num_points_in_pixel[:]
    tmp_pixel_cat_map = np.argmax(pixel_cat_map, axis=2)
    ignore_mask = np.logical_and(
        cell_pts_num <= min_point_num_per_voxel, tmp_pixel_cat_map != 0
    )
    ignore_mask = np.logical_not(ignore_mask)
    ignore_mask = np.expand_dims(ignore_mask, axis=2)

    # Compute the displacement vectors w.r.t. the other sweeps
    all_disp_field_gt_list = list()
    all_valid_pixel_maps_list = (
        list()
    )  # valid pixel map will be used for masking the computation of loss

    # -- Skip some frames if necessary
    past_part = list(range(0, num_past_sweeps, frame_skip + 1))
    future_part = list(range(num_past_sweeps + frame_skip, num_sweeps, frame_skip + 1))
    if return_past_2d_disp_gt:
        zero_disp_field = np.zeros(
            (num_divisions[0], num_divisions[1], 2), dtype=np.float32
        )
        all_disp_field_gt_list.append(
            zero_disp_field
        )  # append once, which corresponds to the current frame
        all_valid_pixel_maps_list.append(non_empty_map)

        frame_considered = np.asarray(past_part + future_part)
        frame_considered = frame_considered[1:]
    else:
        frame_considered = np.asarray(future_part)

    for i in frame_considered:
        curr_disp_vectors = np.zeros_like(refer_pc, dtype=np.float32)
        curr_disp_vectors.fill(np.nan)
        curr_disp_vectors[
            refer_pc_idx_outside_box,
        ] = 0.0

        # First, for each instance, compute the corresponding points displacement.
        for j in range(num_instances):
            instance_box = instance_box_list[j]
            instance_box_data = instance_box[i]  # This is for the i-th sweep

            if np.isnan(
                instance_box_data
            ).any():  # It is possible that in this sweep there is no annotation
                continue

            if h_flip:
                tmp_quad = instance_box_data[6:].copy()
                tmp_quad[2] *= -1  # y
                tmp_quad[3] *= -1  # z
                tmp_quad = Quaternion(tmp_quad)

                tmp_center = instance_box_data[0:3].copy()
                tmp_center[0] = -tmp_center[0]
                tmp_box = Box(
                    center=tmp_center,
                    size=instance_box_data[3:6],
                    orientation=Quaternion(tmp_quad),
                )
            else:
                tmp_box = Box(
                    center=instance_box_data[:3],
                    size=instance_box_data[3:6],
                    orientation=Quaternion(instance_box_data[6:]),
                )
            pc_in_bbox_idx = refer_pc_idx_per_bbox[j]
            disp_vectors = calc_displace_vector(
                refer_pc[pc_in_bbox_idx], refer_box_list[j], tmp_box
            )

            curr_disp_vectors[pc_in_bbox_idx] = disp_vectors[:]

        # Second, compute the mean displacement vector and category for each non-empty pixel
        disp_field = np.zeros(
            [unique_indices.shape[0], 2], dtype=np.float32
        )  # we only consider the 2D field

        # We only compute loss for valid pixels where there are corresponding box annotations between two frames
        valid_pixels = np.zeros(unique_indices.shape[0], dtype=bool)

        for h, v in enumerate(zip(unique_indices, num_points_in_pixel)):

            # Only when the number of majority points exceeds predefined proportion, we compute
            # the displacement vector for this pixel. Otherwise, We consider it is background (possibly ground plane)
            # and has zero displacement.
            pixel_elements_categories = points_category[v[0] : v[0] + v[1]]
            most_freq_cat, most_freq = most_freq_info[h]

            if most_freq >= proportion_thresh:
                most_freq_cat_idx = np.where(
                    pixel_elements_categories == most_freq_cat
                )[0]
                most_freq_cat_disp_vectors = curr_disp_vectors[v[0] : v[0] + v[1], :3]
                most_freq_cat_disp_vectors = most_freq_cat_disp_vectors[
                    most_freq_cat_idx
                ]

                if np.isnan(
                    most_freq_cat_disp_vectors
                ).any():  # contains invalid disp vectors
                    valid_pixels[h] = 0.0
                else:
                    mean_disp_vector = np.mean(most_freq_cat_disp_vectors, axis=0)
                    disp_field[h] = mean_disp_vector[0:2]  # ignore the z direction

                    valid_pixels[h] = 1.0

        # Finally, assemble to a 2D image
        disp_field_sparse = np.zeros(
            (num_divisions[0], num_divisions[1], 2), dtype=np.float32
        )
        disp_field_sparse[pixel_indices[:, 0], pixel_indices[:, 1]] = disp_field[:]
        disp_field_sparse = disp_field_sparse * ignore_mask

        valid_pixel_map = np.zeros(
            (num_divisions[0], num_divisions[1]), dtype=np.float32
        )
        valid_pixel_map[pixel_indices[:, 0], pixel_indices[:, 1]] = valid_pixels[:]

        all_disp_field_gt_list.append(disp_field_sparse)
        all_valid_pixel_maps_list.append(valid_pixel_map)

    all_disp_field_gt_list = np.stack(all_disp_field_gt_list, axis=0)
    all_valid_pixel_maps_list = np.stack(all_valid_pixel_maps_list, axis=0)

    if reordered and return_past_2d_disp_gt:
        num_past = len(past_part)
        all_disp_field_gt_list[0:num_past] = all_disp_field_gt_list[
            (num_past - 1) :: -1
        ]
        all_valid_pixel_maps_list[0:num_past] = all_valid_pixel_maps_list[
            (num_past - 1) :: -1
        ]

    if return_instance_map:
        return (
            all_disp_field_gt_list,
            all_valid_pixel_maps_list,
            non_empty_map,
            pixel_cat_map,
            pixel_indices,
            pixel_instance_map,
            instance_box_list,
            instance_cat_list,
        )
    else:
        return (
            all_disp_field_gt_list,
            all_valid_pixel_maps_list,
            non_empty_map,
            pixel_cat_map,
            pixel_indices,
            instance_box_list,
            instance_cat_list,
        )


def voxelize_occupy(pts, voxel_size, extents=None, return_indices=False):
    """
    Voxelize the input point cloud. We only record if a given voxel is occupied or not, which is just binary indicator.

    The input for the voxelization is expected to be a PointCloud
    with N points in 4 dimension (x,y,z,i). Voxel size is the quantization size for the voxel grid.

    voxel_size: I.e. if voxel size is 1 m, the voxel space will be
    divided up within 1m x 1m x 1m space. This space will be 0 if free/occluded and 1 otherwise.
    min_voxel_coord: coordinates of the minimum on each axis for the voxel grid
    max_voxel_coord: coordinates of the maximum on each axis for the voxel grid
    num_divisions: number of grids in each axis
    leaf_layout: the voxel grid of size (numDivisions) that contain -1 for free, 0 for occupied

    :param pts: Point cloud as N x [x, y, z, i]
    :param voxel_size: Quantization size for the grid, vd, vh, vw
    :param extents: Optional, specifies the full extents of the point cloud.
                    Used for creating same sized voxel grids. Shape (3, 2)
    :param return_indices: Whether to return the non-empty voxel indices.
    """
    # Function Constants
    VOXEL_EMPTY = 0
    VOXEL_FILLED = 1

    # Check if points are 3D, otherwise early exit
    if pts.shape[1] < 3 or pts.shape[1] > 4:
        raise ValueError("Points have the wrong shape: {}".format(pts.shape))

    if extents is not None:
        if extents.shape != (3, 2):
            raise ValueError("Extents are the wrong shape {}".format(extents.shape))

        filter_idx = np.where(
            (extents[0, 0] < pts[:, 0])
            & (pts[:, 0] < extents[0, 1])
            & (extents[1, 0] < pts[:, 1])
            & (pts[:, 1] < extents[1, 1])
            & (extents[2, 0] < pts[:, 2])
            & (pts[:, 2] < extents[2, 1])
        )[0]
        pts = pts[filter_idx]

    # Discretize voxel coordinates to given quantization size
    discrete_pts = np.floor(pts[:, :3] / voxel_size).astype(np.int32)

    # Use Lex Sort, sort by x, then y, then z
    x_col = discrete_pts[:, 0]
    y_col = discrete_pts[:, 1]
    z_col = discrete_pts[:, 2]
    sorted_order = np.lexsort((z_col, y_col, x_col))

    # Save original points in sorted order
    discrete_pts = discrete_pts[sorted_order]

    # Format the array to c-contiguous array for unique function
    contiguous_array = np.ascontiguousarray(discrete_pts).view(
        np.dtype((np.void, discrete_pts.dtype.itemsize * discrete_pts.shape[1]))
    )

    # The new coordinates are the discretized array with its unique indexes
    _, unique_indices = np.unique(contiguous_array, return_index=True)

    # Sort unique indices to preserve order
    unique_indices.sort()

    voxel_coords = discrete_pts[unique_indices]

    # Compute the minimum and maximum voxel coordinates
    if extents is not None:
        min_voxel_coord = np.floor(extents.T[0] / voxel_size)
        max_voxel_coord = np.ceil(extents.T[1] / voxel_size) - 1
    else:
        min_voxel_coord = np.amin(voxel_coords, axis=0)
        max_voxel_coord = np.amax(voxel_coords, axis=0)

    # Get the voxel grid dimensions
    num_divisions = ((max_voxel_coord - min_voxel_coord) + 1).astype(np.int32)

    # Bring the min voxel to the origin
    voxel_indices = (voxel_coords - min_voxel_coord).astype(int)

    # Create Voxel Object with -1 as empty/occluded
    leaf_layout = VOXEL_EMPTY * np.ones(num_divisions.astype(int), dtype=np.float32)

    # Fill out the leaf layout
    leaf_layout[
        voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]
    ] = VOXEL_FILLED

    if return_indices:
        return leaf_layout, voxel_indices
    else:
        return leaf_layout


def voxelize_pillar_indices(pts, voxel_size, extents=None):
    """
    Voxelize the input point cloud into pillars. We only return the indices

    The input for the voxelization is expected to be a PointCloud
    with N points in 4 dimension (x,y,z,i). Voxel size is the quantization size for the voxel grid.

    voxel_size: I.e. if voxel size is 1 m, the voxel space will be
    divided up within 1m x 1m x 1m space. This space will be 0 if free/occluded and 1 otherwise.
    min_voxel_coord: coordinates of the minimum on each axis for the voxel grid
    max_voxel_coord: coordinates of the maximum on each axis for the voxel grid
    num_divisions: number of grids in each axis
    leaf_layout: the voxel grid of size (numDivisions) that contain -1 for free, 0 for occupied

    :param pts: Point cloud as N x [x, y, z, i]
    :param voxel_size: Quantization size for the grid, vh, vw
    :param extents: Optional, specifies the full extents of the point cloud.
                    Used for creating same sized voxel grids. Shape (3, 2)
    """
    # Check if points are 3D, otherwise early exit
    if pts.shape[1] < 3 or pts.shape[1] > 4:
        raise ValueError("Points have the wrong shape: {}".format(pts.shape))

    if extents is not None:
        if extents.shape != (3, 2):
            raise ValueError("Extents are the wrong shape {}".format(extents.shape))

        filter_idx = np.where(
            (extents[0, 0] < pts[:, 0])
            & (pts[:, 0] < extents[0, 1])
            & (extents[1, 0] < pts[:, 1])
            & (pts[:, 1] < extents[1, 1])
            & (extents[2, 0] < pts[:, 2])
            & (pts[:, 2] < extents[2, 1])
        )[0]
        pts = pts[filter_idx]

    # Discretize voxel coordinates to given quantization size
    discrete_pts = np.floor(pts[:, :2] / voxel_size).astype(np.int32)

    # Use Lex Sort, sort by x, then y
    x_col = discrete_pts[:, 0]
    y_col = discrete_pts[:, 1]
    sorted_order = np.lexsort((y_col, x_col))

    # Save original points in sorted order
    points = pts[sorted_order]
    discrete_pts = discrete_pts[sorted_order]

    # Format the array to c-contiguous array for unique function
    contiguous_array = np.ascontiguousarray(discrete_pts).view(
        np.dtype((np.void, discrete_pts.dtype.itemsize * discrete_pts.shape[1]))
    )

    # The new coordinates are the discretized array with its unique indexes
    _, unique_indices = np.unique(contiguous_array, return_index=True)

    # Sort unique indices to preserve order
    unique_indices.sort()

    voxel_coords = discrete_pts[unique_indices]

    # Number of points per voxel, last voxel calculated separately
    num_points_in_pillar = np.diff(unique_indices)
    num_points_in_pillar = np.append(
        num_points_in_pillar, discrete_pts.shape[0] - unique_indices[-1]
    )

    # Compute the minimum and maximum voxel coordinates
    if extents is not None:
        min_voxel_coord = np.floor(extents.T[0, 0:2] / voxel_size)
    else:
        min_voxel_coord = np.amin(voxel_coords, axis=0)

    # Bring the min voxel to the origin
    voxel_indices = (voxel_coords - min_voxel_coord).astype(int)

    return points, voxel_indices, num_points_in_pillar


def voxelize_point_pillar(
    pts,
    grid_size,
    extents=None,
    num_points=100,
    num_pillars=2500,
    seed=None,
    is_padded_pillar=False,
):
    """
    Discretize the input point cloud into pillars.

    The input for the voxelization is expected to be a PointCloud
    with N points in 4 dimension (x,y,z,i). Grid size is the quantization size for the 2d grid.

    grid_size: I.e. if grid size is 1 m, the grid space will be
    divided up within 1m x 1m space. This space will be -1 if free/occluded and 1 otherwise.
    min_grid_coord: coordinates of the minimum on each axis for the 2d grid
    max_grid_coord: coordinates of the maximum on each axis for the 2d grid
    num_divisions: number of grids in each axis
    leaf_layout: the 2d grid of size (numDivisions) that contain -1 for free, 0 for occupied

    :param pts: Point cloud as N x [x, y, z, i]
    :param grid_size: Quantization size for the grid, (vh, vw)
    :param extents: Optional, specifies the full extents of the point cloud.
                    Used for creating same sized voxel grids. Shape (3, 2)
    :param num_points: Number of points in each pillar after sampling/padding
    :param num_pillars: Number of pillars after sampling/padding
    :param seed: Random seed for fixing data generation.
    :param is_padded_pillar: Whether need to pad/sample the pillar
    """
    if seed is not None:
        np.random.seed(seed)

    # Check if points are 3D, otherwise early exit
    if pts.shape[1] < 3 or pts.shape[1] > 4:
        raise ValueError("Points have the wrong shape: {}".format(pts.shape))

    if extents is not None:
        if extents.shape != (3, 2):
            raise ValueError("Extents are the wrong shape {}".format(extents.shape))

        filter_idx = np.where(
            (extents[0, 0] < pts[:, 0])
            & (pts[:, 0] < extents[0, 1])
            & (extents[1, 0] < pts[:, 1])
            & (pts[:, 1] < extents[1, 1])
            & (extents[2, 0] < pts[:, 2])
            & (pts[:, 2] < extents[2, 1])
        )[0]
        pts = pts[filter_idx]

    # Discretize point coordinates to given 2d quantization size
    discrete_pts = np.floor(pts[:, :2] / grid_size).astype(np.int32)

    # Use Lex Sort, sort by x, then y. We do not care about z
    x_col = discrete_pts[:, 0]
    y_col = discrete_pts[:, 1]
    sorted_order = np.lexsort((y_col, x_col))

    # Save original points in sorted order
    points = pts[sorted_order]
    discrete_pts = discrete_pts[sorted_order]

    # Format the array to c-contiguous array for unique function
    contiguous_array = np.ascontiguousarray(discrete_pts).view(
        np.dtype((np.void, discrete_pts.dtype.itemsize * discrete_pts.shape[1]))
    )

    # The new coordinates are the discretized array with its unique indexes
    _, unique_indices = np.unique(contiguous_array, return_index=True)

    # Sort unique indices to preserve order
    unique_indices.sort()

    grid_coords = discrete_pts[unique_indices]

    # Number of points per voxel, last voxel calculated separately
    num_points_in_pillar = np.diff(unique_indices)
    num_points_in_pillar = np.append(
        num_points_in_pillar, discrete_pts.shape[0] - unique_indices[-1]
    )

    # Compute the minimum and maximum voxel coordinates
    if extents is not None:
        min_grid_coord = np.floor(extents.T[0, 0:2] / grid_size)
        max_grid_coord = np.ceil(extents.T[1, 0:2] / grid_size) - 1
    else:
        min_grid_coord = np.amin(grid_coords, axis=0)
        max_grid_coord = np.amax(grid_coords, axis=0)

    # Get the voxel grid dimensions
    num_divisions = ((max_grid_coord - min_grid_coord) + 1).astype(np.int32)

    # Bring the min voxel to the origin
    pixel_indices = (grid_coords - min_grid_coord).astype(int)

    # Padding the points within each voxel
    x_offset = grid_size[0] / 2.0 + extents[0, 0]
    y_offset = grid_size[1] / 2.0 + extents[1, 0]

    padded_grid_points = np.zeros(
        [unique_indices.shape[0], num_points, pts.shape[1] + 3 + 2], dtype=np.float32
    )
    padded_pillar = np.zeros(
        [num_pillars, num_points, pts.shape[1] + 3 + 2], dtype=np.float32
    )
    padded_pixel_indices = np.zeros(
        [num_pillars, pixel_indices.shape[1]], dtype=np.int64
    )

    padded_grid_points = padding_point_pillar(
        padded_grid_points,
        unique_indices,
        num_points,
        num_points_in_pillar,
        points,
        pixel_indices,
        grid_size,
        x_offset,
        y_offset,
        seed,
    )

    if is_padded_pillar:

        # Padding or sampling the pillars  TODO: early sampling to avoid unnecessary computation
        if unique_indices.shape[0] < num_pillars:
            padded_pillar[: unique_indices.shape[0], :, :] = padded_grid_points[:]
            padded_pixel_indices[: unique_indices.shape[0], :] = pixel_indices[:]
        else:
            pillar_inds = np.random.choice(unique_indices.shape[0], num_pillars)
            padded_pillar[:, :, :] = padded_grid_points[pillar_inds, :, :]
            padded_pixel_indices[:, :] = pixel_indices[pillar_inds, :]
    else:
        padded_pillar = padded_grid_points
        padded_pixel_indices = pixel_indices

    return padded_pillar, padded_pixel_indices, num_divisions


@njit
def padding_point_pillar(
    padded_grid_points,
    unique_indices,
    num_points,
    num_points_in_pillar,
    points,
    pixel_indices,
    grid_size,
    x_offset,
    y_offset,
    seed,
):
    if seed is not None:
        np.random.seed(seed)
    for i, v in enumerate(zip(unique_indices, num_points_in_pillar)):
        if v[1] < num_points:
            padded_grid_points[i, : v[1], :4] = points[v[0] : v[0] + v[1], :]
            middle_points_x = np.mean(points[v[0] : v[0] + v[1], 0])
            middle_points_y = np.mean(points[v[0] : v[0] + v[1], 1])
            middle_points_z = np.mean(points[v[0] : v[0] + v[1], 2])

            padded_grid_points[i, : v[1], 4] = (
                padded_grid_points[i, : v[1], 0] - middle_points_x
            )
            padded_grid_points[i, : v[1], 5] = (
                padded_grid_points[i, : v[1], 1] - middle_points_y
            )
            padded_grid_points[i, : v[1], 6] = (
                padded_grid_points[i, : v[1], 2] - middle_points_z
            )

            center_offsets = np.zeros((v[1], 2), dtype=np.float32)
            center_offsets[:, 0] = padded_grid_points[i, : v[1], 0] - (
                pixel_indices[i, 0] * grid_size[0] + x_offset
            )
            center_offsets[:, 1] = padded_grid_points[i, : v[1], 1] - (
                pixel_indices[i, 1] * grid_size[1] + y_offset
            )
            padded_grid_points[i, : v[1], 7:] = center_offsets[:]
        else:
            inds = np.random.choice(v[1], num_points)
            padded_grid_points[i, :, :4] = points[v[0] + inds, :]
            middle_points_x = np.mean(points[v[0] + inds, 0])
            middle_points_y = np.mean(points[v[0] + inds, 1])
            middle_points_z = np.mean(points[v[0] + inds, 2])

            padded_grid_points[i, :, 4] = padded_grid_points[i, :, 0] - middle_points_x
            padded_grid_points[i, :, 5] = padded_grid_points[i, :, 1] - middle_points_y
            padded_grid_points[i, :, 6] = padded_grid_points[i, :, 2] - middle_points_z

            center_offsets = np.zeros((num_points, 2), dtype=np.float32)
            center_offsets[:, 0] = padded_grid_points[i, :, 0] - (
                pixel_indices[i, 0] * grid_size[0] + x_offset
            )
            center_offsets[:, 1] = padded_grid_points[i, :, 1] - (
                pixel_indices[i, 1] * grid_size[1] + y_offset
            )
            padded_grid_points[i, :, 7:] = center_offsets[:]

    return padded_grid_points


def compute_ratio_cat_and_motion(
    dataset_root=None,
    frame_skip=3,
    voxel_size=(0.4, 0.4, 0.4),
    split="train",
    area_extents=np.array([[-30.0, 30.0], [-30.0, 30.0], [-2.0, 2.0]]),
    num_obj_cat=5,
    num_motion_cat=3,
):
    """
    Compute the ratios between foreground and background (and static and moving) cells. The ratios will be used for
    non-uniform weighting to mitigate the class imbalance during training.
    :param dataset_root: The path to the dataset
    :param frame_skip: The number of frame skipped in a sample sequence
    :param voxel_size: Voxel size, which determines the "image" resolution
    :param split: The data split
    :param area_extents: The area of interest for point cloud
    :param num_obj_cat: The number of object categories.
    :param num_motion_cat: The number of motion categories. Currently it is 2 (ie, static and moving).
    """
    if dataset_root is None:
        dataset_root = "/homes/pwu/_drives/cv0/data/homes/pwu/preprocessed_pc"

    scene_dirs = [
        d
        for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
    ]

    if split == "train":
        scene_dirs = scene_dirs[: len(scene_dirs) // 2]
    else:
        scene_dirs = scene_dirs[len(scene_dirs) // 2 :]

    sample_seq_files = []
    for s_dir in scene_dirs:
        sample_dir = os.path.join(dataset_root, s_dir)
        sample_files = [
            os.path.join(sample_dir, f)
            for f in os.listdir(sample_dir)
            if os.path.isfile(os.path.join(sample_dir, f))
        ]

        sample_seq_files += sample_files

    num_sample_seqs = len(sample_seq_files)

    obj_cat_cnt = np.zeros(num_obj_cat, dtype=np.int64)
    motion_cat_cnt = np.zeros(num_motion_cat, dtype=np.int64)

    for idx in range(num_sample_seqs):
        sample_file = sample_seq_files[idx]

        (
            all_disp_field_gt,
            all_valid_pixel_maps,
            non_empty_map,
            pixel_cat_map_gt,
        ) = gen_2d_grid_gt(
            sample_file,
            grid_size=voxel_size[0:2],
            reordered=True,
            extents=area_extents,
            frame_skip=frame_skip,
        )

        # -- Compute speed level ground truth
        motion_status_gt = compute_speed_level(all_disp_field_gt, frame_skip=frame_skip)

        # -- Count the object category number
        max_prob = np.amax(pixel_cat_map_gt, axis=-1)
        filter_mask = max_prob == 1.0
        pixel_cat_map = (
            np.argmax(pixel_cat_map_gt, axis=-1) + 1
        )  # category starts from 1 (background), etc
        pixel_cat_map = (pixel_cat_map * non_empty_map * filter_mask).astype(np.int)

        for i in range(num_obj_cat):
            curr_cat_mask = pixel_cat_map == (i + 1)
            curr_cat_num = np.sum(curr_cat_mask)
            obj_cat_cnt[i] += curr_cat_num

        # -- Count the motion category number
        motion_cat_map = (
            np.argmax(motion_status_gt, axis=-1) + 1
        )  # category starts from 1 (static), etc
        motion_cat_map = (motion_cat_map * non_empty_map * filter_mask).astype(np.int)

        for i in range(num_motion_cat):
            curr_motion_mask = motion_cat_map == (i + 1)
            curr_motion_num = np.sum(curr_motion_mask)
            motion_cat_cnt[i] += curr_motion_num

        print("Finish {}".format(idx))

    print("The category numbers: \n{}".format(obj_cat_cnt))
    print("The motion numbers: \n{}\n".format(motion_cat_cnt))

    # Convert to ratio
    obj_cat_ratio = obj_cat_cnt / np.sum(obj_cat_cnt)
    motion_cat_ratio = motion_cat_cnt / np.sum(motion_cat_cnt)

    print("The category ratios: \n{}".format(obj_cat_ratio))
    print("The motion ratios: \n{}\n".format(motion_cat_ratio))

    # Convert to inverse ratio
    obj_cat_ratio_inverse = np.exp(-obj_cat_ratio)
    motion_cat_ratio_inverse = np.exp(-motion_cat_ratio)

    print("The category reverse ratios: \n{}".format(obj_cat_ratio_inverse))
    print("The motion reverse ratios: \n{}\n".format(motion_cat_ratio_inverse))


def compute_speed_level(all_disp_field_gt, total_future_sweeps=20, frame_skip=3):
    speed_intervals = np.array([[0, 5.0], [5.0, 20.0], [20.0, np.inf]])  # unit: m/s

    selected_future_sweeps = np.arange(0, total_future_sweeps + 1, frame_skip + 1)
    selected_future_sweeps = selected_future_sweeps[1:]
    last_future_sweep_id = selected_future_sweeps[-1]
    distance_intervals = speed_intervals * (last_future_sweep_id / 20.0)

    speed_level = np.zeros(
        (
            all_disp_field_gt.shape[1],
            all_disp_field_gt.shape[2],
            speed_intervals.shape[0],
        ),
        dtype=np.float32,
    )
    last_frame_disp_norm = np.linalg.norm(all_disp_field_gt, ord=2, axis=-1)
    last_frame_disp_norm = last_frame_disp_norm[-1, :, :]

    for s, d in enumerate(distance_intervals):
        mask = np.logical_and(d[0] <= last_frame_disp_norm, last_frame_disp_norm < d[1])

        one_hot_vector = np.zeros(speed_intervals.shape[0], dtype=np.float32)
        one_hot_vector[s] = 1.0

        speed_level[mask] = one_hot_vector[:]

    return speed_level


def compute_speed_level_with_static(
    all_disp_field_gt, total_future_sweeps=20, frame_skip=3
):
    speed_intervals = np.array(
        [[0.0, 0.0], [0, 5.0], [5.0, 20.0], [20.0, np.inf]]
    )  # unit: m/s

    # First, compute the static and moving cell masks
    all_disp_field_gt_norm = np.linalg.norm(all_disp_field_gt, ord=2, axis=-1)

    upper_thresh = 0.2
    upper_bound = (frame_skip + 1) / 20 * upper_thresh
    selected_future_sweeps = np.arange(0, total_future_sweeps + 1, frame_skip + 1)
    selected_future_sweeps = selected_future_sweeps[1:]

    future_sweeps_disp_field_gt_norm = all_disp_field_gt_norm[
        -len(selected_future_sweeps) :, ...
    ]
    static_cell_mask = future_sweeps_disp_field_gt_norm <= upper_bound
    static_cell_mask = np.all(static_cell_mask, axis=0)  # along the sequence axis
    moving_cell_mask = np.logical_not(static_cell_mask)

    # Next, compute the speed level mask
    last_future_sweep_id = selected_future_sweeps[-1]
    distance_intervals = speed_intervals * (last_future_sweep_id / 20.0)

    speed_level = np.zeros(
        (
            all_disp_field_gt.shape[1],
            all_disp_field_gt.shape[2],
            speed_intervals.shape[0],
        ),
        dtype=np.float32,
    )
    last_frame_disp_norm = all_disp_field_gt_norm[-1, :, :]

    for s, d in enumerate(distance_intervals):
        if s == 0:
            mask = static_cell_mask
        else:
            mask = np.logical_and(
                d[0] <= last_frame_disp_norm, last_frame_disp_norm < d[1]
            )
            mask = np.logical_and(mask, moving_cell_mask)

        one_hot_vector = np.zeros(speed_intervals.shape[0], dtype=np.float32)
        one_hot_vector[s] = 1.0

        speed_level[mask] = one_hot_vector[:]

    return speed_level


def classify_speed_level(
    all_disp_field_gt, total_future_sweeps=20, future_frame_skip=0
):
    """
    Classify each cell into static (possibly background) or moving.
    """
    # First, compute the static and moving cell masks
    all_disp_field_gt_norm = np.linalg.norm(all_disp_field_gt, ord=2, axis=-1)

    # Every future_frame_skip frames, if the movement of grid cells does not exceed this thresh (unit: meters),
    # then they are considered as static. This thresh is set to be the maximum perturbation for 1 second.
    upper_thresh = 0.2
    upper_bound = (future_frame_skip + 1) / 20 * upper_thresh
    selected_future_sweeps = np.arange(
        0, total_future_sweeps + 1, future_frame_skip + 1
    )
    selected_future_sweeps = selected_future_sweeps[1:]

    future_sweeps_disp_field_gt_norm = all_disp_field_gt_norm[
        -len(selected_future_sweeps) :, ...
    ]
    static_cell_mask = future_sweeps_disp_field_gt_norm <= upper_bound
    static_cell_mask = np.all(static_cell_mask, axis=0)  # along the temporal axis
    moving_cell_mask = np.logical_not(static_cell_mask)

    # Next, compute corresponding one-hot vectors
    motion_cat = np.zeros(
        (all_disp_field_gt.shape[1], all_disp_field_gt.shape[2], 2), dtype=np.float32
    )
    bg_one_hot_vector = np.zeros(2, dtype=np.float32)
    bg_one_hot_vector[0] = 1.0
    motion_cat[static_cell_mask] = bg_one_hot_vector[:]

    fg_one_hot_vector = np.zeros(2, dtype=np.float32)
    fg_one_hot_vector[1] = 1.0
    motion_cat[moving_cell_mask] = fg_one_hot_vector[:]

    return motion_cat


def apply_pose_noise(pose_noise: float, trans_matrices):
    """Apply pose noise to transformation matrices

    Draw noise from normal distribution with mean of pose noise and std of 0.02,
    apply to displacements of the transformation matrices.

    Args:
        pose_noise (float): the mean of pose noise drawn from normal distribution
        trans_matrices (tensor): the transformation matrices
    """
    noise_mat = np.random.normal(pose_noise, 0.02, size=trans_matrices.shape)
    trans_matrices[:, :, :, :3, 3] += noise_mat[:, :, :, :3, 3]


if __name__ == "__main__":
    compute_ratio_cat_and_motion()
