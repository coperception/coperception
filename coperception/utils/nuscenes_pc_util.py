from nuscenes.utils.data_classes import PointCloud
import numpy as np
from functools import reduce
from typing import Dict
from nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from pyquaternion import Quaternion
import os.path as osp


NUM_CROSS_ROAD_SENSOR = 15
NUM_TYPES_OF_SENSORS = 23


def from_file_multisweep_upperbound_sample_data(
    nusc: "NuScenes",
    ref_sd_rec: Dict,
    return_trans_matrix: bool = False,
    min_distance: float = 1.0,
    no_cross_road=False,
):
    """
    Added by Yiming. 2021.4.14 teacher's input
    Upperbound dataloader: transform the sweeps into the local coordinate of agent 0,
    :param ref_sd_rec: The current sample data record (lidar_top_id_0)
    :param return_trans_matrix: Whether need to return the transformation matrix
    :param min_distance: Distance below which points are discarded.
    :return: (all_pc, all_times). The aggregated point cloud and timestamps.
    """

    # Init
    points = np.zeros((PointCloud.nbr_dims(), 0))
    all_pc = PointCloud(points)
    all_times = np.zeros((1, 0))

    # Get reference pose and timestamp
    ref_pose_rec = nusc.get("ego_pose", ref_sd_rec["ego_pose_token"])
    ref_cs_rec = nusc.get("calibrated_sensor", ref_sd_rec["calibrated_sensor_token"])
    ref_time = 1e-6 * ref_sd_rec["timestamp"]

    # Homogeneous transform from ego car frame to reference frame
    ref_from_car = transform_matrix(
        ref_cs_rec["translation"], Quaternion(ref_cs_rec["rotation"]), inverse=True
    )

    # Homogeneous transformation matrix from global to _current_ ego car frame
    car_from_global = transform_matrix(
        ref_pose_rec["translation"], Quaternion(ref_pose_rec["rotation"]), inverse=True
    )

    # Aggregate current and previous sweeps.
    current_sd_rec = ref_sd_rec
    trans_matrix_list = list()
    skip_frame = 0
    sample_record = nusc.get("sample", ref_sd_rec["sample_token"])

    num_sensor = (
        (len(sample_record["data"]) - NUM_CROSS_ROAD_SENSOR) // NUM_TYPES_OF_SENSORS
    ) + 1

    for k in range(num_sensor):
        if no_cross_road and k == 0:
            continue

        # Load up the pointcloud.
        pointsensor_token = sample_record["data"]["LIDAR_TOP_id_" + str(k)]
        current_sd_rec = nusc.get("sample_data", pointsensor_token)
        current_pc = PointCloud.from_file(
            osp.join(nusc.dataroot, current_sd_rec["filename"])
        )

        # Get past pose.
        current_pose_rec = nusc.get("ego_pose", current_sd_rec["ego_pose_token"])
        global_from_car = transform_matrix(
            current_pose_rec["translation"],
            Quaternion(current_pose_rec["rotation"]),
            inverse=False,
        )

        # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
        current_cs_rec = nusc.get(
            "calibrated_sensor", current_sd_rec["calibrated_sensor_token"]
        )
        car_from_current = transform_matrix(
            current_cs_rec["translation"],
            Quaternion(current_cs_rec["rotation"]),
            inverse=False,
        )

        # Fuse four transformation matrices into one and perform transform.
        trans_matrix = reduce(
            np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current]
        )

        current_pc.transform(trans_matrix)

        # Collect the transformation matrix
        trans_matrix_list.append(trans_matrix)

        # Remove close points and add timevector.
        current_pc.remove_close(min_distance)

        time_lag = ref_time - 1e-6 * (
            current_sd_rec["timestamp"] + k
        )  # positive difference

        if k % (skip_frame + 1) == 0:
            times = time_lag * np.ones((1, current_pc.nbr_points()))
        else:
            times = time_lag * np.ones((1, 1))  # dummy value

        all_times = np.hstack((all_times, times))
        all_pc.points = np.hstack((all_pc.points, current_pc.points))

    trans_matrix_list = np.stack(trans_matrix_list, axis=0)

    if return_trans_matrix:
        return all_pc, np.squeeze(all_times, 0), trans_matrix_list
    else:
        return all_pc, np.squeeze(all_times, 0)


def from_file_multisweep_warp2com_sample_data(
    current_agent,
    nusc: "NuScenes",
    ref_sd_rec: Dict,
    return_trans_matrix: bool = False,
    min_distance: float = 1.0,
):
    """
    Added by Yiming. 2021/3/27
    V2V dataloader: calculate relative pose and overlap mask between different agents
    :param ref_sd_rec: The current sample data record
    :param return_trans_matrix: Whether need to return the transformation matrix
    :param min_distance: Distance below which points are discarded.
    :return: (all_pc, all_times). The aggregated point cloud and timestamps.
    """

    # Init
    all_times = np.zeros((1, 0))

    # Get reference pose and timestamp
    ref_pose_rec = nusc.get("ego_pose", ref_sd_rec["ego_pose_token"])
    ref_cs_rec = nusc.get("calibrated_sensor", ref_sd_rec["calibrated_sensor_token"])
    ref_time = 1e-6 * ref_sd_rec["timestamp"]

    # Homogeneous transform from ego car frame to reference frame
    ref_from_car = transform_matrix(
        ref_cs_rec["translation"], Quaternion(ref_cs_rec["rotation"]), inverse=True
    )

    # Homogeneous transformation matrix from global to _current_ ego car frame
    car_from_global = transform_matrix(
        ref_pose_rec["translation"], Quaternion(ref_pose_rec["rotation"]), inverse=True
    )

    # Aggregate current and previous sweeps.
    current_sd_rec = ref_sd_rec
    trans_matrix_list = list()
    trans_matrix_list_no_cross_road = list()

    sample_record = nusc.get("sample", ref_sd_rec["sample_token"])
    # num_sensor = int(len(sample_record['data'])/2)
    # num_sensor = int((len(sample_record['data']) - 3) / 3)  # 6 camera 6 depth 6 seg 6 seg-raw 2 bev 2 lidar 1 gnss 1 imu

    num_sensor = (
        (len(sample_record["data"]) - NUM_CROSS_ROAD_SENSOR) // NUM_TYPES_OF_SENSORS
    ) + 1
    x_global = [[] for i in range(num_sensor)]
    y_global = [[] for i in range(num_sensor)]
    x_local = [[] for i in range(num_sensor)]
    y_local = [[] for i in range(num_sensor)]

    # the following two lists will store the data for each agent
    target_agent_id = None  # which agent is the center agent

    for k in range(num_sensor):
        # Load up the pointcloud.
        pointsensor_token = sample_record["data"]["LIDAR_TOP_id_" + str(k)]
        current_sd_rec = nusc.get("sample_data", pointsensor_token)

        # Get past pose.
        current_pose_rec = nusc.get("ego_pose", current_sd_rec["ego_pose_token"])

        y_global[k] = current_pose_rec["translation"][1]
        x_global[k] = current_pose_rec["translation"][0]

    for k in range(num_sensor):
        x_local[k] = x_global[k] - x_global[current_agent]
        y_local[k] = y_global[k] - y_global[current_agent]

    for k in range(num_sensor):

        # Load up the pointcloud.
        pointsensor_token = sample_record["data"]["LIDAR_TOP_id_" + str(k)]
        current_sd_rec = nusc.get("sample_data", pointsensor_token)

        if x_local[k] == 0.0 and y_local[k] == 0.0:
            target_agent_id = k
            current_pc = PointCloud.from_file(
                osp.join(nusc.dataroot, current_sd_rec["filename"])
            )
            # Remove close points and add timevector.
            current_pc.remove_close(min_distance)
            time_lag = ref_time - 1e-6 * (
                current_sd_rec["timestamp"]
            )  # positive difference
            times = time_lag * np.ones((1, current_pc.nbr_points()))
            all_times = np.hstack((all_times, times))

        # Get past pose.
        current_pose_rec = nusc.get("ego_pose", current_sd_rec["ego_pose_token"])
        global_from_car = transform_matrix(
            np.sum([current_pose_rec["translation"]], axis=0),
            Quaternion(current_pose_rec["rotation"]),
            inverse=False,
        )

        # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
        current_cs_rec = nusc.get(
            "calibrated_sensor", current_sd_rec["calibrated_sensor_token"]
        )
        car_from_current = transform_matrix(
            current_cs_rec["translation"],
            Quaternion(current_cs_rec["rotation"]),
            inverse=False,
        )

        # Fuse four transformation matrices into one and perform transform.
        trans_matrix = reduce(
            np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current]
        )

        # Collect the transformation matrix
        trans_matrix_list.append(trans_matrix)

        # Collect the transformation matrix for no-cross-road data
        if k != 0:
            trans_matrix_list_no_cross_road.append(trans_matrix)

    # num of agent at most
    max_num_agent = 6
    for k in range(max_num_agent - num_sensor):
        trans_matrix_list.append(np.zeros((4, 4)))
        trans_matrix_list_no_cross_road.append(np.zeros((4, 4)))

    trans_matrix_list = np.stack(trans_matrix_list, axis=0)
    trans_matrix_list_no_cross_road = np.stack(trans_matrix_list_no_cross_road, axis=0)

    if return_trans_matrix:
        return (
            current_pc,
            np.squeeze(all_times, 0),
            trans_matrix_list,
            trans_matrix_list_no_cross_road,
            target_agent_id,
            num_sensor,
        )
    else:
        return current_pc, np.squeeze(all_times, 0), target_agent_id, num_sensor
