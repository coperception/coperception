from nuscenes.utils.data_classes import LidarPointCloud, Box
import numpy as np
from functools import reduce
from typing import Tuple, List, Dict
from nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
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
    points = np.zeros((LidarPointCloud.nbr_dims(), 0))
    all_pc = LidarPointCloud(points)
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
        current_pc = LidarPointCloud.from_file(
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
            current_pc = LidarPointCloud.from_file(
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

def get_ann_of_instance(nusc: "NuScenes", sample_rec: Dict, instance_token: str) -> str:
        """
        Return the annotations within the sample which match the given instance.
        :param sample_rec: The given sample record.
        :param instance_token: The instance which need to be matched.
        :return: The annotation which matches the instance.
        """
        sd_anns = sample_rec['anns']
        instance_ann_token = None
        cnt = 0

        for ann_token in sd_anns:
            tmp_ann_rec = nusc.get('sample_annotation', ann_token)
            tmp_instance_token = tmp_ann_rec['instance_token']

            if instance_token == tmp_instance_token:
                instance_ann_token = ann_token
                cnt += 1

        assert cnt <= 1, 'One instance cannot associate more than 1 annotations.'

        if cnt == 1:
            return instance_ann_token
        else:
            return ""

def get_instance_box(nusc: "NuScenes", sample_data_token: str, instance_token: str):
        """
        Get the bounding box associated with the given instance in the sample data.
        :param sample_data_token: The sample data identifier at a certain time stamp.
        :param instance_token: The queried instance.
        :return: The bounding box associated with the instance.
        """
        # Retrieve sensor & pose records
        sd_record = nusc.get('sample_data', sample_data_token)
        curr_sample_record = nusc.get('sample', sd_record['sample_token'])

        instance_ann_token = get_ann_of_instance(nusc, curr_sample_record, instance_token)
        if instance_ann_token == "":
            return None, None, None

        sample_ann_rec = nusc.get('sample_annotation', instance_ann_token)

        # Get the attribute of this annotation
        if len(sample_ann_rec['attribute_tokens']) != 0:
            attr = nusc.get('attribute', sample_ann_rec['attribute_tokens'][0])['name']
        else:
            attr = None

        # Get the category of this annotation
        cat = sample_ann_rec['category_name']

        if curr_sample_record['prev'] == "" or sd_record['is_key_frame']:
            # If no previous annotations available, or if sample_data is keyframe just return the current ones.
            box = nusc.get_box(instance_ann_token)

        else:
            prev_sample_record = nusc.get('sample', curr_sample_record['prev'])

            curr_ann_rec = nusc.get('sample_annotation', instance_ann_token)
            prev_ann_recs = [nusc.get('sample_annotation', token) for token in prev_sample_record['anns']]

            # Maps instance tokens to prev_ann records
            prev_inst_map = {entry['instance_token']: entry for entry in prev_ann_recs}

            t0 = prev_sample_record['timestamp']
            t1 = curr_sample_record['timestamp']
            t = sd_record['timestamp']

            # There are rare situations where the timestamps in the DB are off so ensure that t0 < t < t1.
            t = max(t0, min(t1, t))

            if instance_token in prev_inst_map:
                # If the annotated instance existed in the previous frame, interpolate center & orientation.
                prev_ann_rec = prev_inst_map[instance_token]

                # Interpolate center.
                center = [np.interp(t, [t0, t1], [c0, c1]) for c0, c1 in zip(prev_ann_rec['translation'],
                                                                             curr_ann_rec['translation'])]

                # Interpolate orientation.
                rotation = Quaternion.slerp(q0=Quaternion(prev_ann_rec['rotation']),
                                            q1=Quaternion(curr_ann_rec['rotation']),
                                            amount=(t - t0) / (t1 - t0))

                box = Box(center, curr_ann_rec['size'], rotation, name=curr_ann_rec['category_name'],
                          token=curr_ann_rec['token'])
            else:
                # If not, simply grab the current annotation.
                box = nusc.get_box(curr_ann_rec['token'])

        return box, attr, cat

def get_instance_boxes_multisweep_sample_data(nusc: 'NuScenes',
                                                ref_sd_rec: Dict,
                                                instance_token: str,
                                                nsweeps_back: int = 5,
                                                nsweeps_forward: int = 5) -> \
        Tuple[List['Box'], np.array, List[str], List[str]]:
    """
    Return the bounding boxes associated with the given instance. The bounding boxes are across different sweeps.
    For each bounding box, we need to map its (global) coordinates to the reference frame.
    For this function, the reference sweep is supposed to be from sample data record (not sample. ie, keyframe).
    :param nusc: A NuScenes instance.
    :param ref_sd_rec: The current sample data record.
    :param instance_token: The current selected instance.
    :param nsweeps_back: Number of sweeps to aggregate. The sweeps trace back.
    :param nsweeps_forward: Number of sweeps to aggregate. The sweeps are obtained from the future.
    :return: (list of bounding boxes, the time stamps of bounding boxes, attribute list, category list)
    """

    # Init
    box_list = list()
    all_times = list()
    attr_list = list()  # attribute list
    cat_list = list()  # category list

    # Get reference pose and timestamp
    ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
    ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
    ref_time = 1e-6 * ref_sd_rec['timestamp']

    # Get the bounding boxes across different sweeps
    boxes = list()

    # Move backward to get the past annotations
    current_sd_rec = ref_sd_rec
    for _ in range(nsweeps_back):
        box, attr, cat = get_instance_box(nusc, current_sd_rec['token'], instance_token)
        boxes.append(box)  # It is possible the returned box is None
        attr_list.append(attr)
        cat_list.append(cat)

        time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']  # positive difference
        all_times.append(time_lag)

        if current_sd_rec['prev'] == '':
            break
        else:
            current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])

    # Move forward to get the future annotations
    current_sd_rec = ref_sd_rec

    # Abort if there are no future sweeps.
    if current_sd_rec['next'] != '':
        current_sd_rec = nusc.get('sample_data', current_sd_rec['next'])

        for _ in range(nsweeps_forward):
            box, attr, cat = get_instance_box(nusc, current_sd_rec['token'], instance_token)
            boxes.append(box)  # It is possible the returned box is None
            attr_list.append(attr)
            cat_list.append(cat)

            time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']  # negative difference
            all_times.append(time_lag)

            if current_sd_rec['next'] == '':
                break
            else:
                current_sd_rec = nusc.get('sample_data', current_sd_rec['next'])

    # Map the bounding boxes to the local sensor coordinate
    for box in boxes:
        if box is not None:
            # Move box to ego vehicle coord system
            box.translate(-np.array(ref_pose_rec['translation']))
            box.rotate(Quaternion(ref_pose_rec['rotation']).inverse)

            # Move box to sensor coord system
            box.translate(-np.array(ref_cs_rec['translation']))
            box.rotate(Quaternion(ref_cs_rec['rotation']).inverse)

            # caused by coordinate inconsistency of nuscene-toolkit
            box.center[0] = - box.center[0]

            # debug
            shift = [box.center[0], box.center[1], box.center[2]]
            box.translate(-np.array(shift))
            box.rotate(Quaternion([0, 1, 0, 0]).inverse)
            box.translate(np.array(shift))

        box_list.append(box)
    #print(temp)
    return box_list, all_times, attr_list, cat_list