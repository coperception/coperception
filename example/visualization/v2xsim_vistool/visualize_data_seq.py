"""
This file shows how to use simple_vis.py 

You should provide 

"""
from matplotlib import pyplot as plt
import matplotlib
import simple_plot3d.simple_vis as simple_vis
import numpy as np
import os
import copy
from pyquaternion import Quaternion
from simple_dataset import SimpleDataset
import torch

left_hand = False
vis_gt_box = True
vis_pred_box = True
lidar_range = (-32,-32,-3,32,32,2)
output_path = "./vis_seq"
root_dir = './v2xsim2_info/v2xsim_infos_vis[31-33].pkl'

def generate_object_corners_v2x(cav_contents,
                               reference_lidar_pose):
        """
        v2x-sim dataset

        Retrieve all objects (gt boxes)

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.
            In fact, only the ego vehile needs to generate object center

        reference_lidar_pose : list
            The final target lidar pose with length 6.

        Returns
        -------
        object_np : np.ndarray
            Shape is (n, 8, 3).
        object_ids : list
            Length is number of bbx in current sample.
        """
        # from opencood.data_utils.datasets import GT_RANGE
        max_num = 200
        gt_boxes = cav_contents[0]['params']['vehicles'] # notice [N,10], 10 includes [x,y,z,dx,dy,dz,w,a,b,c]
        object_ids = cav_contents[0]['params']['object_ids']
        
        object_dict = {"gt_boxes": gt_boxes, "object_ids":object_ids}

        output_dict = {}
        lidar_range = (-32,-32,-3,32,32,2)
        x_min, y_min, z_min, x_max, y_max, z_max = lidar_range
        
        gt_boxes = object_dict['gt_boxes']
        object_ids = object_dict['object_ids']
        for i, object_content in enumerate(gt_boxes):
            x,y,z,dx,dy,dz,w,a,b,c = object_content

            q = Quaternion([w,a,b,c])
            T_world_object = q.transformation_matrix
            T_world_object[:3,3] = object_content[:3]

            T_world_lidar = reference_lidar_pose

            object2lidar = np.linalg.solve(T_world_lidar, T_world_object) # T_lidar_object

            # shape (3, 8)
            x_corners = dx / 2 * np.array([ 1,  1, -1, -1,  1,  1, -1, -1]) # (8,)
            y_corners = dy / 2 * np.array([-1,  1,  1, -1, -1,  1,  1, -1])
            z_corners = dz / 2 * np.array([-1, -1, -1, -1,  1,  1,  1,  1])

            bbx = np.vstack((x_corners, y_corners, z_corners)) # (3, 8)

            # bounding box under ego coordinate shape (4, 8)
            bbx = np.r_[bbx, [np.ones(bbx.shape[1])]]

            # project the 8 corners to world coordinate
            bbx_lidar = np.dot(object2lidar, bbx).T # (8, 4)
            bbx_lidar = np.expand_dims(bbx_lidar[:, :3], 0) # (1, 8, 3)

            bbox_corner = copy.deepcopy(bbx_lidar)

            center = np.mean(bbox_corner, axis=1)[0]

            if (center[0] > x_min and center[0] < x_max and 
               center[1] > y_min and center[1] < y_max and 
               center[2] > z_min and center[2] < z_max) or i==3:
                output_dict.update({object_ids[i]: bbox_corner})

        object_np = np.zeros((max_num, 8, 3))
        object_ids = []

        for i, (object_id, object_bbx) in enumerate(output_dict.items()):
            object_np[i] = object_bbx[0, :]
            object_ids.append(object_id)

        # should not appear repeated items
        object_np = object_np[:len(object_ids)]

        return object_np, object_ids


def main():
    ## basic setting
    dataset = SimpleDataset(root_dir=root_dir)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    ## draw
    print("loop over dataset")
    dataset_len = len(dataset)
    for idx in range(dataset_len):
        print(idx)
        base_data_dict = dataset[idx]
        
        # only select agent1 to visualize
        cav_content = base_data_dict[1]

        ego_pose = cav_content['params']['lidar_pose']
        object_np, object_ids = generate_object_corners_v2x([cav_content], ego_pose)
        lidar_np_ego = cav_content['lidar_np']

        # Output of network is tensor of bounding boxes with shape [N, 8, 3]
        # We simply tranform numpy array to tensor for gt_boxes
        # Add noise to the box to simulate network predict

        gt_boxes_tensor = torch.from_numpy(object_np)
        pred_boxes_tensor = torch.clone(gt_boxes_tensor)
        pred_boxes_tensor[...,:2] += torch.randn(2) * 0.2
        point_cloud = torch.from_numpy(lidar_np_ego)

        vis_save_path = os.path.join(output_path, '3d_%05d.png' % idx)
        simple_vis.visualize(pred_boxes_tensor,
                            gt_boxes_tensor,
                            point_cloud,
                            lidar_range,
                            vis_save_path,
                            method='3d',
                            vis_gt_box = vis_gt_box,
                            vis_pred_box = vis_pred_box,
                            left_hand=left_hand)
            
        vis_save_path = os.path.join(output_path, 'bev_%05d.png' % idx)

        simple_vis.visualize(pred_boxes_tensor,
                            gt_boxes_tensor,
                            point_cloud,
                            lidar_range,
                            vis_save_path,
                            method='bev',
                            vis_gt_box = vis_gt_box,
                            vis_pred_box = vis_pred_box,
                            left_hand=left_hand)

        

if __name__ == "__main__":
    main()