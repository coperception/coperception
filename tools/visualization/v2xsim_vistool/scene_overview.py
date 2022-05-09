import os

import matplotlib
import matplotlib.patches as mpatches
import numpy as np
import simple_plot3d.canvas_3d as canvas_3d
from matplotlib import pyplot as plt

from simple_dataset import SimpleDataset
from tqdm import tqdm

COLOR = ['red','springgreen','dodgerblue', 'darkviolet', 'orange']
COLOR_RGB = [ tuple([int(cc * 255) for cc in matplotlib.colors.to_rgb(c)]) for c in COLOR]
COLOR_PC = [tuple([int(cc*0.2 + 255*0.8) for cc in c]) for c in COLOR_RGB]
CLASSES = ['agent1', 'agent2', 'agent3', 'agent4', 'agent5']

canvas_shape=(800, 1200)
camera_center_coords=(10, 32, 50)
camera_focus_coords=(10 , 32 + 0.8396926, 50 - 0.84202014)
focal_length = 400
left_hand = False
point_color = "Mixed"

def main():
    ## basic setting
    from_scene = 31
    to_scene = 33
    output_dir = './result_v2x'

    dataset = SimpleDataset(root_dir=f'./v2xsim2_info', from_scene=from_scene, to_scene=to_scene)

    ## matplotlib setting
    plt.figure()
    plt.style.use('dark_background')

    ## box setting
    ## ego coord
    dx = 4.5
    dy = 2
    dz = 1.6
    x_corners = dx / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])  # (8,)
    y_corners = dy / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
    z_corners = dz / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
    box_corners = np.stack((x_corners, y_corners, z_corners), axis=-1) # (8, 3)
    box_corners = np.pad(box_corners,((0,0),(0,1)), constant_values=1) # (8, 4)


    ## draw
    current_scene = from_scene - 1
    current_frame_idx = 0
    print(f"Start visualization. Results save to {output_dir}")
    dataset_len = len(dataset)
    for idx in tqdm(range(dataset_len)):
        if idx % 100 == 0:
            current_scene += 1
            current_frame_idx = 0

        base_data_dict = dataset[idx]
        cav_ids = list(base_data_dict.keys())
        cav_invert_dict = dict() # cav_id -> o/1/2
        for (idx, cav_id) in enumerate(cav_ids):
            cav_invert_dict[cav_id] = idx

        recs = []
        classes = []
        for i in range(0,len(cav_ids)):
            recs.append(mpatches.Rectangle((0,0),1,1,fc=COLOR[i]))
            classes.append(CLASSES[i])

        lidar_np_world_agg = np.zeros((0, 4))
        cav_box_agg = dict()
        cav_lidar_agg = dict()

        for cav_id, cav_content in base_data_dict.items():
            T_world_lidar = cav_content['params']['lidar_pose']  # [4,4]
            lidar_np_lidar = cav_content['lidar_np'] # [N, 4], ego coord
            lidar_np_lidar[:, 3] = 1
            lidar_np_world = (T_world_lidar @ lidar_np_lidar.T).T # [N, 4], world coord
            cav_lidar_agg[cav_id] = lidar_np_world
            lidar_np_world_agg = np.concatenate((lidar_np_world_agg, lidar_np_world), axis=0)

            # get bbox for each one.
            T_world_ego = T_world_lidar
            cav_box_agg[cav_id] = ((T_world_ego @ box_corners.T).T)[np.newaxis,:,:3] # (1,8,3)

        canvas = canvas_3d.Canvas_3D(canvas_shape, camera_center_coords, camera_focus_coords, focal_length, left_hand=left_hand) 
        # canvas_xy, valid_mask = canvas.get_canvas_coords(lidar_np_world_agg)
        # canvas.draw_canvas_points(canvas_xy[valid_mask], colors=COLOR_PC[cav_invert_dict[cav_id]])
        
        for cav_id in cav_ids:
            # draw point cloud for each cav
            canvas_xy, valid_mask = canvas.get_canvas_coords(cav_lidar_agg[cav_id])
            canvas.draw_canvas_points(canvas_xy[valid_mask], colors=COLOR_PC[cav_invert_dict[cav_id]])
            # draw bbox for each cav
            canvas.draw_boxes(cav_box_agg[cav_id], colors=COLOR_RGB[cav_invert_dict[cav_id]]) 

        plt.legend(recs,classes,loc='lower left')
        plt.axis("off")
        plt.imshow(canvas.canvas)
        plt.tight_layout()

        save_path = os.path.join(output_dir, f'scene_overview_{point_color}/{current_scene}')
        os.makedirs(save_path, exist_ok=True)

        plt.savefig(f"{save_path}/overview{current_frame_idx:02d}.png", transparent=False, dpi=500)
        plt.clf()
        current_frame_idx += 1

if __name__ == "__main__":
    main()
