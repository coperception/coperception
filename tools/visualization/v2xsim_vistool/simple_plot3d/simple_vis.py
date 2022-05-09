from matplotlib import pyplot as plt
import numpy as np

from . import canvas_3d 
from . import canvas_bev

def visualize(pred_box_tensor, gt_tensor, pcd, pc_range, save_path, method='3d', vis_gt_box=True, vis_pred_box=True, left_hand=False):
        """
        Visualize the prediction, ground truth with point cloud together.
        They may be flipped in y axis. Since carla is left hand coordinate, while kitti is right hand.

        Parameters
        ----------
        pred_box_tensor : torch.Tensor
            (N, 8, 3) prediction.

        gt_tensor : torch.Tensor
            (N, 8, 3) groundtruth bbx

        pcd : torch.Tensor
            PointCloud, (N, 4).

        pc_range : list
            [xmin, ymin, zmin, xmax, ymax, zmax]

        save_path : str
            Save the visualization results to given path

        method: str, 'bev' or '3d'

        left_hand: The point cloud is left hand coordinate or right hand coordniate?
            V2X-Sim dataset is right-hand. OPV2V is right hand

        """

        pc_range = [int(i) for i in pc_range]
        pcd_np = pcd.cpu().numpy()

        if vis_pred_box:
            pred_box_np = pred_box_tensor.cpu().numpy()
            pred_name = ['pred'] * pred_box_np.shape[0]
        if vis_gt_box:
            gt_box_np = gt_tensor.cpu().numpy()
            gt_name = ['gt'] * gt_box_np.shape[0]

        if method == 'bev':
            canvas = canvas_bev.Canvas_BEV_heading_right(canvas_shape=((pc_range[4]-pc_range[1])*10, (pc_range[3]-pc_range[0])*10),
                                            canvas_x_range=(pc_range[0], pc_range[3]), 
                                            canvas_y_range=(pc_range[1], pc_range[4]),
                                            left_hand=left_hand
                                          ) 

            canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np) # Get Canvas Coords
            canvas.draw_canvas_points(canvas_xy[valid_mask]) # Only draw valid points
            if vis_pred_box:
                canvas.draw_boxes(pred_box_np, colors=(255,0,0), texts=pred_name)
            if vis_gt_box:
                canvas.draw_boxes(gt_box_np,colors=(0,255,0), texts=gt_name)

        elif method == '3d':
            canvas = canvas_3d.Canvas_3D(left_hand=left_hand)
            canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np)
            canvas.draw_canvas_points(canvas_xy[valid_mask])
            if vis_pred_box:
                canvas.draw_boxes(pred_box_np, colors=(255,0,0), texts=pred_name)
            if vis_gt_box:
                canvas.draw_boxes(gt_box_np,colors=(0,255,0), texts=gt_name)
        else:
            raise(f"Not Completed for f{method} visualization.")

        plt.axis("off")

        plt.imshow(canvas.canvas)

        plt.tight_layout()
        plt.savefig(save_path, transparent=False, dpi=400)
        plt.clf()
