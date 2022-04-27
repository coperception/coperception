import matplotlib.pyplot as plt

from coperception.datasets import V2XSimDet, V2XSimSeg
from coperception.configs import Config, ConfigGlobal
from coperception.utils.obj_util import *

split = "train"
config = Config(binary=True, split=split, use_vis=True)

data_path = f"/scratch/dm4524/data/V2X-Sim-seg/{split}/agent0"

v2x_seg_dataset = V2XSimSeg(
    dataset_roots=[data_path],
    split=split,
    config=config,
    val=True,
    kd_flag=True,
    bound="both",
    no_cross_road=False,
)

padded_voxel_points, padded_voxel_points_teacher, seg_bev = v2x_seg_dataset[0][0]
padded_voxel_points = padded_voxel_points.cpu().detach().numpy()

gt_image = np.zeros(shape=(256, 256, 3), dtype=np.dtype("uint8"))

# map colors according to our scheme
for key, value in config.class_to_rgb.items():
    gt_image[np.where(seg_bev == key)] = value

plt.imshow(gt_image)
plt.savefig("./voxel_seg_2")
