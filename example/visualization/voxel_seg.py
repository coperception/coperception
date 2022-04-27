import matplotlib.pyplot as plt

from coperception.datasets import V2XSimDet, V2XSimSeg
from coperception.configs import Config, ConfigGlobal
from coperception.utils.obj_util import *

split = "train"
config = Config(binary=True, split=split, use_vis=True)

data_path = f"/scratch/dm4524/data/V2X-Sim-seg/{split}/agent0"

data_carscenes = V2XSimSeg(
    dataset_roots=[data_path],
    split=split,
    config=config,
    val=True,
    kd_flag=True,
    bound="both",
    no_cross_road=False,
)

padded_voxel_points, padded_voxel_points_teacher, seg_bev = data_carscenes[0][0]
padded_voxel_points = padded_voxel_points.cpu().detach().numpy()

plt.imshow(
    np.max(padded_voxel_points.reshape(256, 256, 13), axis=2), alpha=1.0, zorder=12
)
plt.savefig("./voxel")
